"""Kernel-specialization request emission (M-39).

Section 21 — Shape-specialized kernel generation. M-39 is the
data-only stage: convert a selected Recipe IR decision + region
facts into a concrete ``KernelSpecializationRequest``. No codegen
fires here — that lands in M-40 (Triton emitter) and M-41 (C
reference emitter). The request is the bounded contract Claude Code
or any other kernel-codegen provider sees.

What's bounded in the request:

- Concrete shape (M, N, K from the captured model) — this is M-39's
  ``shape_mode = "concrete"``. Shape-class and dynamic-guarded modes
  are M-45.
- Tile geometry (from the recipe op).
- Layout, dtype, accumulator, tolerance — derived from the region
  dossier + recipe_gate verdict.
- Contract hash (M-26): the same exact-kernel cache key the
  promotion library uses, so M-44's kernel cache reuses the same
  index without inventing a new scheme.
- Forbidden-mutation list: things the kernel codegen MUST NOT change
  (contract, tolerance, layout) — Claude Code's bounded surface.

Pipeline placement: ``run.py`` calls
:func:`run_kernel_specialization_request` after recipe_planning when
``stop_after >= "kernel-specialization-request"``. M-39 always
operates on the **selected** candidate; non-applicable kinds (today
that's anything except ``set_tile_params``) emit a typed
``not_applicable`` request rather than a kernel one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


_SCHEMA_VERSION = "kernel_specialization_request_v1"


@dataclass(frozen=True)
class _Shape:
    M: int
    N: int
    K: int
    shape_mode: str  # "concrete" | "shape_class" | "dynamic_guarded"

    def to_dict(self) -> dict[str, Any]:
        return {"M": self.M, "N": self.N, "K": self.K,
                "shape_mode": self.shape_mode}


@dataclass(frozen=True)
class _Layout:
    lhs: str  # row_major | col_major
    rhs: str
    out: str

    def to_dict(self) -> dict[str, Any]:
        return {"lhs": self.lhs, "rhs": self.rhs, "out": self.out}


@dataclass(frozen=True)
class _Tile:
    M: int
    N: int
    K: int

    def to_dict(self) -> dict[str, Any]:
        return {"M": self.M, "N": self.N, "K": self.K}


@dataclass(frozen=True)
class _Tolerance:
    refinement: str  # bit_equality | tolerance_eps | unknown
    bound: str       # named bound: "exact" | "higham_4kepsabmax"

    def to_dict(self) -> dict[str, Any]:
        return {"refinement": self.refinement, "bound": self.bound}


@dataclass(frozen=True)
class _Contract:
    accumulator: str           # f32 typically
    tolerance: _Tolerance
    aliasing: str              # no_output_alias | aliasing_allowed
    dispatch_model: str        # sync | async
    contract_hash: str         # M-26 exact-kernel cache key

    def to_dict(self) -> dict[str, Any]:
        return {
            "accumulator": self.accumulator,
            "tolerance": self.tolerance.to_dict(),
            "aliasing": self.aliasing,
            "dispatch_model": self.dispatch_model,
            "contract_hash": self.contract_hash,
        }


@dataclass(frozen=True)
class _Source:
    """Pointers to the producing artifacts so the kernel-codegen
    provider can audit *exactly* which decisions produced this
    request. All paths are relative to the run dir."""

    recipe_op_id: str          # "recipe_0000"
    candidate_id: str
    region_id: str
    candidate_selection_path: str
    recipe_summary_path: str
    region_dossier_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Target:
    target_id: str
    target_class: str
    backend: str  # triton | c_reference | accel_dialect

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KernelSpecializationRequest:
    """Bounded specification for a shape-specialized kernel.

    M-39 emits this. M-40+ consume it. The agent / kernel-codegen
    provider is allowed to choose the *implementation strategy*
    (template, autocomp-style search, provider/cache hit, reference
    fallback) but every field below is a hard constraint —
    re-deriving them, mutating tolerance, ignoring layout, or
    inventing new shape classes are all explicitly forbidden."""

    schema_version: str
    request_id: str
    generated_at_utc: str
    request_kind: str  # "kernel_specialization" | "not_applicable"

    # Sources + target.
    source: _Source
    target: _Target

    # Specialization.
    shape: _Shape
    layout: _Layout
    tile: _Tile
    dtype: str  # f32 only in M-39

    # Contract.
    contract: _Contract

    # Required outputs the codegen MUST produce.
    required_outputs: tuple[str, ...]

    # Forbidden mutations (bounded surface for Claude Code).
    forbidden: tuple[str, ...]

    # Optional reason if request_kind == "not_applicable".
    not_applicable_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "generated_at_utc": self.generated_at_utc,
            "request_kind": self.request_kind,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "shape": self.shape.to_dict(),
            "layout": self.layout.to_dict(),
            "tile": self.tile.to_dict(),
            "dtype": self.dtype,
            "contract": self.contract.to_dict(),
            "required_outputs": list(self.required_outputs),
            "forbidden": list(self.forbidden),
            "not_applicable_reason": self.not_applicable_reason,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _request_id(*, candidate_id: str, region_id: str) -> str:
    """Deterministic request id derived from the candidate + region."""
    raw = f"{region_id}|{candidate_id}".encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:8]
    return f"kspec_{h}"


def _resolve_region_dossier(run_dir: Path, region_id: str) -> Path | None:
    """Locate the region's dossier JSON. The dossier lives at
    ``02_graph_analysis/region_dossiers/<region_id>__<hash>.json`` —
    glob matches the prefix, picks the lexicographically-first to
    stay deterministic."""
    rd_dir = run_dir / "02_graph_analysis" / "region_dossiers"
    if rd_dir.is_dir():
        # Exact match first (no hash suffix).
        exact = rd_dir / f"{region_id}.json"
        if exact.exists():
            return exact
        # Prefix match (region_id followed by ``__<hash>``).
        prefix_matches = sorted(rd_dir.glob(f"{region_id}__*.json"))
        if prefix_matches:
            return prefix_matches[0]
    # Legacy flat layout fallback.
    legacy = run_dir / "02_graph_analysis" / f"region_dossier__{region_id}.json"
    if legacy.exists():
        return legacy
    return None


# --------------------------------------------------------------------------- #
# Public emitter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelSpecializationResult:
    out_dir: Path
    request_path: Path
    request_id: str
    request_kind: str  # "kernel_specialization" | "not_applicable"
    overall: str       # "pass" | "skipped"


def build_kernel_specialization_request(
    run_dir: Path,
) -> KernelSpecializationRequest:
    """Build the request from on-disk recipe-planning artifacts.

    Pure function — reads only on-disk state, constructs the dataclass.
    The caller (``run_kernel_specialization_request``) writes it to
    disk + records the stage."""
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"

    sel = _read_json(rp / "candidate_selection.json")
    summary = _read_json(rp / "recipe_summary.json")

    candidate_kind = sel.get("candidate_kind", "")
    candidate_id = sel.get("selected_candidate_id", "") or ""
    region_id = sel.get("region_id", "") or ""
    target_id = summary.get("target_id", "") or ""
    request_id = _request_id(candidate_id=candidate_id, region_id=region_id)

    source = _Source(
        recipe_op_id="recipe_0000",
        candidate_id=candidate_id,
        region_id=region_id,
        candidate_selection_path=str(
            (rp / "candidate_selection.json").relative_to(run_dir)
        ),
        recipe_summary_path=str(
            (rp / "recipe_summary.json").relative_to(run_dir)
        ),
        region_dossier_path=str(
            (_resolve_region_dossier(run_dir, region_id) or rp / "candidate_selection.json")
            .relative_to(run_dir)
        ),
    )

    # Non-set-tile-params recipes are out-of-scope for M-39. Emit a
    # typed ``not_applicable`` request so the pipeline records the
    # honest skip; M-40+ will widen the supported recipe kinds.
    if candidate_kind != "set_tile_params":
        return KernelSpecializationRequest(
            schema_version=_SCHEMA_VERSION,
            request_id=request_id,
            generated_at_utc=_utcnow(),
            request_kind="not_applicable",
            source=source,
            target=_Target(target_id=target_id, target_class="unknown",
                           backend="none"),
            shape=_Shape(M=0, N=0, K=0, shape_mode="concrete"),
            layout=_Layout(lhs="row_major", rhs="row_major", out="row_major"),
            tile=_Tile(M=0, N=0, K=0),
            dtype="unknown",
            contract=_Contract(
                accumulator="unknown",
                tolerance=_Tolerance(refinement="unknown", bound="unknown"),
                aliasing="unknown",
                dispatch_model="unknown",
                contract_hash="",
            ),
            required_outputs=(),
            forbidden=(),
            not_applicable_reason=(
                f"M-39 supports only set_tile_params; got "
                f"candidate_kind={candidate_kind!r}"
            ),
        )

    cost_preview = sel.get("cost_preview") or {}
    region_dims = cost_preview.get("region_dims") or {}
    M_dim = int(region_dims.get("M", 0) or 0)
    N_dim = int(region_dims.get("N", 0) or 0)
    K_dim = int(region_dims.get("K", 0) or 0)

    # Tile is encoded in the candidate label like "tile_M4_N16_K16".
    label = sel.get("label", "") or ""
    import re as _re
    m = _re.search(r"tile_M(\d+)_N(\d+)_K(\d+)", label)
    if not m:
        # Fall back to recipe_delta attrs.
        tile_M = tile_N = tile_K = 0
        for op in sel.get("recipe_delta") or []:
            tile_M = int(op.get("M", 0) or 0)
            tile_N = int(op.get("N", 0) or 0)
            tile_K = int(op.get("K", 0) or 0)
    else:
        tile_M, tile_N, tile_K = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Refinement: the recipe gate's M-37.13 single_k_iter rule
    # decides whether bit_equality or tolerance_eps is the claim. We
    # read the verdict (recipe_gate_verdict.json) when available;
    # otherwise derive from cost_preview.clean_divide + tile_K vs K.
    verdict = _read_json_or_none(rp / "recipe_gate_verdict.json")
    declared_refinement = "unknown"
    if verdict:
        for op in verdict.get("checked_recipe_ops") or []:
            if op.get("source_candidate") == candidate_id:
                declared_refinement = op.get("declared_refinement", "unknown")
                break
    if declared_refinement == "unknown":
        clean = cost_preview.get("clean_divide")
        single_k = (
            tile_K > 0 and K_dim > 0 and tile_K >= K_dim
        )
        if clean is True and single_k:
            declared_refinement = "bit_equality"
        elif clean is True or clean is False:
            declared_refinement = "tolerance_eps"

    tolerance_bound = (
        "exact" if declared_refinement == "bit_equality"
        else "higham_4kepsabmax" if declared_refinement == "tolerance_eps"
        else "unknown"
    )

    # Region dossier facts: dtype, layout, target_class.
    dossier_path = _resolve_region_dossier(run_dir, region_id)
    dossier = _read_json(dossier_path) if dossier_path else {}
    dtype = dossier.get("dtype", "f32") or "f32"
    layout_summary = dossier.get("layout") or {}
    layout = _Layout(
        lhs=str(layout_summary.get("lhs", "row_major")),
        rhs=str(layout_summary.get("rhs", "row_major")),
        out=str(layout_summary.get("out", "row_major")),
    )
    target_class = str(dossier.get("target_class") or target_id or "host_cpu")

    # M-26 contract_hash. Reuse the existing derivation so M-44's
    # cache uses the same index.
    from compgen.graph_compilation.promotion_bridge import (
        derive_contract_hash, derive_region_signature,
    )

    region_signature_fields: dict[str, str]
    try:
        _sig, region_signature_fields = derive_region_signature(
            run_dir=run_dir, region_id=region_id, target_id=target_id,
            kind=candidate_kind,
        )
    except Exception:  # noqa: BLE001 — degrade with explicit fields
        region_signature_fields = {
            "op_family": "matmul", "dtype": dtype,
            "layout": "row_major",
            "shape_class": f"{M_dim}x{N_dim}x{K_dim}",
            "target_class": target_class,
        }
    contract_hash = derive_contract_hash(
        candidate_selection=sel,
        region_signature_fields=region_signature_fields,
    )

    backend = "triton" if "cuda" in target_class else "c_reference"

    return KernelSpecializationRequest(
        schema_version=_SCHEMA_VERSION,
        request_id=request_id,
        generated_at_utc=_utcnow(),
        request_kind="kernel_specialization",
        source=source,
        target=_Target(
            target_id=target_id,
            target_class=target_class,
            backend=backend,
        ),
        shape=_Shape(M=M_dim, N=N_dim, K=K_dim, shape_mode="concrete"),
        layout=layout,
        tile=_Tile(M=tile_M, N=tile_N, K=tile_K),
        dtype=dtype,
        contract=_Contract(
            accumulator="f32",
            tolerance=_Tolerance(
                refinement=declared_refinement,
                bound=tolerance_bound,
            ),
            aliasing="no_output_alias",
            dispatch_model="async" if backend == "triton" else "sync",
            contract_hash=contract_hash,
        ),
        required_outputs=(
            "kernel_source",
            "compiled_artifact",
            "launch_metadata",
            "differential_report",
            "benchmark_report",
            "certificate",
        ),
        forbidden=(
            "change_contract",
            "change_tolerance",
            "ignore_layout",
            "emit_unverified_kernel",
            "invent_shape_class",
            "widen_tolerance_beyond_higham_bound",
        ),
    )


def run_kernel_specialization_request(
    run_dir: Path,
) -> KernelSpecializationResult:
    """Pipeline-stage entry point. Build the request and write it to
    ``04_kernel_specialization/requests/<request_id>.json``."""
    run_dir = Path(run_dir).resolve()
    request = build_kernel_specialization_request(run_dir)

    out_dir = run_dir / "04_kernel_specialization"
    requests_dir = out_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)

    request_path = requests_dir / f"{request.request_id}.json"
    request_path.write_text(
        json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_path = out_dir / "kernel_specialization_summary.json"
    summary = {
        "schema_version": "kernel_specialization_summary_v1",
        "generated_at_utc": _utcnow(),
        "requests": [
            {
                "request_id": request.request_id,
                "request_kind": request.request_kind,
                "candidate_id": request.source.candidate_id,
                "region_id": request.source.region_id,
                "shape": request.shape.to_dict(),
                "tile": request.tile.to_dict(),
                "contract_hash": request.contract.contract_hash,
                "not_applicable_reason": request.not_applicable_reason,
            }
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return KernelSpecializationResult(
        out_dir=out_dir,
        request_path=request_path,
        request_id=request.request_id,
        request_kind=request.request_kind,
        overall="pass" if request.request_kind == "kernel_specialization"
                else "skipped",
    )
