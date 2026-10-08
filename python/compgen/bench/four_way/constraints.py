"""Slice CompGen's dossier + kernel contracts into per-region payloads.

Pipelines C and D consume a compact dict per (region × target) that
the optimizer sees as "constraint artifacts": the dossier row for
that region, the matching ``kernel_contracts/<region>.yaml``, plus
the readiness-report row (precision budget, working-set fit).

This module is the only place that knows how to read those artifacts —
the runner adapters are dossier-shape-agnostic and just pass the
returned dict through.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RegionConstraints:
    """The constraint payload one region hands to pipelines C / D.

    Attributes:
        region_id: Stable id (``"matmul_0"`` ...).
        dossier: The dossier row for the region — typically op_family,
            dtype, abstracted_shape, hot_axis.
        contract: The kernel-contract YAML loaded into a dict — layouts,
            dtypes, tile_params, aliasing.
        readiness: Readiness-report row (precision budget, working-set
            fit). May be ``None`` if not yet emitted.
        problem_spec: A short prose description the runner can hand
            straight to the LLM. Pre-rendered so each runner doesn't
            need to re-derive it.
    """

    region_id: str
    dossier: dict[str, Any]
    contract: dict[str, Any]
    readiness: dict[str, Any] | None = None
    problem_spec: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Shape the dict that ``ChipyardConfig.constraint_artifacts`` accepts."""
        return {
            "region_id": self.region_id,
            "dossier": self.dossier,
            "contract": self.contract,
            "readiness": self.readiness,
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    """Best-effort YAML/JSON loader.

    Kernel contracts ship as YAML; the dossier ships as JSON. We
    tolerate either extension without forcing PyYAML on every caller —
    if PyYAML isn't installed and the file isn't JSON we fall back to a
    text dict carrying just the raw contents.
    """
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "constraints.yaml_unavailable",
            path=str(path),
            hint="install [solve] or [llm] extra to get PyYAML",
        )
        return {"_raw": text}
    return yaml.safe_load(text)


def _select_dossier_row(dossier: dict[str, Any], region_id: str) -> dict[str, Any]:
    """Find the dossier row for ``region_id``.

    ``graph_dossier_v3.json`` is shaped as ``{"regions": [{"region_id":
    "...", ...}, ...]}``. We tolerate ``{"region_id": ...}`` dicts and
    bare lists too — whichever format the current dossier emitter is
    using.
    """
    candidates: list[dict[str, Any]] = []
    if isinstance(dossier, dict):
        for key in ("regions", "region_facts", "matmul_regions", "rows"):
            if key in dossier and isinstance(dossier[key], list):
                candidates = dossier[key]
                break
        if not candidates and "region_id" in dossier:
            candidates = [dossier]
    elif isinstance(dossier, list):
        candidates = dossier

    for row in candidates:
        if not isinstance(row, dict):
            continue
        if row.get("region_id") == region_id or row.get("id") == region_id:
            return row
    # No row matched — return an honest empty dict so the prompt block
    # is rendered without inventing facts.
    logger.warning("constraints.region_not_in_dossier", region_id=region_id)
    return {"region_id": region_id, "_note": "region not found in dossier"}


def build_region_constraints(
    *,
    bundle_dir: Path,
    region_id: str,
    problem_spec: str = "",
) -> RegionConstraints:
    """Load + slice the constraint artifacts for one region.

    Args:
        bundle_dir: Output directory of CompGen's compile (where
            ``graph_dossier_v3.json``, ``kernel_contracts/``, and the
            readiness reports live).
        region_id: Region to slice for. The runner asks for one at a time.
        problem_spec: Optional pre-rendered problem description. If
            empty, a minimal one is auto-derived from the dossier row.

    Returns:
        A :class:`RegionConstraints`. Caller calls ``.to_payload()`` to
        get the dict shaped for ``ChipyardConfig.constraint_artifacts``.

    Raises:
        FileNotFoundError: If the dossier file is missing — the caller
            must compile first.
    """
    bundle_dir = Path(bundle_dir)
    dossier_path = bundle_dir / "graph_dossier_v3.json"
    if not dossier_path.exists():
        # Some bundles emit it one directory up under 02_graph_analysis/.
        alt = bundle_dir / "02_graph_analysis" / "graph_dossier_v3.json"
        if alt.exists():
            dossier_path = alt
    if not dossier_path.exists():
        raise FileNotFoundError(
            f"graph_dossier_v3.json not found under {bundle_dir}. "
            "Did you run compgen.quantization.smolvla_int8 first?"
        )

    full_dossier = _read_json(dossier_path)
    dossier_row = _select_dossier_row(full_dossier, region_id)

    # Contract: kernel_contracts/<region>.yaml — be permissive about
    # extension since some emitters use .json.
    contracts_dir = bundle_dir / "kernel_contracts"
    contract: dict[str, Any] = {}
    if contracts_dir.exists():
        for suffix in (".yaml", ".yml", ".json"):
            candidate = contracts_dir / f"{region_id}{suffix}"
            if candidate.exists():
                contract = _read_yaml_or_json(candidate)
                break
        else:
            logger.info(
                "constraints.no_contract_for_region",
                region_id=region_id,
                contracts_dir=str(contracts_dir),
            )

    # Readiness: try a couple of known names — the readiness reports
    # are stored under different names depending on the milestone.
    readiness: dict[str, Any] | None = None
    for cand in (
        bundle_dir / "graph_analysis_readiness_matrix.json",
        bundle_dir / "02_graph_analysis" / "graph_analysis_readiness_matrix.json",
        bundle_dir / "precision_budget_report.json",
    ):
        if cand.exists():
            try:
                readiness = _read_json(cand)
                break
            except json.JSONDecodeError:
                pass

    # Auto-derive a problem spec if the caller didn't supply one.
    if not problem_spec:
        problem_spec = _default_problem_spec(dossier_row)

    return RegionConstraints(
        region_id=region_id,
        dossier=dossier_row,
        contract=contract,
        readiness=readiness,
        problem_spec=problem_spec,
    )


def _default_problem_spec(dossier_row: dict[str, Any]) -> str:
    """Render a minimal prose description from the dossier row.

    Caller can override by passing ``problem_spec`` to
    :func:`build_region_constraints`. Both autocomp and KB consume
    this string as the user-facing problem statement.
    """
    op = dossier_row.get("op_family") or dossier_row.get("op") or "matmul"
    dtype = dossier_row.get("dtype") or "int8"
    shape = dossier_row.get("abstracted_shape") or dossier_row.get("shape") or {}
    if op == "matmul":
        m = shape.get("M") or shape.get("m") or "?"
        n = shape.get("N") or shape.get("n") or "?"
        k = shape.get("K") or shape.get("k") or "?"
        return (
            f"{dtype} matmul M={m} N={n} K={k}; output int32. "
            "Minimize cycles measured by the bare-metal mcycle harness."
        )
    return f"{dtype} {op} kernel; shape={shape}. Minimize cycles."
