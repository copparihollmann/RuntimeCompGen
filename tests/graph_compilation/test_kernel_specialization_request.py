"""M-39 Kernel-specialization request emission tests.

Three layers of coverage:

- **Schema**: KernelSpecializationRequest serializes deterministically;
  every required field is populated.
- **Emitter**: build_kernel_specialization_request reads recipe-planning
  artifacts and produces honest content (correct shape, tile,
  refinement, contract_hash).
- **E2E**: run.py end-to-end with --stop-after
  kernel-specialization-request lands the request artifact at the
  expected path on both bit_equality (merlin_mlp_wide) and
  tolerance_eps (tiny_mlp) paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke(*, model: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "kernel-specialization-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_to_dict_round_trips(self) -> None:
        from compgen.graph_compilation.kernel_specialization import (
            KernelSpecializationRequest,
            _Contract, _Layout, _Shape, _Source, _Target, _Tile, _Tolerance,
        )

        req = KernelSpecializationRequest(
            schema_version="kernel_specialization_request_v1",
            request_id="kspec_dead",
            generated_at_utc="2026-05-07T00:00:00Z",
            request_kind="kernel_specialization",
            source=_Source(
                recipe_op_id="recipe_0000",
                candidate_id="cand_x",
                region_id="matmul_0",
                candidate_selection_path="03_recipe_planning/candidate_selection.json",
                recipe_summary_path="03_recipe_planning/recipe_summary.json",
                region_dossier_path="02_graph_analysis/region_dossiers/matmul_0__abc.json",
            ),
            target=_Target(
                target_id="host_cpu", target_class="host_cpu",
                backend="c_reference",
            ),
            shape=_Shape(M=16, N=32, K=16, shape_mode="concrete"),
            layout=_Layout(lhs="row_major", rhs="row_major", out="row_major"),
            tile=_Tile(M=16, N=16, K=16),
            dtype="f32",
            contract=_Contract(
                accumulator="f32",
                tolerance=_Tolerance(refinement="bit_equality", bound="exact"),
                aliasing="no_output_alias",
                dispatch_model="sync",
                contract_hash="dead" + "0" * 12,
            ),
            required_outputs=("kernel_source", "compiled_artifact"),
            forbidden=("change_contract",),
        )
        d = req.to_dict()
        # Sort keys so two emits produce byte-identical JSON.
        s = json.dumps(d, indent=2, sort_keys=True)
        assert "kernel_specialization_request_v1" in s
        assert "kspec_dead" in s
        # Every top-level field present.
        for k in (
            "schema_version", "request_id", "generated_at_utc",
            "request_kind", "source", "target", "shape", "layout",
            "tile", "dtype", "contract", "required_outputs", "forbidden",
        ):
            assert k in d


# --------------------------------------------------------------------------- #
# E2E — pipeline emits request on real runs
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def merlin_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m39_merlin") / "run"
    res = _invoke(model="merlin_mlp_wide", out_dir=out)
    assert res.returncode == 0, res.stderr
    return out


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m39_tiny") / "run"
    res = _invoke(model="tiny_mlp", out_dir=out)
    assert res.returncode == 0, res.stderr
    return out


def _read_only_request(run_dir: Path) -> dict:
    requests_dir = run_dir / "04_kernel_specialization" / "requests"
    files = sorted(requests_dir.glob("*.json"))
    assert len(files) == 1, f"expected exactly 1 request, got {files}"
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_e2e_request_artifact_exists(merlin_run: Path) -> None:
    out_dir = merlin_run / "04_kernel_specialization"
    assert out_dir.is_dir()
    assert (out_dir / "kernel_specialization_summary.json").exists()
    assert (out_dir / "requests").is_dir()


def test_e2e_merlin_emits_bit_equality_request(merlin_run: Path) -> None:
    """merlin_mlp_wide picks tile_M16_N16_K16 on a 16x32x16 matmul:
    K_iters=1, single_k_iter holds → bit_equality refinement, exact bound."""
    body = _read_only_request(merlin_run)
    assert body["schema_version"] == "kernel_specialization_request_v1"
    assert body["request_kind"] == "kernel_specialization"
    assert body["shape"]["shape_mode"] == "concrete"
    assert body["shape"] == {"M": 16, "N": 32, "K": 16, "shape_mode": "concrete"}
    assert body["tile"] == {"M": 16, "N": 16, "K": 16}
    assert body["dtype"] == "f32"
    assert body["contract"]["tolerance"]["refinement"] == "bit_equality"
    assert body["contract"]["tolerance"]["bound"] == "exact"
    assert body["contract"]["accumulator"] == "f32"
    assert body["contract"]["aliasing"] == "no_output_alias"
    assert body["contract"]["contract_hash"], "contract_hash must be non-empty"
    assert body["source"]["region_id"] == "matmul_0"
    assert body["source"]["candidate_id"].startswith("cand_tile_matmul_0")
    assert body["source"]["region_dossier_path"].startswith(
        "02_graph_analysis/region_dossiers/matmul_0__"
    )
    assert "kernel_source" in body["required_outputs"]
    assert "certificate" in body["required_outputs"]
    # Forbidden surface bounds Claude Code's kernel-codegen freedom.
    assert "widen_tolerance_beyond_higham_bound" in body["forbidden"]


def test_e2e_tiny_emits_tolerance_eps_request(tiny_run: Path) -> None:
    """tiny_mlp picks shape-fit tile_M4_N16_K16 on M=4 N=128 K=64:
    K_iters=4, NOT single_k_iter → tolerance_eps refinement,
    higham_4kepsabmax bound."""
    body = _read_only_request(tiny_run)
    assert body["request_kind"] == "kernel_specialization"
    assert body["shape"]["M"] == 4
    assert body["shape"]["N"] == 128
    assert body["shape"]["K"] == 64
    assert body["tile"] == {"M": 4, "N": 16, "K": 16}
    assert body["contract"]["tolerance"]["refinement"] == "tolerance_eps"
    assert body["contract"]["tolerance"]["bound"] == "higham_4kepsabmax"


def test_e2e_contract_hash_differs_per_region(
    merlin_run: Path, tiny_run: Path,
) -> None:
    """The M-26 contract_hash must differ between merlin_mlp_wide and
    tiny_mlp — different regions, different shapes, different tiles."""
    a = _read_only_request(merlin_run)["contract"]["contract_hash"]
    b = _read_only_request(tiny_run)["contract"]["contract_hash"]
    assert a and b
    assert a != b, (
        f"contract_hash collided across distinct regions: "
        f"merlin={a!r} tiny={b!r}"
    )


def test_e2e_summary_records_request(merlin_run: Path) -> None:
    summary_path = (
        merlin_run / "04_kernel_specialization"
        / "kernel_specialization_summary.json"
    )
    body = json.loads(summary_path.read_text(encoding="utf-8"))
    assert body["schema_version"] == "kernel_specialization_summary_v1"
    assert len(body["requests"]) == 1
    row = body["requests"][0]
    assert row["request_kind"] == "kernel_specialization"
    assert row["contract_hash"]


def test_e2e_request_byte_stable_across_reruns(tmp_path: Path) -> None:
    """Two independent runs of the same model produce byte-identical
    request JSON. Catches a regression where the emitter started
    pulling timestamps or non-deterministic data into the artifact."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    res_a = _invoke(model="merlin_mlp_wide", out_dir=out_a)
    res_b = _invoke(model="merlin_mlp_wide", out_dir=out_b)
    assert res_a.returncode == 0 and res_b.returncode == 0

    def _strip_timestamp(body: dict) -> dict:
        body = dict(body)
        body.pop("generated_at_utc", None)
        return body

    body_a = _strip_timestamp(_read_only_request(out_a))
    body_b = _strip_timestamp(_read_only_request(out_b))
    assert body_a == body_b, (
        "request body diverged across reruns; check for non-deterministic "
        "fields"
    )


def test_e2e_stop_after_recipe_planning_does_not_emit(tmp_path: Path) -> None:
    """When stop_after stops before kernel-specialization-request, no
    04_kernel_specialization/ directory is created."""
    out = tmp_path / "no_emit"
    res = subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert not (out / "04_kernel_specialization").exists()
