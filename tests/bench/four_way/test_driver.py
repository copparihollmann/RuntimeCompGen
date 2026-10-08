"""Tests for the four_way sweep driver.

The driver wraps three concerns: (1) the cartesian-product expansion,
(2) the budget guard, (3) the runs.jsonl persistence. Tests cover all
three using in-process mode (``workers=1``) so we don't fork and so
monkeypatching the runners is reliable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from compgen.bench.four_way import driver
from compgen.bench.four_way.spec import PipeId, PipelineResult, PipelineSpec, TargetId


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path / "tracker"))
    monkeypatch.setenv("COMPGEN_REPO_ROOT", str(tmp_path))
    return tmp_path


def _seed_bundle(bundle: Path) -> None:
    """Write a minimal bundle the driver can read."""
    bundle.mkdir(parents=True)
    (bundle / "kernel_contracts").mkdir()
    (bundle / "graph_dossier_v3.json").write_text(json.dumps({
        "regions": [
            {
                "region_id": "matmul_0",
                "op_family": "matmul", "dtype": "int8",
                "abstracted_shape": {"M": 64, "N": 64, "K": 64},
            },
            {
                "region_id": "matmul_1",
                "op_family": "matmul", "dtype": "int8",
                "abstracted_shape": {"M": 32, "N": 32, "K": 32},
            },
        ]
    }))


def _stub_runner(monkeypatch: pytest.MonkeyPatch, *, cycles: int = 100,
                 prompt_tokens: int = 1000, completion_tokens: int = 200) -> None:
    """Replace both runner adapters with cheap stubs that return success."""

    def fake_kb(spec: PipelineSpec) -> PipelineResult:
        return PipelineResult(
            spec=spec,
            succeeded=True,
            best_kernel_source="// k",
            best_cycles=cycles,
            total_iterations=3,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            n_llm_calls=3,
            wall_seconds=0.1,
        )

    def fake_autocomp(spec: PipelineSpec) -> PipelineResult:
        return PipelineResult(
            spec=spec,
            succeeded=False,
            best_kernel_source=None,
            best_cycles=None,
            total_iterations=0,
            prompt_tokens=0,
            completion_tokens=0,
            n_llm_calls=0,
            wall_seconds=0.0,
            failure_reason="autocomp_custom_problem_injection_pending",
        )

    # The driver imports the modules at top level, so we monkeypatch
    # the function attributes on those modules.
    import compgen.bench.four_way.runner_autocomp as ra
    import compgen.bench.four_way.runner_kernelblaster as rk

    monkeypatch.setattr(rk, "run_kernelblaster_pipeline", fake_kb)
    monkeypatch.setattr(ra, "run_autocomp_pipeline", fake_autocomp)


def test_build_sweep_specs_enumerates_full_grid(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)

    specs = driver.build_sweep_specs(
        bundle_dir=bundle,
        pipelines=[PipeId.A, PipeId.C],
        targets=[TargetId.GEMMINI],
        regions=["matmul_0", "matmul_1"],
        budgets=["small", "med"],
        model="gemini-2.5-flash",
        out_root=tmp_path / "out",
        run_id="exp_test",
    )

    # 2 pipelines × 1 target × 2 regions × 2 budgets = 8 cells
    assert len(specs) == 8
    # Pipeline A always sees None; pipeline C always sees a payload.
    for s in specs:
        if s.pipe is PipeId.A:
            assert s.constraint_artifacts is None
        if s.pipe is PipeId.C:
            assert s.constraint_artifacts is not None
            assert s.constraint_artifacts["region_id"] == s.region_id
    # The auto-derived problem_spec contains the matmul shape.
    assert any("M=64" in s.problem_spec for s in specs)
    assert any("M=32" in s.problem_spec for s in specs)


def test_run_sweep_writes_one_row_per_cell_to_jsonl(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)
    _stub_runner(monkeypatch)

    specs = driver.build_sweep_specs(
        bundle_dir=bundle,
        pipelines=[PipeId.A, PipeId.B, PipeId.C, PipeId.D],
        targets=[TargetId.GEMMINI],
        regions=["matmul_0"],
        budgets=["small"],
        model="gemini-2.5-flash",
        out_root=tmp_path / "out",
        run_id="exp_test",
    )
    rows = driver.run_sweep(
        specs=specs,
        out_dir=tmp_path / "out",
        workers=1,
        enforce_budget_guard=False,
    )

    assert len(rows) == 4
    # KB pipelines succeed, autocomp pipelines surface the typed
    # pending failure honestly.
    kb_rows = [r for r in rows if r["pipe"] in {"A", "C"}]
    ac_rows = [r for r in rows if r["pipe"] in {"B", "D"}]
    assert all(r["succeeded"] for r in kb_rows)
    assert all(not r["succeeded"] for r in ac_rows)
    assert all(
        "autocomp_custom_problem_injection_pending" in (r["failure_reason"] or "")
        for r in ac_rows
    )

    # Persisted to runs.jsonl correctly.
    written = [
        json.loads(line)
        for line in (tmp_path / "out" / "runs.jsonl").read_text().splitlines()
        if line
    ]
    assert len(written) == 4
    assert {r["pipe"] for r in written} == {"A", "B", "C", "D"}


def test_budget_guard_stops_remaining_cells(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When current spend trips the hard threshold, the next cell short-circuits."""
    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)
    _stub_runner(monkeypatch)

    # Set a tiny cap, then seed a fake event that already burned 100% of it.
    from compgen.observability.gemini_usage import Budget, record_call

    b = Budget()
    b.cumulative_usd = 0.10  # cents-level so we know the guard trips
    b.save()
    # 1M gemini-2.5-flash output tokens = $2.50 → way over the $0.10 cap.
    record_call("gemini-2.5-flash", 0, 1_000_000, source="seed")

    specs = driver.build_sweep_specs(
        bundle_dir=bundle,
        pipelines=[PipeId.A, PipeId.C],
        targets=[TargetId.GEMMINI],
        regions=["matmul_0"],
        budgets=["small"],
        model="gemini-2.5-flash",
        out_root=tmp_path / "out",
        run_id="exp_test",
    )
    rows = driver.run_sweep(
        specs=specs,
        out_dir=tmp_path / "out",
        workers=1,
        enforce_budget_guard=True,
    )

    # Every cell must have been short-circuited with budget_guard.
    assert len(rows) == 2
    for r in rows:
        assert r["succeeded"] is False
        assert r["failure_reason"].startswith("budget_guard")


def test_spec_dict_roundtrip_preserves_constrained_flag(tmp_path: Path) -> None:
    spec = PipelineSpec(
        pipe=PipeId.D,
        target=TargetId.SATURN,
        region_id="matmul_3",
        problem_spec="int8 matmul",
        constraint_artifacts={"region_id": "matmul_3"},
        search_budget="med",
        model="gemini-2.5-flash",
        work_dir=tmp_path,
        extra_metadata={"run_id": "x"},
    )
    sd = driver._spec_to_dict(spec)
    assert sd["constrained"] is True
    rt = driver._spec_from_dict(sd)
    assert rt.pipe is PipeId.D
    assert rt.target is TargetId.SATURN
    assert rt.constraint_artifacts == {"region_id": "matmul_3"}


def test_typed_failure_row_carries_traceback(tmp_path: Path) -> None:
    sd = {"pipe": "pipe=A", "target": "gemmini", "region_id": "r0",
          "search_budget": "small", "model": "x", "constrained": False}
    try:
        raise RuntimeError("simulated explosion")
    except RuntimeError as exc:
        row = driver._typed_failure_row(sd, exc)
    assert row["succeeded"] is False
    assert "simulated explosion" in row["failure_reason"]
    assert "traceback" in row["extra"]
