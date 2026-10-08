"""Tests for the four-way report builder."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.bench.four_way import report


def _sample_runs() -> list[dict[str, str | int | bool | None]]:
    """Synthetic runs.jsonl content covering the four pipelines on Gemmini.

    Cycles trajectory chosen so:
      * Best overall on gemmini is 40 cycles (pipeline C with med budget).
      * Pipeline A reaches within 10% of best at 80k tokens.
      * Pipeline C reaches it at 40k tokens — half the spend.
    """
    return [
        {"pipe": "A", "target": "gemmini", "region": "matmul_0",
         "budget": "small", "succeeded": True, "best_cycles": 90,
         "total_tokens": 40_000, "prompt_tokens": 30_000,
         "completion_tokens": 10_000, "constrained": False,
         "failure_reason": None},
        {"pipe": "A", "target": "gemmini", "region": "matmul_0",
         "budget": "med", "succeeded": True, "best_cycles": 44,
         "total_tokens": 80_000, "prompt_tokens": 60_000,
         "completion_tokens": 20_000, "constrained": False,
         "failure_reason": None},
        {"pipe": "C", "target": "gemmini", "region": "matmul_0",
         "budget": "small", "succeeded": True, "best_cycles": 42,
         "total_tokens": 40_000, "prompt_tokens": 30_000,
         "completion_tokens": 10_000, "constrained": True,
         "failure_reason": None},
        {"pipe": "C", "target": "gemmini", "region": "matmul_0",
         "budget": "med", "succeeded": True, "best_cycles": 40,
         "total_tokens": 80_000, "prompt_tokens": 60_000,
         "completion_tokens": 20_000, "constrained": True,
         "failure_reason": None},
        # Autocomp pipelines unsuccessful — must be tolerated.
        {"pipe": "B", "target": "gemmini", "region": "matmul_0",
         "budget": "small", "succeeded": False, "best_cycles": None,
         "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "constrained": False, "failure_reason": "autocomp_pending"},
        {"pipe": "D", "target": "gemmini", "region": "matmul_0",
         "budget": "small", "succeeded": False, "best_cycles": None,
         "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "constrained": True, "failure_reason": "autocomp_pending"},
    ]


def test_aggregate_by_pipe_target_summarizes_per_bucket(tmp_path: Path) -> None:
    rows = _sample_runs()
    buckets = report.aggregate_by_pipe_target(rows)
    # Four buckets seen (A,B,C,D × gemmini).
    assert ("A", "gemmini") in buckets
    assert ("C", "gemmini") in buckets
    a = buckets[("A", "gemmini")]
    c = buckets[("C", "gemmini")]
    assert a["best_cycles"] == 44
    assert c["best_cycles"] == 40
    assert a["success_rate"] == 1.0
    assert buckets[("B", "gemmini")]["success_rate"] == 0.0


def test_iso_perf_table_proves_C_reaches_iso_perf_with_fewer_tokens(tmp_path: Path) -> None:
    """The H1 invariant: pipeline C should hit iso-perf cheaper than pipeline A.

    Iso-perf threshold = best × 1.10 = 44 cycles. Pipeline A only reaches
    that at its 80k-token (med) cell; pipeline C reaches it at 40k tokens
    (small budget) — half the spend.
    """
    rows = _sample_runs()
    iso = report.iso_perf_table(rows)
    assert len(iso) == 1
    row = iso[0]
    assert row["target"] == "gemmini"
    assert row["best_cycles_any_pipeline"] == 40
    assert row["pipe_A_tokens_to_iso_perf"] == 80_000
    assert row["pipe_C_tokens_to_iso_perf"] == 40_000
    # H1: C used HALF the tokens of A to reach the same perf threshold.
    assert row["pipe_C_tokens_to_iso_perf"] < row["pipe_A_tokens_to_iso_perf"]


def test_iso_token_table_holds_budget_constant(tmp_path: Path) -> None:
    """The H2 invariant: at the same budget, pipeline C should be ≤ A."""
    rows = _sample_runs()
    iso = report.iso_token_table(rows)
    by_budget = {r["budget"]: r for r in iso}
    assert by_budget["small"]["pipe_A_best_cycles"] == 90
    assert by_budget["small"]["pipe_C_best_cycles"] == 42
    assert by_budget["med"]["pipe_A_best_cycles"] == 44
    assert by_budget["med"]["pipe_C_best_cycles"] == 40
    # H2: same budget, C is faster.
    assert by_budget["small"]["pipe_C_best_cycles"] < by_budget["small"]["pipe_A_best_cycles"]
    assert by_budget["med"]["pipe_C_best_cycles"] < by_budget["med"]["pipe_A_best_cycles"]


def test_build_report_writes_all_artifacts(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text("\n".join(json.dumps(r) for r in _sample_runs()))

    summary = report.build_report(tmp_path)

    assert (tmp_path / "ablation_table.md").exists()
    assert (tmp_path / "summary.json").exists()
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["total_cells"] == 6
    assert written["succeeded_cells"] == 4

    # The ablation markdown must mention the constraint-effect numbers.
    md = (tmp_path / "ablation_table.md").read_text()
    assert "H1" in md
    assert "H2" in md
    assert "40000" in md or "40,000" in md
    # Pipeline letters all referenced.
    assert "A: KB vanilla" not in md  # short letter form in the table header
    assert "| A |" in md or "A tokens" in md


def test_render_handles_empty_runs(tmp_path: Path) -> None:
    """Empty runs.jsonl must not crash — the bench may have aborted early."""
    iso_token = report.iso_token_table([])
    iso_perf = report.iso_perf_table([])
    md = report.render_ablation_md([], iso_token, iso_perf)
    assert "H1 can't be evaluated" in md or "No successful cells" in md


def test_build_report_raises_when_runs_missing(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(FileNotFoundError):
        report.build_report(tmp_path)
