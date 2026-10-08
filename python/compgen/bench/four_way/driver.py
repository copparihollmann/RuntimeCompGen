"""Sweep driver: fan PipelineSpecs out across pipelines × targets × regions × budgets.

Each cell is one ``PipelineSpec``; the driver picks the right runner
adapter (KB or autocomp) and persists ``PipelineResult`` rows to a
``runs.jsonl`` the reporter consumes.

Concurrency model: ``ProcessPoolExecutor`` with one worker per
pipeline letter by default. Inside a worker the per-target runs are
sequential to avoid Spike subprocess contention. The default
``--workers 4`` puts pipelines A/B/C/D each in their own process so
the live ``compgen-gemini-usage watch --group-by pipe`` view actually
shows four independent streams.

Budget guard: before each cell starts, the driver evaluates current
spend against the configured cumulative cap. If spend ≥ 95% of the
cap the remaining cells are dropped with ``failure_reason="budget_guard"``,
so the operator's $200 cap doesn't blow past silently.

Run from the CLI:

    uv run python -m compgen.bench.four_way.driver \\
        --bundle artifacts/smolvla_int8 \\
        --regions matmul_0,matmul_1 \\
        --pipelines A,C \\
        --targets gemmini \\
        --budgets small,med \\
        --out results/four_way/$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import structlog

from compgen.bench.four_way.constraints import build_region_constraints
from compgen.bench.four_way.spec import (
    PipeId,
    PipelineResult,
    PipelineSpec,
    TargetId,
)

logger = structlog.get_logger()


# --------------------------------------------------------------------- #
# Budget guard
# --------------------------------------------------------------------- #


def _current_cumulative_usd() -> float:
    """Read the live cumulative spend from CompGen's tracker."""
    try:
        from compgen.observability.gemini_usage import load_summary
    except ImportError:
        return 0.0
    return float(load_summary().total_cost_usd)


def _cumulative_budget_usd() -> float | None:
    try:
        from compgen.observability.gemini_usage import Budget
    except ImportError:
        return None
    return Budget.load().cumulative_usd


def _budget_guard(soft_warn_pct: float = 0.80, hard_stop_pct: float = 0.95) -> str | None:
    """Return a string reason if the driver should bail, else ``None``.

    ``hard_stop_pct`` defaults to 95% of the cumulative cap. The
    last 5% is reserved so the in-flight cells the bench already
    queued can finish without busting the limit.
    """
    cap = _cumulative_budget_usd()
    if cap is None or cap <= 0:
        return None
    used = _current_cumulative_usd()
    if used >= cap * hard_stop_pct:
        return (
            f"budget_guard: cumulative spend ${used:.2f} >= "
            f"{hard_stop_pct * 100:.0f}% of ${cap:.2f} cap"
        )
    if used >= cap * soft_warn_pct:
        logger.warning(
            "four_way.budget_soft_warn",
            used_usd=used,
            cap_usd=cap,
            pct=round(used / cap * 100, 2),
        )
    return None


# --------------------------------------------------------------------- #
# Worker entry — kept top-level for ProcessPoolExecutor picklability
# --------------------------------------------------------------------- #


def _run_one_cell(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Worker entry. Reconstructs a PipelineSpec, picks the runner, runs.

    Returns the ``PipelineResult.to_dict()`` payload. Exceptions are
    caught and converted to a typed failure_reason so a single broken
    cell never kills the whole sweep.
    """
    # Defer heavy imports into the worker so the parent process stays cheap.
    import compgen.bench.four_way.runner_autocomp as ra
    import compgen.bench.four_way.runner_kernelblaster as rk

    try:
        spec = _spec_from_dict(spec_dict)
        if spec.pipe.optimizer == "kernelblaster":
            result = rk.run_kernelblaster_pipeline(spec)
        else:
            result = ra.run_autocomp_pipeline(spec)
        return result.to_dict()
    except Exception as exc:  # noqa: BLE001 — convert to typed failure row
        return _typed_failure_row(spec_dict, exc)


def _typed_failure_row(spec_dict: dict[str, Any], exc: Exception) -> dict[str, Any]:
    """Shape a runs.jsonl row for a cell that raised before producing a result."""
    return {
        "pipe": spec_dict.get("pipe"),
        "target": spec_dict.get("target"),
        "region": spec_dict.get("region_id"),
        "budget": spec_dict.get("search_budget"),
        "model": spec_dict.get("model"),
        "constrained": spec_dict.get("constrained"),
        "succeeded": False,
        "best_cycles": None,
        "total_iterations": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_llm_calls": 0,
        "wall_seconds": 0.0,
        "failure_reason": f"unhandled_exception: {type(exc).__name__}: {exc}",
        "extra": {"traceback": traceback.format_exc(limit=20)},
    }


def _spec_to_dict(spec: PipelineSpec) -> dict[str, Any]:
    return {
        "pipe": spec.pipe.value,
        "target": spec.target.value,
        "region_id": spec.region_id,
        "problem_spec": spec.problem_spec,
        "constraint_artifacts": spec.constraint_artifacts,
        "search_budget": spec.search_budget,
        "model": spec.model,
        "work_dir": str(spec.work_dir) if spec.work_dir else None,
        "extra_metadata": dict(spec.extra_metadata),
        # Cached for the typed-failure row.
        "constrained": spec.pipe.constrained,
    }


def _spec_from_dict(d: dict[str, Any]) -> PipelineSpec:
    return PipelineSpec(
        pipe=PipeId(d["pipe"]),
        target=TargetId(d["target"]),
        region_id=d["region_id"],
        problem_spec=d["problem_spec"],
        constraint_artifacts=d.get("constraint_artifacts"),
        search_budget=d["search_budget"],
        model=d.get("model", "gemini-2.5-flash"),
        work_dir=Path(d["work_dir"]) if d.get("work_dir") else None,
        extra_metadata=dict(d.get("extra_metadata", {})),
    )


# --------------------------------------------------------------------- #
# Sweep planner
# --------------------------------------------------------------------- #


def build_sweep_specs(
    *,
    bundle_dir: Path,
    pipelines: list[PipeId],
    targets: list[TargetId],
    regions: list[str],
    budgets: list[str],
    model: str,
    out_root: Path,
    run_id: str,
) -> list[PipelineSpec]:
    """Cartesian product of (pipeline × target × region × budget) → specs.

    Constraint artifacts are loaded once per region and reused across
    every cell that targets it. Pipelines A and B pass ``None``;
    pipelines C and D pass the loaded payload.
    """
    cache: dict[str, dict[str, Any] | None] = {}
    for region in regions:
        rc = build_region_constraints(bundle_dir=bundle_dir, region_id=region)
        cache[region] = rc.to_payload()
        # The same prose problem spec drives both pipelines per region
        # so the only difference is whether the constraint payload
        # lands in the prompt.
        cache[f"{region}__problem"] = rc.problem_spec  # type: ignore[assignment]

    specs: list[PipelineSpec] = []
    for pipe, target, region, budget in product(pipelines, targets, regions, budgets):
        constraint_artifacts = cache[region] if pipe.constrained else None
        work_dir = out_root / "cells" / (
            f"{pipe.short}_{target.value}_{region}_{budget}"
        )
        specs.append(
            PipelineSpec(
                pipe=pipe,
                target=target,
                region_id=region,
                problem_spec=str(cache[f"{region}__problem"]),
                constraint_artifacts=constraint_artifacts,
                search_budget=budget,
                model=model,
                work_dir=work_dir,
                extra_metadata={"run_id": run_id},
            )
        )
    return specs


def run_sweep(
    *,
    specs: list[PipelineSpec],
    out_dir: Path,
    workers: int = 4,
    enforce_budget_guard: bool = True,
) -> list[dict[str, Any]]:
    """Run all specs and persist results to ``out_dir/runs.jsonl``.

    Cells are dispatched in pipeline-letter order so the live
    ``compgen-gemini-usage watch --group-by pipe`` shows the four
    streams converging. Each completed cell appends one row to
    runs.jsonl (atomic line write).

    Returns:
        The list of row-dicts written, in completion order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    runs_path.write_text("")  # truncate

    rows: list[dict[str, Any]] = []
    spec_dicts = [_spec_to_dict(s) for s in specs]

    # If running in-process (workers == 1) we can short-circuit the
    # executor — tests rely on this so they don't fork.
    if workers <= 1:
        for sd in spec_dicts:
            guard = _budget_guard() if enforce_budget_guard else None
            if guard:
                rows.append(_budget_guard_row(sd, guard))
                _append_row(runs_path, rows[-1])
                continue
            row = _run_one_cell(sd)
            rows.append(row)
            _append_row(runs_path, row)
        return rows

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_sd = {pool.submit(_run_one_cell, sd): sd for sd in spec_dicts}
        # Even in the multi-worker path we still poll the budget
        # guard. If it trips, queued futures are cancelled and the
        # remaining cells log a typed budget_guard row.
        for fut in as_completed(future_to_sd):
            sd = future_to_sd[fut]
            guard = _budget_guard() if enforce_budget_guard else None
            if guard:
                # Cancel still-queued cells. Already-running ones
                # finish; their results land in runs.jsonl normally.
                for other_fut, other_sd in future_to_sd.items():
                    if other_fut.cancel():
                        rows.append(_budget_guard_row(other_sd, guard))
                        _append_row(runs_path, rows[-1])
                logger.warning("four_way.driver.budget_stop", reason=guard)
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                row = _typed_failure_row(sd, exc)
            rows.append(row)
            _append_row(runs_path, row)
    return rows


def _budget_guard_row(spec_dict: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "pipe": spec_dict.get("pipe"),
        "target": spec_dict.get("target"),
        "region": spec_dict.get("region_id"),
        "budget": spec_dict.get("search_budget"),
        "model": spec_dict.get("model"),
        "constrained": spec_dict.get("constrained"),
        "succeeded": False,
        "best_cycles": None,
        "total_iterations": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_llm_calls": 0,
        "wall_seconds": 0.0,
        "failure_reason": reason,
        "extra": {},
    }


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def _parse_csv(s: str) -> list[str]:
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Four-way constrained-vs-unconstrained sweep driver."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="CompGen bundle directory (e.g. artifacts/smolvla_int8).",
    )
    parser.add_argument(
        "--regions",
        type=_parse_csv,
        required=True,
        help="Comma-separated region ids from graph_dossier_v3.json.",
    )
    parser.add_argument(
        "--pipelines",
        type=_parse_csv,
        default="A,B,C,D",
        help="Comma-separated pipeline letters (default A,B,C,D).",
    )
    parser.add_argument(
        "--targets",
        type=_parse_csv,
        default="gemmini,saturn",
        help="Comma-separated target ids (default gemmini,saturn).",
    )
    parser.add_argument(
        "--budgets",
        type=_parse_csv,
        default="small,med",
        help="Comma-separated search budgets (small,med,large,xl).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="LLM model id (default the cheap Flash tier).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="ProcessPoolExecutor worker count (default 4 — one per pipeline letter).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/four_way/<run_id>/.",
    )
    parser.add_argument(
        "--no-budget-guard",
        action="store_true",
        help="Skip the cumulative-spend cap check before each cell.",
    )
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:8]
    out_dir = args.out or Path("results/four_way") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pipelines = [PipeId(f"pipe={letter.upper()}") for letter in args.pipelines]
    targets = [TargetId(t) for t in args.targets]

    specs = build_sweep_specs(
        bundle_dir=args.bundle,
        pipelines=pipelines,
        targets=targets,
        regions=args.regions,
        budgets=args.budgets,
        model=args.model,
        out_root=out_dir,
        run_id=run_id,
    )

    manifest = {
        "run_id": run_id,
        "started_at": time.time(),
        "bundle": str(args.bundle),
        "pipelines": [p.short for p in pipelines],
        "targets": [t.value for t in targets],
        "regions": args.regions,
        "budgets": args.budgets,
        "model": args.model,
        "workers": args.workers,
        "n_cells": len(specs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "four_way.driver.start",
        run_id=run_id,
        cells=len(specs),
        out_dir=str(out_dir),
    )

    rows = run_sweep(
        specs=specs,
        out_dir=out_dir,
        workers=args.workers,
        enforce_budget_guard=not args.no_budget_guard,
    )

    manifest["finished_at"] = time.time()
    manifest["wall_seconds"] = manifest["finished_at"] - manifest["started_at"]
    manifest["n_succeeded"] = sum(1 for r in rows if r.get("succeeded"))
    manifest["total_prompt_tokens"] = sum(r.get("prompt_tokens", 0) for r in rows)
    manifest["total_completion_tokens"] = sum(r.get("completion_tokens", 0) for r in rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nSweep complete. {manifest['n_succeeded']}/{len(rows)} cells succeeded.")
    print(f"Results: {out_dir / 'runs.jsonl'}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
