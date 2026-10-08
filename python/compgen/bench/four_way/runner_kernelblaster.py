"""Runner adapter for pipelines A (vanilla) and C (CompGen-constrained).

Both pipelines drive ``kb_chipyard`` — the difference is whether the
``ChipyardConfig.constraint_artifacts`` field is populated.

The CompGen-side adapter does three things:

  1. Ensures ``third_party/kernelblaster/src`` is on ``sys.path`` (KB
     isn't pip-installed in CompGen's venv).
  2. Translates ``PipelineSpec`` → ``ChipyardConfig`` and the
     ``search_budget`` enum → ``max_iterations``.
  3. Returns a :class:`PipelineResult` shaped for the bench's runs.jsonl.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import structlog

from compgen.bench.four_way.spec import (
    PipelineResult,
    PipelineSpec,
    TargetId,
)

logger = structlog.get_logger()

#: Maps the bench's named budgets to kb_chipyard iteration counts.
#: Calibrated so xl is ~8× small — covers a roughly two-decade sweep
#: on the Pareto plot.
SEARCH_BUDGET_TO_ITERS: dict[str, int] = {
    "small": 20,
    "med": 40,
    "large": 80,
    "xl": 160,
}


def _ensure_kb_on_path() -> None:
    """Prepend ``third_party/kernelblaster/src`` once per process.

    Mirrors the trick KB's own ``scripts/run_RL.py`` does for itself.
    """
    here = Path(__file__).resolve()
    # python/compgen/bench/four_way/ → repo root is 4 parents up
    repo_root = here.parents[4]
    kb_src = repo_root / "third_party" / "kernelblaster" / "src"
    if str(kb_src) not in sys.path:
        sys.path.insert(0, str(kb_src))


def _to_chipyard_target(target: TargetId) -> Any:
    """Map our :class:`TargetId` to kb_chipyard's :class:`ChipyardTarget`."""
    _ensure_kb_on_path()
    from kernelblaster.kb_chipyard.target import ChipyardTarget  # type: ignore[import-not-found]

    if target is TargetId.GEMMINI:
        return ChipyardTarget.GEMMINI
    return ChipyardTarget.SATURN


def _build_chipyard_config(spec: PipelineSpec) -> Any:
    _ensure_kb_on_path()
    from kernelblaster.kb_chipyard.config import ChipyardConfig  # type: ignore[import-not-found]

    iters = SEARCH_BUDGET_TO_ITERS.get(spec.search_budget)
    if iters is None:
        raise ValueError(
            f"unknown search_budget {spec.search_budget!r}; "
            f"choose one of {sorted(SEARCH_BUDGET_TO_ITERS)}"
        )

    return ChipyardConfig(
        target=_to_chipyard_target(spec.target),
        model=spec.model,
        max_iterations=iters,
        # 25% of budget without improvement → bail. Lets the bench
        # finish cleanly when an arm has clearly converged.
        early_stop_no_improvement=max(5, iters // 4),
        usage_source=spec.pipe.value,
        usage_metadata=spec.usage_metadata(),
        constraint_artifacts=spec.constraint_artifacts,
    )


def run_kernelblaster_pipeline(spec: PipelineSpec) -> PipelineResult:
    """Drive kb_chipyard for one (pipe × target × region × budget) cell.

    Synchronous wrapper around the underlying async workflow — the
    sweep driver (``four_way/driver.py``) farms these out across
    ProcessPoolExecutor workers, so each call gets its own asyncio
    loop without contention.
    """
    return asyncio.run(_run_async(spec))


async def _run_async(spec: PipelineSpec) -> PipelineResult:
    _ensure_kb_on_path()
    from kernelblaster.kb_chipyard.workflow import run_chipyard_workflow  # type: ignore[import-not-found]

    if spec.pipe.optimizer != "kernelblaster":
        raise ValueError(
            f"runner_kernelblaster called with non-KB pipe {spec.pipe!r}"
        )

    cfg = _build_chipyard_config(spec)
    work_dir = spec.work_dir or Path(".four_way_tmp") / (
        f"{spec.pipe.short}_{spec.target.value}_{spec.region_id}_{spec.search_budget}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    logger.info(
        "four_way.kb.start",
        pipe=spec.pipe.short,
        target=spec.target.value,
        region=spec.region_id,
        budget=spec.search_budget,
        constrained=spec.pipe.constrained,
    )

    workflow_result = await run_chipyard_workflow(
        cfg,
        problem_spec=spec.problem_spec,
        region_id=spec.region_id,
        work_dir=work_dir,
    )

    logger.info(
        "four_way.kb.done",
        pipe=spec.pipe.short,
        target=spec.target.value,
        region=spec.region_id,
        succeeded=workflow_result.succeeded,
        best_cycles=workflow_result.best_cycles,
        prompt_tokens=workflow_result.total_prompt_tokens,
        completion_tokens=workflow_result.total_completion_tokens,
        iterations=workflow_result.total_iterations,
    )

    return PipelineResult(
        spec=spec,
        succeeded=workflow_result.succeeded,
        best_kernel_source=workflow_result.best_kernel_source,
        best_cycles=workflow_result.best_cycles,
        total_iterations=workflow_result.total_iterations,
        prompt_tokens=workflow_result.total_prompt_tokens,
        completion_tokens=workflow_result.total_completion_tokens,
        n_llm_calls=workflow_result.total_iterations,
        wall_seconds=time.time() - started,
        failure_reason=workflow_result.failure_reason,
        extra={"audit_path": str(work_dir / "audit.json")},
    )
