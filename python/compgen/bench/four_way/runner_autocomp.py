"""Runner adapter for pipelines B (vanilla) and D (CompGen-constrained).

Both pipelines drive autocomp's :class:`BeamSearchStrategy`. The
difference is whether the dossier + kernel-contract slice is folded
into the problem prose the planner agent sees.

Autocomp's :class:`autocomp.search.prob.Prob` carries the problem
definition. Its agent ensemble doesn't expose a first-class
``extra_system_prompt`` hook today, so the constraint payload is
appended as an annotated postscript to the user-facing problem text.
The planner observes it verbatim; verification test #4 in the plan
asserts that the rendered prompt for pipeline D contains the
dossier substring (proving the injection is live).

Token attribution: autocomp already wraps every Gemini call with
``tracking_source()``, so the events land in CompGen's
``events.jsonl`` tagged with the pipeline letter as long as we
push the right context-var before invoking the search.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog

from compgen.bench.four_way.spec import (
    PipelineResult,
    PipelineSpec,
    TargetId,
)
from compgen.observability.gemini_usage import tracking_source

logger = structlog.get_logger()

#: Autocomp's ``iterations`` axis. Same overall budget as KB so the
#: 4-way Pareto plot is apples-to-apples on the token axis.
SEARCH_BUDGET_TO_AUTOCOMP_ITERS: dict[str, int] = {
    "small": 20,
    "med": 40,
    "large": 80,
    "xl": 160,
}


def _format_constraint_postscript(constraints: dict[str, Any] | None) -> str:
    """Render the constraint payload as a problem-text postscript.

    Matches the shape kb_chipyard's prompt builder uses, so an
    audit comparing prompt texts across pipelines C and D sees the
    same constraint substring. The autocomp planner reads it as part
    of the problem statement.
    """
    if not constraints:
        return ""
    import json as _json

    parts = ["\n\n## CompGen constraint artifacts (honor these)"]
    region = constraints.get("region_id")
    if region:
        parts.append(f"\nRegion id: {region}")
    dossier = constraints.get("dossier")
    if dossier:
        parts.append("\nGraph dossier:\n```json\n" + _json.dumps(dossier, indent=2) + "\n```")
    contract = constraints.get("contract")
    if contract:
        parts.append("\nKernel contract:\n```yaml\n" + _json.dumps(contract, indent=2) + "\n```")
    readiness = constraints.get("readiness")
    if readiness:
        parts.append("\nReadiness report:\n```json\n" + _json.dumps(readiness, indent=2) + "\n```")
    return "".join(parts)


def _autocomp_agent_and_backend(target: TargetId) -> tuple[str, str]:
    """Map our TargetId to autocomp's (agent_name, backend_name) pair."""
    if target is TargetId.GEMMINI:
        # autocomp ships agents/gemmini/ and backend/gemmini/ — see
        # autocomp/search/search.py::create_backend_and_agents.
        return "gemmini", "gemmini"
    # Saturn uses the agent_builder pack autocomp shipped for RVV.
    return "built:saturn-rvv", "saturn"


def run_autocomp_pipeline(spec: PipelineSpec) -> PipelineResult:
    """Drive autocomp for one cell.

    Wraps the synchronous autocomp search inside CompGen's
    ``tracking_source`` so every Gemini event lands tagged with the
    pipeline letter. If autocomp itself isn't importable in this
    venv (heavy deps), returns a typed ``failed`` result rather than
    raising — keeps the rest of the sweep alive.
    """
    return asyncio.run(_run_async(spec))


async def _run_async(spec: PipelineSpec) -> PipelineResult:
    if spec.pipe.optimizer != "autocomp":
        raise ValueError(
            f"runner_autocomp called with non-autocomp pipe {spec.pipe!r}"
        )

    iters = SEARCH_BUDGET_TO_AUTOCOMP_ITERS.get(spec.search_budget)
    if iters is None:
        raise ValueError(
            f"unknown search_budget {spec.search_budget!r}; "
            f"choose one of {sorted(SEARCH_BUDGET_TO_AUTOCOMP_ITERS)}"
        )

    agent_name, backend_name = _autocomp_agent_and_backend(spec.target)
    postscript = _format_constraint_postscript(spec.constraint_artifacts)
    full_problem = (spec.problem_spec + postscript).strip()

    work_dir = spec.work_dir or Path(".four_way_tmp") / (
        f"{spec.pipe.short}_{spec.target.value}_{spec.region_id}_{spec.search_budget}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    logger.info(
        "four_way.autocomp.start",
        pipe=spec.pipe.short,
        target=spec.target.value,
        agent=agent_name,
        backend=backend_name,
        region=spec.region_id,
        budget=spec.search_budget,
        constrained=spec.pipe.constrained,
        postscript_chars=len(postscript),
    )

    try:
        outcome = await _drive_autocomp(
            agent_name=agent_name,
            backend_name=backend_name,
            iterations=iters,
            problem_spec=full_problem,
            model=spec.model,
            target=spec.target,
            usage_source=spec.pipe.value,
            usage_metadata=spec.usage_metadata(),
            work_dir=work_dir,
        )
    except _AutocompUnavailable as exc:
        return PipelineResult(
            spec=spec,
            succeeded=False,
            best_kernel_source=None,
            best_cycles=None,
            total_iterations=0,
            prompt_tokens=0,
            completion_tokens=0,
            n_llm_calls=0,
            wall_seconds=time.time() - started,
            failure_reason=f"autocomp_unavailable: {exc}",
        )

    return PipelineResult(
        spec=spec,
        succeeded=outcome.get("succeeded", False),
        best_kernel_source=outcome.get("best_kernel_source"),
        best_cycles=outcome.get("best_cycles"),
        total_iterations=outcome.get("total_iterations", 0),
        prompt_tokens=outcome.get("prompt_tokens", 0),
        completion_tokens=outcome.get("completion_tokens", 0),
        n_llm_calls=outcome.get("n_llm_calls", 0),
        wall_seconds=time.time() - started,
        failure_reason=outcome.get("failure_reason"),
        extra={
            "agent": agent_name,
            "backend": backend_name,
            "audit_path": str(work_dir / "autocomp_audit.json"),
            "prompt_postscript_chars": len(postscript),
        },
    )


class _AutocompUnavailable(RuntimeError):
    """Raised when autocomp can't be imported / driven in this env."""


async def _drive_autocomp(
    *,
    agent_name: str,
    backend_name: str,
    iterations: int,
    problem_spec: str,
    model: str,
    target: TargetId,
    usage_source: str,
    usage_metadata: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    """Bridge to autocomp's BeamSearchStrategy.

    The autocomp surface expects a :class:`Prob` instance plus an
    ``agent_name`` / ``backend_name`` / ``hw_config`` triple. We wrap
    the whole invocation in :func:`tracking_source` so every Gemini
    event lands tagged with the pipeline letter and metadata.

    Designed to be importable + monkey-patchable from tests so the
    runner can be unit-tested without actually invoking autocomp's
    heavy planner / Spike loop.
    """
    try:
        from autocomp.hw_config import (  # type: ignore[import-not-found]
            GemminiHardwareConfig,
            SaturnHardwareConfig,
        )
        from autocomp.search.prob import Prob  # type: ignore[import-not-found]
        from autocomp.search.search import (  # type: ignore[import-not-found]
            BeamSearchStrategy,
            create_backend_and_agents,
        )
    except ImportError as exc:
        raise _AutocompUnavailable(str(exc)) from exc

    if target is TargetId.GEMMINI:
        hw_config = GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)
    else:
        hw_config = SaturnHardwareConfig(vlen=128, dlen=64)

    with tracking_source(usage_source, **usage_metadata):
        # autocomp's surface is synchronous (and heavy). Run in a
        # thread so the async runner's event loop stays responsive
        # if a sibling pipeline shares it.
        return await asyncio.to_thread(
            _drive_autocomp_sync,
            agent_name=agent_name,
            backend_name=backend_name,
            hw_config=hw_config,
            iterations=iterations,
            problem_spec=problem_spec,
            model=model,
            work_dir=work_dir,
            Prob=Prob,
            BeamSearchStrategy=BeamSearchStrategy,
            create_backend_and_agents=create_backend_and_agents,
        )


def _drive_autocomp_sync(
    *,
    agent_name: str,
    backend_name: str,
    hw_config: Any,
    iterations: int,
    problem_spec: str,
    model: str,
    work_dir: Path,
    Prob: Any,
    BeamSearchStrategy: Any,
    create_backend_and_agents: Any,
) -> dict[str, Any]:
    """The synchronous body — separated so tests can monkey-patch."""
    import json as _json

    # autocomp's Prob signature is (prob_type, prob_id). The four-way
    # bench feeds custom problems, not the bundled benchmark IDs;
    # we synthesize a single-shot Prob by writing the spec to a
    # temporary file the autocomp loader can find. The minimal-viable
    # integration here records the call and returns a typed outcome
    # — the heavy autocomp wiring is a follow-up patch on the autocomp
    # submodule itself (it currently requires bundled problem files
    # for its harness).
    audit_path = work_dir / "autocomp_audit.json"
    audit_path.write_text(
        _json.dumps(
            {
                "agent_name": agent_name,
                "backend_name": backend_name,
                "iterations": iterations,
                "model": model,
                "problem_spec_chars": len(problem_spec),
                "hw_config": getattr(hw_config, "get_hw_description", lambda: str(hw_config))(),
                "note": (
                    "autocomp adapter MVP — record-only pending the "
                    "autocomp-side custom-problem injection patch. "
                    "Token tracking via tracking_source() IS live."
                ),
            },
            indent=2,
        )
    )

    # Honest typed return: pipeline ran but the heavy search isn't
    # wired yet. The four-way driver will see succeeded=False and
    # surface this in the bench report's "limitations" box.
    return {
        "succeeded": False,
        "best_kernel_source": None,
        "best_cycles": None,
        "total_iterations": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "n_llm_calls": 0,
        "failure_reason": (
            "autocomp_custom_problem_injection_pending: "
            "the autocomp submodule's Prob class expects a bundled "
            "problem id; the four-way driver needs the autocomp-side "
            "patch to accept arbitrary problem text. KB-side cells "
            "(A, C) run end-to-end today."
        ),
    }
