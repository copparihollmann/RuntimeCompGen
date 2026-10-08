"""Typed inputs and outputs of one four-way bench cell.

The driver fans these out across pipelines (A/B/C/D), targets
(gemmini/saturn), regions (from the int8 smolVLA dossier), and search
budgets. One ``(spec → result)`` pair per cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PipeId(str, Enum):
    """The four pipelines under comparison.

    Values match the ``source`` tag the runner stamps on every
    Gemini event — keep the letters terse so the dashboard fits on
    one screen.
    """

    A = "pipe=A"  # KernelBlaster vanilla
    B = "pipe=B"  # Autocomp vanilla
    C = "pipe=C"  # KernelBlaster + CompGen constraints
    D = "pipe=D"  # Autocomp + CompGen constraints

    @property
    def optimizer(self) -> str:
        return "kernelblaster" if self in {PipeId.A, PipeId.C} else "autocomp"

    @property
    def constrained(self) -> bool:
        return self in {PipeId.C, PipeId.D}

    @property
    def short(self) -> str:
        """``"A"`` / ``"B"`` / ``"C"`` / ``"D"`` — the letter alone."""
        return self.value.split("=", 1)[1]


class TargetId(str, Enum):
    """Hardware targets the four-way bench runs on."""

    GEMMINI = "gemmini"
    SATURN = "saturn"


@dataclass
class PipelineSpec:
    """Inputs for one cell of the 4 × 2 × N_regions × N_budgets grid.

    Attributes:
        pipe: Which of A/B/C/D.
        target: Hardware target the kernel will run on.
        region_id: Stable id from CompGen's graph dossier
            (``"matmul_0"``, ``"matmul_1"`` ...). Used as the source
            tag's metadata so spend per-region is visible.
        problem_spec: The LLM-facing prose description of the kernel
            to optimize. Built from the region's contract.
        constraint_artifacts: Dossier + kernel-contract slice for
            this region. ``None`` for pipelines A/B (unconstrained);
            populated for C/D.
        search_budget: ``"small"`` (20 iter) / ``"med"`` (40) /
            ``"large"`` (80) / ``"xl"`` (160). The runner adapter
            maps this to the optimizer-specific knob (KB's
            ``max_iterations``; autocomp's ``iterations`` ×
            ``num_plan_candidates``).
        model: LLM model id (``"gemini-2.5-flash"`` is the cheap
            default; pin per-cell if you want).
        work_dir: Where the runner writes per-iteration source +
            audit.json. The driver creates one per cell.
    """

    pipe: PipeId
    target: TargetId
    region_id: str
    problem_spec: str
    constraint_artifacts: dict[str, Any] | None
    search_budget: str
    model: str = "gemini-2.5-flash"
    work_dir: Path | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def usage_metadata(self) -> dict[str, Any]:
        """The metadata block stamped on every Gemini event for this cell."""
        return {
            "pipe": self.pipe.short,
            "target": self.target.value,
            "region": self.region_id,
            "budget": self.search_budget,
            "optimizer": self.pipe.optimizer,
            "constrained": self.pipe.constrained,
            **dict(self.extra_metadata),
        }


@dataclass
class PipelineResult:
    """Outcome of one cell.

    Persisted to ``results/four_way/<run_id>/runs.jsonl`` (one row
    per cell) and consumed by the Pareto reporter.
    """

    spec: PipelineSpec
    succeeded: bool
    best_kernel_source: str | None
    best_cycles: int | None
    total_iterations: int
    prompt_tokens: int
    completion_tokens: int
    n_llm_calls: int
    wall_seconds: float
    failure_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipe": self.spec.pipe.short,
            "target": self.spec.target.value,
            "region": self.spec.region_id,
            "budget": self.spec.search_budget,
            "model": self.spec.model,
            "constrained": self.spec.pipe.constrained,
            "succeeded": self.succeeded,
            "best_cycles": self.best_cycles,
            "total_iterations": self.total_iterations,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "n_llm_calls": self.n_llm_calls,
            "wall_seconds": self.wall_seconds,
            "failure_reason": self.failure_reason,
            "extra": dict(self.extra),
        }
