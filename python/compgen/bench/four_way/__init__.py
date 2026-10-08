"""Four-way constrained-vs-unconstrained kernel-search benchmark.

Compares four pipelines on the same (target × region × budget) cell:

  A. KernelBlaster vanilla         (no CompGen constraints)
  B. Autocomp vanilla              (no CompGen constraints)
  C. KernelBlaster + CompGen       (dossier + kernel-contract injected)
  D. Autocomp + CompGen            (dossier + kernel-contract injected)

Hypotheses (per target):
  H1: tokens(C) << tokens(A) and tokens(D) << tokens(B) at iso-perf
  H2: perf(C) ≈ perf(A) and perf(D) ≈ perf(B) at iso-token-budget

Every Gemini call is routed through
:mod:`compgen.observability.gemini_usage` with ``source="pipe={letter}"``
and metadata ``{pipe, target, region}`` so
``compgen-gemini-usage watch --group-by source`` (or ``pipe``)
separates spend live across pipelines.
"""

from compgen.bench.four_way.constraints import (
    RegionConstraints,
    build_region_constraints,
)
from compgen.bench.four_way.spec import (
    PipelineResult,
    PipelineSpec,
    PipeId,
    TargetId,
)

__all__ = [
    "PipeId",
    "PipelineResult",
    "PipelineSpec",
    "RegionConstraints",
    "TargetId",
    "build_region_constraints",
]
