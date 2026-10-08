"""Back-compat shim: the tracker now lives in ``compgen_tracker``.

This module exists so existing ``from compgen.observability.gemini_usage
import ...`` callers keep working after the extraction. New code should
import from :mod:`compgen_tracker` directly.

See ``packages/compgen-tracker/`` for the canonical source.
"""

from __future__ import annotations

# Re-export every public symbol from the standalone package. Star-import
# is the right call here: this file IS the symbol-forwarding contract,
# and the package's ``__all__`` curates what's public.
from compgen_tracker.gemini_usage import *  # noqa: F401, F403
from compgen_tracker.gemini_usage import (  # explicit re-exports — keep tools happy
    PRICING,
    PRICING_SOURCE_URL,
    PRICING_VERIFIED_AT,
    Budget,
    BudgetStatus,
    MonthBucket,
    UsageEvent,
    UsageSummary,
    budget_path,
    build_summary,
    compute_cost_usd,
    evaluate_budget,
    events_path,
    get_storage_dir,
    install_genai_instrumentation,
    install_openai_instrumentation,
    is_genai_instrumented,
    is_openai_instrumented,
    iter_events,
    load_pricing_overrides,
    load_summary,
    record_call,
    record_from_response,
    resolve_rates,
    summary_path,
    tracking_source,
)

__all__ = [
    "Budget",
    "BudgetStatus",
    "MonthBucket",
    "PRICING",
    "PRICING_SOURCE_URL",
    "PRICING_VERIFIED_AT",
    "UsageEvent",
    "UsageSummary",
    "budget_path",
    "build_summary",
    "compute_cost_usd",
    "evaluate_budget",
    "events_path",
    "get_storage_dir",
    "install_genai_instrumentation",
    "install_openai_instrumentation",
    "is_genai_instrumented",
    "is_openai_instrumented",
    "iter_events",
    "load_pricing_overrides",
    "load_summary",
    "record_call",
    "record_from_response",
    "resolve_rates",
    "summary_path",
    "tracking_source",
]
