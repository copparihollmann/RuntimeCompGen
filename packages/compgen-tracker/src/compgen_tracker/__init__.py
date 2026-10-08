"""compgen-tracker — Gemini API usage + cost accounting (standalone).

Carved out of CompGen so any project that calls Gemini can depend on
the tracker without pulling in CompGen's full compiler stack. CompGen
itself re-exports this surface via :mod:`compgen.observability` for
back-compat.

Public surface mirrors the historical CompGen surface so existing
callers don't need to change anything:

    from compgen_tracker import (
        record_call,
        load_summary,
        Budget,
        tracking_source,
        install_genai_instrumentation,
    )

CLI: ``compgen-gemini-usage`` (entry point shipped by this package).
"""

from compgen_tracker.gemini_usage import (
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

__version__ = "0.1.0"
