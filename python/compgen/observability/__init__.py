"""Observability subsystem: usage tracking, cost accounting, live monitoring.

Thin pass-through to the standalone ``compgen-tracker`` package. The
tracker was carved out so xpu-rt-integration (and other downstream
projects) can depend on it without pulling in CompGen's full compiler
stack. See ``packages/compgen-tracker/`` for the canonical source.
"""

from __future__ import annotations

from compgen_tracker import (
    Budget,
    UsageEvent,
    UsageSummary,
    compute_cost_usd,
    get_storage_dir,
    install_genai_instrumentation,
    is_genai_instrumented,
    load_summary,
    record_call,
    tracking_source,
)

__all__ = [
    "Budget",
    "UsageEvent",
    "UsageSummary",
    "compute_cost_usd",
    "get_storage_dir",
    "install_genai_instrumentation",
    "is_genai_instrumented",
    "load_summary",
    "record_call",
    "tracking_source",
]
