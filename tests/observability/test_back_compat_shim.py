"""Verify the ``compgen.observability`` shim forwards to ``compgen_tracker``.

The full test suite lives in ``packages/compgen-tracker/tests/`` — this
file just locks in that historical imports through
``compgen.observability`` resolve to the same objects so a downstream
caller that hasn't migrated yet keeps working.
"""

from __future__ import annotations


def test_package_shim_forwards_public_api() -> None:
    """``from compgen.observability import X`` must equal ``from compgen_tracker import X``."""
    from compgen_tracker import (
        Budget,
        UsageEvent,
        UsageSummary,
        load_summary,
        record_call,
        tracking_source,
    )
    from compgen.observability import (
        Budget as BudgetShim,
        UsageEvent as UsageEventShim,
        UsageSummary as UsageSummaryShim,
        load_summary as load_summary_shim,
        record_call as record_call_shim,
        tracking_source as tracking_source_shim,
    )

    assert Budget is BudgetShim
    assert UsageEvent is UsageEventShim
    assert UsageSummary is UsageSummaryShim
    assert load_summary is load_summary_shim
    assert record_call is record_call_shim
    assert tracking_source is tracking_source_shim


def test_module_shim_forwards_module_attributes() -> None:
    """``compgen.observability.gemini_usage`` must alias the real module."""
    from compgen.observability import gemini_usage as gu_shim
    from compgen_tracker import gemini_usage as gu_real

    # Critical public surface used by the four-way bench:
    assert gu_shim.record_call is gu_real.record_call
    assert gu_shim.load_summary is gu_real.load_summary
    assert gu_shim.PRICING is gu_real.PRICING
    assert gu_shim.tracking_source is gu_real.tracking_source


def test_cli_shim_forwards_main_entry() -> None:
    from compgen.observability.cli import main as cli_main_shim
    from compgen_tracker.cli import main as cli_main_real

    assert cli_main_shim is cli_main_real
