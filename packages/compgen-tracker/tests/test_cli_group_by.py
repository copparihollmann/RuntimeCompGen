"""Tests for the ``compgen-gemini-usage watch --group-by ...`` aggregator.

The four-way bench tags every Gemini call with
``tracking_source("pipe=A", pipe="A", target="gemmini", region="r0")``.
``--group-by source`` and ``--group-by pipe`` both have to split spend
cleanly so the user can watch all four pipelines in one pane.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from compgen_tracker import cli as cli_mod
from compgen_tracker import gemini_usage as gu


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("COMPGEN_REPO_ROOT", str(tmp_path))
    return tmp_path


def _seed_four_way_events() -> None:
    # Mimic the four-way bench's tagging convention. Each pipeline makes
    # a different number of calls and burns a different number of tokens.
    gu.record_call(
        "gemini-2.5-flash", 1000, 200,
        source="pipe=A", metadata={"pipe": "A", "target": "gemmini", "region": "r0"},
    )
    gu.record_call(
        "gemini-2.5-flash", 1500, 300,
        source="pipe=A", metadata={"pipe": "A", "target": "saturn", "region": "r1"},
    )
    gu.record_call(
        "gemini-2.5-flash", 800, 100,
        source="pipe=B", metadata={"pipe": "B", "target": "gemmini", "region": "r0"},
    )
    gu.record_call(
        "gemini-2.5-flash", 400, 80,
        source="pipe=C", metadata={"pipe": "C", "target": "gemmini", "region": "r0"},
    )
    gu.record_call(
        "gemini-2.5-flash", 300, 60,
        source="pipe=D", metadata={"pipe": "D", "target": "saturn", "region": "r1"},
    )


def test_aggregate_by_source_splits_pipelines(isolated_storage: Path) -> None:
    """Per-pipeline rollup must keep token counts disjoint."""
    _seed_four_way_events()

    buckets = cli_mod._aggregate_by(None, "source")
    assert set(buckets.keys()) == {"pipe=A", "pipe=B", "pipe=C", "pipe=D"}
    assert buckets["pipe=A"]["calls"] == 2
    assert buckets["pipe=A"]["prompt_tokens"] == 2500
    assert buckets["pipe=B"]["calls"] == 1
    assert buckets["pipe=B"]["prompt_tokens"] == 800
    # All four pipelines together must equal the cumulative total.
    summary = gu.load_summary()
    assert sum(b["calls"] for b in buckets.values()) == summary.total_calls
    assert sum(b["prompt_tokens"] for b in buckets.values()) == summary.total_prompt_tokens
    assert sum(b["completion_tokens"] for b in buckets.values()) == summary.total_completion_tokens


def test_aggregate_by_metadata_key_splits_targets(isolated_storage: Path) -> None:
    """Splitting by a metadata key (``target``) must separate Gemmini vs Saturn."""
    _seed_four_way_events()

    by_target = cli_mod._aggregate_by(None, "target")
    assert set(by_target.keys()) == {"gemmini", "saturn"}
    # Three calls were tagged gemmini, two were tagged saturn.
    assert by_target["gemmini"]["calls"] == 3
    assert by_target["saturn"]["calls"] == 2


def test_aggregate_by_unknown_dimension_falls_back_to_none_bucket(
    isolated_storage: Path,
) -> None:
    """Missing metadata keys must surface as ``(none)`` instead of crashing."""
    _seed_four_way_events()

    by_unknown = cli_mod._aggregate_by(None, "nonexistent_key")
    assert set(by_unknown.keys()) == {"(none)"}
    assert by_unknown["(none)"]["calls"] == 5


def test_render_group_by_handles_empty(isolated_storage: Path) -> None:
    """Empty bucket dict must still produce a renderable table."""
    table = cli_mod._render_group_by("source", {})
    # Rendering should not raise; the placeholder row keeps the panel
    # informative when the bench hasn't started yet.
    assert table is not None
    assert table.row_count == 1


def test_status_subcommand_still_works(isolated_storage: Path) -> None:
    """Adding --group-by must not regress the no-flag default ``status``."""
    _seed_four_way_events()
    runner = CliRunner()
    result = runner.invoke(cli_mod.main, ["status"])
    assert result.exit_code == 0, result.output
    # Cumulative totals show up in the rendered output.
    assert "Total calls" in result.output
