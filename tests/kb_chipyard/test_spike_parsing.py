"""Parsing tests for spike_eval — the regex paths must be honest."""

from __future__ import annotations

import pytest

from kernelblaster.kb_chipyard.spike_eval import _parse_correct, _parse_cycles


def test_parse_cycles_picks_up_the_harness_line() -> None:
    stdout = "boot\nCYCLES=12345\nbye\n"
    assert _parse_cycles(stdout) == 12345


def test_parse_cycles_tolerates_whitespace() -> None:
    stdout = "   CYCLES =  42\n"
    assert _parse_cycles(stdout) == 42


def test_parse_cycles_returns_none_when_missing() -> None:
    assert _parse_cycles("nothing to see here") is None
    assert _parse_cycles("") is None


def test_parse_cycles_returns_none_on_garbage_value() -> None:
    # If the LLM stripped the harness and printed "CYCLES=foo", parser
    # must return None rather than swallow a bad value.
    assert _parse_cycles("CYCLES=notanumber") is None


def test_parse_correct_round_trips() -> None:
    assert _parse_correct("CORRECT=1") is True
    assert _parse_correct("CORRECT=0") is False
    assert _parse_correct("") is None


def test_parse_cycles_grabs_first_occurrence() -> None:
    # The harness prints exactly once; if the LLM wrapper duplicates
    # we still take the first cycle count (it's the kernel proper).
    stdout = "CYCLES=100\nCYCLES=200\n"
    assert _parse_cycles(stdout) == 100
