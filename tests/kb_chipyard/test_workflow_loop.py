"""End-to-end test of the kb_chipyard workflow loop using fakes.

This is the most valuable single test in the kb_chipyard suite: it
runs the actual workflow.run_chipyard_workflow control flow, but with
the LLM call, cross-compile, and spike subprocess all replaced by
in-process fakes so the test runs in <2s and doesn't need a RISC-V
toolchain.

What it proves:
  1. The loop terminates on the budget bound (max_iterations).
  2. ``best_cycles`` is monotonically improving (we feed it 100→50→40).
  3. Token totals match what the fake LLM reported per iteration.
  4. ``record_kb_response`` lands one event per iteration tagged with
     the correct source + metadata.
  5. The audit.json file the workflow drops carries iteration-level
     details the bench's verification test #7 (token accounting
     invariant) depends on.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kernelblaster.kb_chipyard import workflow as wf
from kernelblaster.kb_chipyard.config import ChipyardConfig
from kernelblaster.kb_chipyard.cross_compile import CrossCompileResult
from kernelblaster.kb_chipyard.spike_eval import SpikeEvalResult
from kernelblaster.kb_chipyard.target import ChipyardTarget

from compgen.observability import gemini_usage as gu


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test gets its own events.jsonl so token attribution claims are clean."""
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("COMPGEN_REPO_ROOT", str(tmp_path))
    return tmp_path


def _fake_llm_factory(cycles_trajectory: list[int]) -> Any:
    """Build an async callable that returns one ```c ... ``` block per call.

    Each call advances through ``cycles_trajectory`` so the workflow
    sees monotonically improving outcomes if the cross_compile + spike
    fakes are wired to use the same index.
    """
    call_count = {"n": 0}

    async def fake(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any], float, Any]:
        idx = call_count["n"]
        call_count["n"] += 1
        body = (
            "```c\n"
            f"void kernel(const int8_t *A, const int8_t *B, int32_t *C,\n"
            f"            int M, int N, int K) {{ /* attempt {idx} */ }}\n"
            "```"
        )
        usage = {"prompt_tokens": 1000 + idx * 100, "completion_tokens": 200 + idx * 20}
        raw = SimpleNamespace(usage=usage, model="gemini-2.5-flash", elapsed_time=0.01)
        return body, usage, 0.01, raw

    return fake, call_count


def _patch_compile_and_spike(
    monkeypatch: pytest.MonkeyPatch,
    cycles_trajectory: list[int],
) -> None:
    """Stub out the riscv64 toolchain and Spike subprocess.

    The fakes return one outcome per call indexed by ``call_count``.
    Cycles trajectory determines the perf signal the workflow sees.
    """
    call_count = {"n": 0}

    def fake_cross_compile(cfg, source_path, **_kwargs):
        # Always succeed: we want the workflow to *reach* the spike phase.
        elf_path = source_path.with_suffix(".riscv")
        elf_path.write_bytes(b"")  # placeholder
        return CrossCompileResult(
            success=True,
            elf_path=elf_path,
            stdout="",
            stderr="",
            command=["fake-gcc"],
            duration_s=0.001,
        )

    def fake_run_spike(cfg, elf_path):
        idx = call_count["n"]
        call_count["n"] += 1
        cycles = cycles_trajectory[min(idx, len(cycles_trajectory) - 1)]
        return SpikeEvalResult(
            succeeded=True,
            cycles=cycles,
            correct=True,
            stdout=f"CYCLES={cycles}\nCORRECT=1\n",
            stderr="",
            return_code=0,
            duration_s=0.001,
            failure_reason=None,
        )

    monkeypatch.setattr(wf, "cross_compile", fake_cross_compile)
    monkeypatch.setattr(wf, "run_spike", fake_run_spike)


def test_workflow_finds_best_kernel_and_tracks_tokens(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All five behaviors listed in this module's docstring, in one test."""
    cycles_trajectory = [100, 75, 50, 50, 50]
    fake_llm, call_count = _fake_llm_factory(cycles_trajectory)
    _patch_compile_and_spike(monkeypatch, cycles_trajectory)

    cfg = ChipyardConfig(
        target=ChipyardTarget.GEMMINI,
        max_iterations=5,
        early_stop_no_improvement=10,  # disable for this test
        usage_source="pipe=A",
        usage_metadata={"pipe": "A", "target": "gemmini"},
    )

    result = asyncio.run(
        wf.run_chipyard_workflow(
            cfg,
            problem_spec="int8 matmul 16x16x16 - smoke",
            region_id="r0",
            work_dir=tmp_path / "wf",
            llm_callable=fake_llm,
        )
    )

    # 1. Loop terminated cleanly at max_iterations.
    assert result.total_iterations == 5
    assert result.succeeded is True
    # 2. Best cycles tracks the global minimum of the trajectory.
    assert result.best_cycles == 50
    assert "attempt" in (result.best_kernel_source or "")
    # 3. Token totals match the fake LLM's reported usage.
    expected_prompt = sum(1000 + i * 100 for i in range(5))
    expected_completion = sum(200 + i * 20 for i in range(5))
    assert result.total_prompt_tokens == expected_prompt
    assert result.total_completion_tokens == expected_completion
    # 4. CompGen usage tracker recorded one event per LLM call,
    # all tagged with the pipeline label.
    summary = gu.load_summary()
    assert summary.total_calls == 5
    events_path = isolated_storage / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    assert len(events) == 5
    for ev in events:
        assert ev["source"] == "pipe=A"
        assert ev["metadata"]["pipe"] == "A"
        assert ev["metadata"]["target"] == "gemmini"
        assert ev["metadata"]["region"] == "r0"
    # 5. audit.json captures iteration-level details.
    audit = json.loads((tmp_path / "wf" / "audit.json").read_text())
    assert audit["target"] == "gemmini"
    assert audit["best_cycles"] == 50
    assert len(audit["iterations"]) == 5
    assert audit["iterations"][0]["cycles"] == 100
    assert audit["iterations"][2]["cycles"] == 50


def test_workflow_early_stops_after_no_improvement_streak(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plateau at 100 cycles for 5 iterations → stop before max_iterations."""
    cycles_trajectory = [100, 100, 100, 100, 100, 100, 100, 100]
    fake_llm, _ = _fake_llm_factory(cycles_trajectory)
    _patch_compile_and_spike(monkeypatch, cycles_trajectory)

    cfg = ChipyardConfig(
        target=ChipyardTarget.SATURN,
        max_iterations=20,
        early_stop_no_improvement=4,
        usage_source="pipe=A",
    )

    result = asyncio.run(
        wf.run_chipyard_workflow(
            cfg,
            problem_spec="int8 matmul - plateau",
            region_id="r0",
            work_dir=tmp_path / "wf",
            llm_callable=fake_llm,
        )
    )

    # iter 0: cycles=100, sets best. Streak=0.
    # iters 1-4: same cycles, streak grows to 4 → stop.
    assert result.total_iterations <= 6
    assert result.best_cycles == 100
    assert result.succeeded is True


def test_workflow_distinguishes_constrained_from_unconstrained(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline C must mark its result as constrained; pipeline A must not."""
    fake_llm, _ = _fake_llm_factory([100, 100])
    _patch_compile_and_spike(monkeypatch, [100, 100])

    cfg_a = ChipyardConfig(
        target=ChipyardTarget.GEMMINI,
        max_iterations=1,
        early_stop_no_improvement=0,
        usage_source="pipe=A",
        constraint_artifacts=None,
    )
    cfg_c = ChipyardConfig(
        target=ChipyardTarget.GEMMINI,
        max_iterations=1,
        early_stop_no_improvement=0,
        usage_source="pipe=C",
        constraint_artifacts={
            "region_id": "r0",
            "dossier": {"op_family": "matmul"},
            "contract": {"tile_params": {"BM": 16}},
        },
    )

    a = asyncio.run(
        wf.run_chipyard_workflow(
            cfg_a, problem_spec="x", region_id="r0", work_dir=tmp_path / "a",
            llm_callable=fake_llm,
        )
    )
    fake_llm_c, _ = _fake_llm_factory([100, 100])
    c = asyncio.run(
        wf.run_chipyard_workflow(
            cfg_c, problem_spec="x", region_id="r0", work_dir=tmp_path / "c",
            llm_callable=fake_llm_c,
        )
    )

    assert a.constrained is False
    assert c.constrained is True
    audit_a = json.loads((tmp_path / "a" / "audit.json").read_text())
    audit_c = json.loads((tmp_path / "c" / "audit.json").read_text())
    assert audit_a["constrained"] is False
    assert audit_c["constrained"] is True


def test_workflow_handles_no_code_block_extracted(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the LLM forgets the code fence, the iteration is recorded as a miss but the loop doesn't crash."""

    async def fake_llm(_messages):
        return "no fence here just prose", {"prompt_tokens": 50, "completion_tokens": 10}, 0.01, SimpleNamespace(
            usage={"prompt_tokens": 50, "completion_tokens": 10},
            model="gemini-2.5-flash",
            elapsed_time=0.01,
        )

    cfg = ChipyardConfig(
        target=ChipyardTarget.SATURN,
        max_iterations=3,
        early_stop_no_improvement=10,
        usage_source="pipe=A",
    )
    result = asyncio.run(
        wf.run_chipyard_workflow(
            cfg, problem_spec="x", region_id=None, work_dir=tmp_path / "wf",
            llm_callable=fake_llm,
        )
    )
    assert result.succeeded is False
    assert result.best_cycles is None
    assert result.total_iterations == 3
    # Tokens were spent even though no kernel landed — that's the
    # honest accounting the bench needs.
    assert result.total_prompt_tokens == 150
