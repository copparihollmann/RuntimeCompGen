"""End-to-end test of the KB runner adapter with kb_chipyard's fake hooks.

Exercises the full ``PipelineSpec → kb_chipyard workflow → PipelineResult``
path. The LLM call and the subprocess hops (cross_compile, run_spike)
are replaced with in-process fakes, so the test runs in seconds and
needs neither Gemini nor a RISC-V toolchain.

What it proves:
  * Pipelines A and C call kb_chipyard with the right
    ``constraint_artifacts`` value (None vs populated dict).
  * Search-budget enum → max_iterations is wired correctly.
  * Token totals propagate from kb_chipyard back to PipelineResult.
  * Every Gemini event is tagged ``source="pipe={A|C}"`` and carries
    the ``{pipe, target, region}`` metadata the live dashboard
    splits on.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from compgen.bench.four_way import runner_kernelblaster
from compgen.bench.four_way.spec import PipeId, PipelineSpec, TargetId


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("COMPGEN_REPO_ROOT", str(tmp_path))
    return tmp_path


def _patch_kb_workflow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cycles_trajectory: list[int] | None = None,
) -> dict[str, Any]:
    """Stub kb_chipyard's cross_compile + spike + LLM call so the test
    exercises the four_way → kb_chipyard plumbing without subprocess hops.

    Returns a mutable ``captured`` dict the test can introspect to verify
    that the right ChipyardConfig was constructed.
    """
    runner_kernelblaster._ensure_kb_on_path()
    from kernelblaster.kb_chipyard import workflow as wf  # type: ignore[import-not-found]
    from kernelblaster.kb_chipyard.cross_compile import CrossCompileResult  # type: ignore[import-not-found]
    from kernelblaster.kb_chipyard.spike_eval import SpikeEvalResult  # type: ignore[import-not-found]

    cycles_trajectory = cycles_trajectory or [100, 75, 50]
    captured: dict[str, Any] = {"llm_calls": [], "cfg_seen": None}
    spike_idx = {"n": 0}

    def fake_cross_compile(cfg, source_path, **_kw):
        captured["cfg_seen"] = cfg
        elf_path = source_path.with_suffix(".riscv")
        elf_path.write_bytes(b"")
        return CrossCompileResult(
            success=True,
            elf_path=elf_path,
            stdout="",
            stderr="",
            command=["fake-gcc"],
            duration_s=0.001,
        )

    def fake_run_spike(cfg, elf_path):
        i = spike_idx["n"]
        spike_idx["n"] += 1
        c = cycles_trajectory[min(i, len(cycles_trajectory) - 1)]
        return SpikeEvalResult(
            succeeded=True, cycles=c, correct=True,
            stdout=f"CYCLES={c}\nCORRECT=1\n", stderr="", return_code=0,
            duration_s=0.0, failure_reason=None,
        )

    async def fake_call_llm(*, cfg, messages):
        captured["llm_calls"].append({"messages": messages, "cfg": cfg})
        usage = {"prompt_tokens": 1000, "completion_tokens": 200}
        raw = SimpleNamespace(usage=usage, model=cfg.model, elapsed_time=0.01)
        return (
            "```c\nvoid kernel(const int8_t *A, const int8_t *B,\n"
            "            int32_t *C, int M, int N, int K) { /* k */ }\n```",
            usage, 0.01, raw,
        )

    monkeypatch.setattr(wf, "cross_compile", fake_cross_compile)
    monkeypatch.setattr(wf, "run_spike", fake_run_spike)
    monkeypatch.setattr(wf, "_call_llm", fake_call_llm)
    return captured


def test_pipeline_a_runs_unconstrained_and_records_events(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_kb_workflow(monkeypatch, cycles_trajectory=[100, 75, 50])

    spec = PipelineSpec(
        pipe=PipeId.A,
        target=TargetId.GEMMINI,
        region_id="matmul_0",
        problem_spec="int8 matmul 16x16x16",
        constraint_artifacts=None,  # pipeline A: unconstrained
        search_budget="small",
        model="gemini-2.5-flash",
        work_dir=tmp_path / "a",
    )
    result = runner_kernelblaster.run_kernelblaster_pipeline(spec)

    assert result.spec.pipe is PipeId.A
    assert result.succeeded is True
    assert result.best_cycles == 50
    assert result.total_iterations >= 3
    # The right number of tokens flowed through.
    assert result.prompt_tokens == result.total_iterations * 1000
    assert result.completion_tokens == result.total_iterations * 200

    # The ChipyardConfig the workflow saw had no constraint payload.
    cfg = captured["cfg_seen"]
    assert cfg is not None
    assert cfg.constraint_artifacts is None
    assert cfg.usage_source == "pipe=A"

    # And the events.jsonl has one entry per LLM call, tagged correctly.
    events = (isolated_storage / "events.jsonl").read_text().splitlines()
    events = [json.loads(e) for e in events if e]
    assert len(events) == result.total_iterations
    for ev in events:
        assert ev["source"] == "pipe=A"
        assert ev["metadata"]["pipe"] == "A"
        assert ev["metadata"]["target"] == "gemmini"
        assert ev["metadata"]["region"] == "matmul_0"
        assert ev["metadata"]["constrained"] is False


def test_pipeline_c_runs_constrained_and_injects_dossier(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_kb_workflow(monkeypatch, cycles_trajectory=[80, 60, 40])

    constraints = {
        "region_id": "matmul_0",
        "dossier": {"op_family": "matmul", "dtype": "int8",
                    "abstracted_shape": {"M": 128, "N": 64, "K": 64}},
        "contract": {"tile_params": {"BM": 16, "BN": 16, "BK": 16}},
        "readiness": {"precision_budget": "int8_ok"},
    }
    spec = PipelineSpec(
        pipe=PipeId.C,
        target=TargetId.GEMMINI,
        region_id="matmul_0",
        problem_spec="int8 matmul 128x64x64",
        constraint_artifacts=constraints,
        search_budget="small",
        work_dir=tmp_path / "c",
    )
    result = runner_kernelblaster.run_kernelblaster_pipeline(spec)

    assert result.spec.pipe is PipeId.C
    assert result.best_cycles == 40
    cfg = captured["cfg_seen"]
    assert cfg.constraint_artifacts == constraints
    assert cfg.usage_source == "pipe=C"

    # The very first LLM call must have seen the constraint substring
    # in its system prompt — this is the verification-test-#4 invariant.
    first_call = captured["llm_calls"][0]
    system_msg = next(
        m for m in first_call["messages"] if m["role"] == "system"
    )
    assert "Graph dossier" in system_msg["content"]
    assert "matmul" in system_msg["content"]

    # Events carry the constrained=True flag in metadata.
    events = [
        json.loads(line)
        for line in (isolated_storage / "events.jsonl").read_text().splitlines()
        if line
    ]
    assert all(ev["metadata"]["constrained"] is True for ev in events)
    assert all(ev["source"] == "pipe=C" for ev in events)


def test_search_budget_maps_to_max_iterations(
    isolated_storage: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Budget enum is the only knob the driver varies — wire it tightly."""
    captured = _patch_kb_workflow(monkeypatch, cycles_trajectory=[100])

    spec = PipelineSpec(
        pipe=PipeId.A,
        target=TargetId.SATURN,
        region_id="r0",
        problem_spec="x",
        constraint_artifacts=None,
        search_budget="large",
        work_dir=tmp_path / "wf",
    )
    runner_kernelblaster.run_kernelblaster_pipeline(spec)
    assert captured["cfg_seen"].max_iterations == 80


def test_unknown_budget_raises_typed_error(tmp_path: Path) -> None:
    spec = PipelineSpec(
        pipe=PipeId.A, target=TargetId.GEMMINI, region_id="r0",
        problem_spec="x", constraint_artifacts=None,
        search_budget="ENORMOUS",
        work_dir=tmp_path / "wf",
    )
    with pytest.raises(ValueError, match="unknown search_budget"):
        runner_kernelblaster.run_kernelblaster_pipeline(spec)


def test_runner_rejects_autocomp_pipeline(tmp_path: Path) -> None:
    """Calling the KB runner with pipe=B/D is an honest programmer error."""
    spec = PipelineSpec(
        pipe=PipeId.B,  # autocomp pipeline
        target=TargetId.GEMMINI, region_id="r0",
        problem_spec="x", constraint_artifacts=None,
        search_budget="small",
        work_dir=tmp_path / "wf",
    )
    with pytest.raises(ValueError, match="non-KB"):
        runner_kernelblaster.run_kernelblaster_pipeline(spec)
