"""Prompt-building tests.

Verification test #4 in the plan asserts that pipelines C/D actually
inject the dossier+contract into the prompts (so the constraint-effect
claim isn't vacuous). These tests cover that boundary.
"""

from __future__ import annotations

from kernelblaster.kb_chipyard.cross_compile import CrossCompileResult
from kernelblaster.kb_chipyard.prompts.common import (
    build_prompt_bundle,
    format_constraint_block,
)
from kernelblaster.kb_chipyard.spike_eval import SpikeEvalResult
from kernelblaster.kb_chipyard.target import ChipyardTarget


CONSTRAINTS = {
    "region_id": "matmul_0",
    "dossier": {
        "op_family": "matmul",
        "dtype": "int8",
        "abstracted_shape": {"M": 128, "K": 64, "N": 64},
        "hot_axis": "K",
    },
    "contract": {
        "layouts": {"A": "row_major", "B": "col_major"},
        "tile_params": {"BM": 16, "BN": 16, "BK": 16},
    },
    "readiness": {"precision_budget": "int8", "working_set_fit": "ok"},
}


def test_constraint_block_includes_all_three_artifacts() -> None:
    block = format_constraint_block(CONSTRAINTS)
    assert "Graph dossier" in block
    assert "Kernel contract" in block
    assert "Readiness report" in block
    assert "matmul_0" in block
    assert "BM" in block  # tile params landed


def test_constraint_block_is_empty_for_unconstrained() -> None:
    assert format_constraint_block(None) == ""
    assert format_constraint_block({}) == ""


def test_pipeline_a_and_c_prompts_differ_for_same_problem() -> None:
    """The crux test for the constraint-effect claim.

    Pipeline A (unconstrained) and pipeline C (constrained) sharing a
    problem spec must produce different system prompts. If they're
    identical the constraint injection is dead — the four-way bench's
    claim is vacuous.
    """
    a = build_prompt_bundle(
        target=ChipyardTarget.GEMMINI,
        problem_spec="int8 matmul M=128 N=64 K=64",
        iteration=0,
        history=[],
        constraints=None,
        best_cycles=None,
    )
    c = build_prompt_bundle(
        target=ChipyardTarget.GEMMINI,
        problem_spec="int8 matmul M=128 N=64 K=64",
        iteration=0,
        history=[],
        constraints=CONSTRAINTS,
        best_cycles=None,
    )
    assert a.constrained is False
    assert c.constrained is True
    assert a.system != c.system
    assert "Graph dossier" in c.system
    assert "Graph dossier" not in a.system
    # User message also picks up constraints on iter 0.
    assert a.user != c.user


def test_gemmini_system_prompt_pins_DIM_16_int8() -> None:
    bundle = build_prompt_bundle(
        target=ChipyardTarget.GEMMINI,
        problem_spec="any",
        iteration=0,
        history=[],
        constraints=None,
        best_cycles=None,
    )
    # The shape constants must land in the prompt verbatim so the LLM
    # doesn't invent an int4 systolic dim or a wrong tile size.
    assert "DIM = 16" in bundle.system
    assert "int8_t" in bundle.system
    assert "RoCC" in bundle.system


def test_saturn_system_prompt_pins_vlen_128_rvv() -> None:
    bundle = build_prompt_bundle(
        target=ChipyardTarget.SATURN,
        problem_spec="any",
        iteration=0,
        history=[],
        constraints=None,
        best_cycles=None,
    )
    assert "VLEN = 128" in bundle.system
    assert "RVV 1.0" in bundle.system
    assert "VOPACC" in bundle.system


def test_feedback_message_includes_compile_stderr_on_compile_failure() -> None:
    """When a previous iteration failed to compile, the LLM must see why."""
    cc = CrossCompileResult(
        success=False,
        elf_path=None,
        stdout="",
        stderr="iter_001.c:3:5: error: 'foo' undeclared",
        command=["gcc"],
        duration_s=0.5,
    )
    history = [{
        "compile": cc,
        "spike": None,
        "kernel_source": "void kernel(...) { foo(); }",
    }]
    bundle = build_prompt_bundle(
        target=ChipyardTarget.GEMMINI,
        problem_spec="anything",
        iteration=1,
        history=history,
        constraints=None,
        best_cycles=None,
    )
    assert "'foo' undeclared" in bundle.user
    assert "## Compile feedback" in bundle.user


def test_feedback_message_includes_cycles_on_successful_iteration() -> None:
    """A successful but suboptimal run must surface the cycle count.

    Without this the LLM has no signal to drive cycles down — the
    "RL" reduces to pure structural variation.
    """
    sp = SpikeEvalResult(
        succeeded=True,
        cycles=4242,
        correct=True,
        stdout="CYCLES=4242\nCORRECT=1\n",
        stderr="",
        return_code=0,
        duration_s=0.1,
        failure_reason=None,
    )
    history = [{
        "compile": CrossCompileResult(
            success=True, elf_path=None, stdout="", stderr="",
            command=["gcc"], duration_s=0.5,
        ),
        "spike": sp,
        "kernel_source": "void kernel(...) { /* slow */ }",
    }]
    bundle = build_prompt_bundle(
        target=ChipyardTarget.SATURN,
        problem_spec="x",
        iteration=1,
        history=history,
        constraints=None,
        best_cycles=4242,
    )
    assert "4242" in bundle.user
    assert "Spike feedback" in bundle.user
