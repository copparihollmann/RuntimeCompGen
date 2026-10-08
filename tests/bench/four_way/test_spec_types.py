"""Sanity checks for the four_way PipelineSpec / PipelineResult types."""

from __future__ import annotations

from compgen.bench.four_way.spec import (
    PipeId,
    PipelineResult,
    PipelineSpec,
    TargetId,
)


def test_pipe_id_classifies_optimizer_and_constrained() -> None:
    assert PipeId.A.optimizer == "kernelblaster"
    assert PipeId.B.optimizer == "autocomp"
    assert PipeId.C.optimizer == "kernelblaster"
    assert PipeId.D.optimizer == "autocomp"

    assert PipeId.A.constrained is False
    assert PipeId.B.constrained is False
    assert PipeId.C.constrained is True
    assert PipeId.D.constrained is True

    assert PipeId.A.short == "A"
    assert PipeId.D.short == "D"
    assert PipeId.A.value == "pipe=A"


def test_spec_usage_metadata_carries_pipeline_target_region_and_budget() -> None:
    spec = PipelineSpec(
        pipe=PipeId.C,
        target=TargetId.GEMMINI,
        region_id="matmul_0",
        problem_spec="x",
        constraint_artifacts={"region_id": "matmul_0"},
        search_budget="med",
        extra_metadata={"run_id": "exp_2026-05-15"},
    )
    md = spec.usage_metadata()
    assert md["pipe"] == "C"
    assert md["target"] == "gemmini"
    assert md["region"] == "matmul_0"
    assert md["budget"] == "med"
    assert md["optimizer"] == "kernelblaster"
    assert md["constrained"] is True
    assert md["run_id"] == "exp_2026-05-15"


def test_result_to_dict_round_trips_to_jsonl_shape() -> None:
    spec = PipelineSpec(
        pipe=PipeId.A,
        target=TargetId.SATURN,
        region_id="matmul_3",
        problem_spec="x",
        constraint_artifacts=None,
        search_budget="small",
        model="gemini-2.5-flash",
    )
    result = PipelineResult(
        spec=spec,
        succeeded=True,
        best_kernel_source="// kernel",
        best_cycles=4242,
        total_iterations=12,
        prompt_tokens=10_000,
        completion_tokens=2_000,
        n_llm_calls=12,
        wall_seconds=30.5,
    )
    d = result.to_dict()
    # The shape the bench's runs.jsonl will consume.
    expected_keys = {
        "pipe", "target", "region", "budget", "model", "constrained",
        "succeeded", "best_cycles", "total_iterations", "prompt_tokens",
        "completion_tokens", "total_tokens", "n_llm_calls", "wall_seconds",
        "failure_reason", "extra",
    }
    assert set(d.keys()) == expected_keys
    assert d["pipe"] == "A"
    assert d["constrained"] is False
    assert d["total_tokens"] == 12_000
