"""Benchmark runner — orchestrate full pipeline and record all metrics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog

from benchmarks.collector import (
    collect_capture_metrics,
    collect_eqsat_metrics,
    collect_ir_metrics,
    collect_recipe_metrics,
)
from benchmarks.record import RunRecord

log = structlog.get_logger()

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def run_benchmark(
    model_name: str,
    target_spec_path: str,
    *,
    objective: str = "latency",
    output_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> RunRecord:
    """Run a full compilation benchmark and record all metrics.

    Args:
        model_name: Name of the model to benchmark (e.g., "simple_mlp").
        target_spec_path: Path to target YAML spec.
        objective: Optimization objective.
        output_dir: Where to save results JSON.
        config: Additional configuration overrides.

    Returns:
        RunRecord with all metrics populated.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_RESULTS_DIR
    record = RunRecord(
        model_name=model_name,
        target_name=Path(target_spec_path).stem,
        objective=objective,
        config=config or {},
    )

    total_start = time.perf_counter()

    try:
        # Stage 1: Load target
        log.info("benchmark.stage", stage="target_load", model=model_name)
        from compgen.api import device
        target_device = device(target_spec_path)

        # Stage 2: Load model
        log.info("benchmark.stage", stage="model_load", model=model_name)
        model, sample_inputs = _load_model(model_name)

        # Stage 3: Capture
        log.info("benchmark.stage", stage="capture", model=model_name)
        capture_start = time.perf_counter()
        from compgen.capture.torch_export import capture_model
        exported = capture_model(model, sample_inputs)
        capture_ms = (time.perf_counter() - capture_start) * 1000

        record.capture = collect_capture_metrics(
            export_success=exported is not None,
            export_time_ms=capture_ms,
        )

        if exported is None:
            record.errors.append("torch.export failed")
            record.save(output_dir)
            return record

        # Stage 4: FX→xDSL
        log.info("benchmark.stage", stage="import_fx", model=model_name)
        from compgen.ir.payload.import_fx import fx_to_xdsl
        module, diagnostics = fx_to_xdsl(exported)

        record.capture.decomposition_coverage = sum(
            1 for d in diagnostics if d.level != "error"
        ) / max(len(diagnostics), 1)
        record.ir = collect_ir_metrics(module)

        # Stage 5: EqSat
        log.info("benchmark.stage", stage="eqsat", model=model_name)
        eqsat_start = time.perf_counter()
        from compgen.eqsat.pipeline import run_eqsat_pass
        eqsat_result = run_eqsat_pass(module)
        eqsat_ms = (time.perf_counter() - eqsat_start) * 1000

        record.eqsat = collect_eqsat_metrics(eqsat_result, eqsat_ms)

        # Stage 6: Recipe seed
        log.info("benchmark.stage", stage="recipe_seed", model=model_name)
        seed_start = time.perf_counter()
        from compgen.ir.recipe.seed import generate_seed_recipe
        recipe_module = generate_seed_recipe(module, target_device.profile)
        seed_ms = (time.perf_counter() - seed_start) * 1000

        record.recipe = collect_recipe_metrics(recipe_module)
        record.recipe.seed_generation_time_ms = seed_ms

        # Stage 7: Recipe validation
        from compgen.ir.recipe.validate import validate_recipe_module
        validation = validate_recipe_module(recipe_module)
        record.recipe.validation_passed = validation.valid
        record.recipe.validation_errors = len(validation.errors)

        # Stage 8: Recipe lowering
        from compgen.ir.recipe.lower import lower_recipe
        lowered = lower_recipe(recipe_module)
        record.recipe.transform_scripts_count = len(lowered.transform_scripts)
        record.recipe.kernel_jobs_count = len(lowered.kernel_jobs)
        record.recipe.plan_fragments_count = len(lowered.plan_fragments)
        record.recipe.verification_obligations_count = len(lowered.verification_obligations)
        record.recipe.eqsat_jobs_count = len(lowered.eqsat_jobs)
        record.recipe.lowering_diagnostics = len(lowered.diagnostics)

    except Exception as e:
        record.errors.append(f"Pipeline error: {e}")
        log.error("benchmark.error", error=str(e), model=model_name)

    record.total_compile_time_ms = (time.perf_counter() - total_start) * 1000
    path = record.save(output_dir)
    log.info("benchmark.done", model=model_name, path=str(path), time_ms=record.total_compile_time_ms)
    return record


def _load_model(model_name: str) -> tuple[Any, Any]:
    """Load a model by name from examples."""

    if model_name == "simple_mlp":
        from examples.models.simple_mlp import SimpleMLP, get_sample_inputs
        return SimpleMLP(), get_sample_inputs()
    elif model_name == "transformer_block":
        from examples.models.transformer_block import TransformerBlock, get_sample_inputs
        return TransformerBlock(), get_sample_inputs()
    elif model_name == "quantized_mlp":
        from examples.models.quantized_mlp import QuantizedMLP, get_sample_inputs
        return QuantizedMLP(), get_sample_inputs()
    else:
        # Generic: try importing from examples.models.{name}
        import importlib
        mod = importlib.import_module(f"examples.models.{model_name}")
        model_cls = getattr(mod, model_name.title().replace("_", ""))
        get_inputs = getattr(mod, "get_sample_inputs")
        return model_cls(), get_inputs()


def run_ablation(
    model_name: str,
    target_spec_path: str,
    *,
    ablations: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[RunRecord]:
    """Run benchmark with components selectively disabled.

    Ablations: "no_eqsat", "no_recipe", "no_solver", "baseline_only"
    """
    ablations = ablations or ["full", "no_eqsat", "no_recipe"]
    records = []
    for ablation in ablations:
        config = {"ablation": ablation}
        record = run_benchmark(
            model_name, target_spec_path,
            config=config,
            output_dir=output_dir,
        )
        records.append(record)
    return records


__all__ = ["run_ablation", "run_benchmark"]
