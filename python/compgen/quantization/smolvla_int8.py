"""SmolVLA end-to-end int8 quantization for Gemmini and Saturn/OPU.

Branches :mod:`compgen.quantization.smolvla_e2e` (which targets the NPU's
FP8 E4M3 po2 path) and swaps the scheme to **int8 per-channel weights +
per-tensor dynamic int8 activations** via TorchAO's
``Int8DynamicActivationInt8WeightConfig``.

This matches:

* Gemmini's ``elem_t = int8_t`` systolic input (``DIM=16`` tile,
  int32 accumulator) — see
  ``/scratch2/agustin/chipyard/generators/gemmini/software/libgemmini/gemmini_params.h``.
* Saturn/OPU's ``vsetvli e8 m1`` int8 vector lanes with VOPACC int32
  accumulators — see
  ``/scratch2/agustin/chipyard/generators/saturn/benchmarks/opu-gemm/kernel.h``.

The resulting artifact bundle is the constraint substrate that the
4-way comparison feeds to pipelines C and D (CompGen-constrained
KernelBlaster + Autocomp): ``payload.mlir``, ``graph_dossier_v3.json``,
``kernel_contracts/*.yaml``. Pipelines A and B (vanilla KB / Autocomp)
ignore these.

Usage::

    python -m compgen.quantization.smolvla_int8 --output-dir artifacts/smolvla_int8

Or from Python::

    from compgen.quantization.smolvla_int8 import run_smolvla_int8_pipeline
    report = run_smolvla_int8_pipeline()
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from compgen.capture.torchao_pipeline import QuantizationConfig
from compgen.quantization.graph_analyzer import format_analysis_report
from compgen.quantization.pipeline import QuantizedModelPipeline

logger = structlog.get_logger()

#: The scheme name used by :func:`compgen.capture.torchao_pipeline.apply_quantization`.
#: Resolves to ``torchao.quantization.Int8DynamicActivationInt8WeightConfig``.
INT8_SCHEME = "int8_dyn_act_int8_weight"


def run_smolvla_int8_pipeline(
    output_dir: str | Path = "artifacts/smolvla_int8",
    device: str = "cpu",
    model_name: str = "smolvla_int8",
) -> object:
    """Run the full smolVLA int8 pipeline.

    Steps mirror :func:`compgen.quantization.smolvla_e2e.run_smolvla_npu_pipeline`,
    swapping the quantization scheme to int8/int8 for Gemmini + Saturn/OPU.

    Args:
        output_dir: Directory for artifact output.
        device: Device for model loading. ``"cpu"`` is recommended for
            capture; the int8 weights ship dtype-correct regardless.
        model_name: Identifier embedded in the pipeline report.

    Returns:
        ``PipelineReport`` with the bundle, kernel contracts, FX graph
        analysis, and Payload IR. The bundle directory is also written
        to ``output_dir`` and contains the constraint artifacts pipelines
        C and D will read.
    """
    from compgen.models.robotics import load_smolvla_bundle

    logger.info("smolvla_int8_start", output_dir=str(output_dir), device=device)

    wrapper, flat_inputs, num_cams = load_smolvla_bundle(device=device)
    param_count = sum(p.numel() for p in wrapper.parameters())
    logger.info(
        "smolvla_loaded",
        params=f"{param_count:,}",
        input_tensors=len(flat_inputs),
        num_cams=num_cams,
    )

    pipeline = QuantizedModelPipeline(
        model=wrapper,
        sample_inputs=flat_inputs,
        model_name=model_name,
        quant_config=QuantizationConfig(scheme=INT8_SCHEME),
        output_dir=output_dir,
    )
    report = pipeline.run()

    # The 4-way benchmark consumes kernel contracts directly; generate
    # them in the same place smolvla_e2e does so the downstream
    # constraint loader doesn't care which scheme produced the bundle.
    kernel_contracts_list: list[object] = []
    if report.capture_artifact and report.capture_artifact.graphs:
        from compgen.kernels.providers.npu_contracts import (
            export_contracts_autocomp,
            export_contracts_yaml,
            generate_npu_kernel_contracts,
        )

        logger.info("generating_kernel_contracts")
        kernel_contracts_list = generate_npu_kernel_contracts(
            list(report.capture_artifact.graphs),
            model=wrapper,
        )
        logger.info("kernel_contracts_generated", count=len(kernel_contracts_list))

        contracts_dir = Path(output_dir) / "kernel_contracts"
        export_contracts_yaml(kernel_contracts_list, contracts_dir)

        autocomp_dir = Path(output_dir) / "autocomp_problems"
        export_contracts_autocomp(kernel_contracts_list, autocomp_dir)

        logger.info(
            "kernel_contracts_exported",
            yaml_dir=str(contracts_dir),
            autocomp_dir=str(autocomp_dir),
        )

    print("\n" + "=" * 70)
    print("  SmolVLA int8 Pipeline — Results (Gemmini + Saturn/OPU)")
    print("=" * 70)
    print(f"\n  {report.summary()}")

    if report.graph_analysis:
        print(f"\n{format_analysis_report(report.graph_analysis)}")

    if report.artifact_dir:
        print(f"\n  Artifacts saved to: {report.artifact_dir}")
        print(f"  Scheme: {INT8_SCHEME}")

    if report.errors:
        print(f"\n  Errors ({len(report.errors)}):")
        for err in report.errors:
            print(f"    - {err}")

    print("\n  Timings:")
    for step, t in sorted(report.timings.items()):
        print(f"    {step}: {t:.2f}s")

    print("\n" + "=" * 70)
    return report


def main() -> None:
    """CLI entry point for `python -m compgen.quantization.smolvla_int8`."""
    parser = argparse.ArgumentParser(
        description=(
            "SmolVLA end-to-end int8 quantization + capture for Gemmini "
            "and Saturn/OPU. Emits payload.mlir + graph_dossier_v3.json + "
            "kernel_contracts/*.yaml that the 4-way benchmark consumes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/smolvla_int8",
        help="Directory for artifact output (default: artifacts/smolvla_int8).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading (default: cpu).",
    )
    args = parser.parse_args()

    run_smolvla_int8_pipeline(output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
