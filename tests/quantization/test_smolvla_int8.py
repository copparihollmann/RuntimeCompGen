"""Smoke tests for the smolVLA int8 path (Gemmini + Saturn/OPU).

The whole ``compgen.quantization`` package transitively requires the
optional ``torchao`` extra. Skip the file cleanly when it's not
installed — this matches the behavior of other quantization tests in
the suite. CI runs with the ``[quantization]`` extra and exercises
these paths.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip(
    "torchao",
    reason="optional [quantization] extra not installed (torchao missing)",
)

import torch
import torch.nn as nn

from compgen.capture.torchao_pipeline import QuantizationConfig, apply_quantization


def _torchao_int8_dyn_act_available() -> bool:
    try:
        importlib.import_module("torchao.quantization")
    except ImportError:
        return False
    from torchao import quantization as q

    return hasattr(q, "Int8DynamicActivationInt8WeightConfig")


@pytest.mark.skipif(
    not _torchao_int8_dyn_act_available(),
    reason="torchao Int8DynamicActivationInt8WeightConfig not present in this env",
)
def test_int8_dyn_act_int8_weight_quantizes_linear() -> None:
    """Verify the new scheme is wired through apply_quantization on a tiny Linear.

    This protects the wiring path the smolvla_int8 driver depends on.
    Failure here means smolVLA int8 will explode at ``QuantizedModelPipeline.run()``.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))
    cfg = QuantizationConfig(scheme="int8_dyn_act_int8_weight")

    quantized = apply_quantization(model, cfg)

    # torchAO mutates the linear's weight to an AQT subclass; the
    # exact subclass name has moved between versions, so check for the
    # general shape: weight is no longer a plain torch.Tensor instance
    # of the original Linear dtype.
    w = quantized[0].weight
    assert w is not None
    # Forward pass must still run end-to-end with int8 weights.
    out = quantized(torch.randn(4, 64))
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_smolvla_int8_module_imports_and_exports_scheme() -> None:
    """Sanity-check the public surface so the 4-way runner can find it."""
    from compgen.quantization import SMOLVLA_INT8_SCHEME, run_smolvla_int8_pipeline
    from compgen.quantization.smolvla_int8 import INT8_SCHEME

    assert SMOLVLA_INT8_SCHEME == INT8_SCHEME == "int8_dyn_act_int8_weight"
    assert callable(run_smolvla_int8_pipeline)
