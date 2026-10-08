"""Sanity checks for the ChipyardTarget enum + ChipyardConfig defaults."""

from __future__ import annotations

from pathlib import Path

from kernelblaster.kb_chipyard.config import ChipyardConfig
from kernelblaster.kb_chipyard.target import ChipyardTarget, TargetShape


def test_target_shape_gemmini_has_int8_tile_dim_16() -> None:
    shape = TargetShape.for_target(ChipyardTarget.GEMMINI)
    assert shape.elem_bits == 8
    assert shape.accum_bits == 32
    assert shape.tile_dim == 16
    assert shape.vlen_bits is None
    assert shape.march_flag == "-march=rv64gc"


def test_target_shape_saturn_has_int8_vlen_128() -> None:
    shape = TargetShape.for_target(ChipyardTarget.SATURN)
    assert shape.elem_bits == 8
    assert shape.accum_bits == 32
    assert shape.tile_dim is None
    assert shape.vlen_bits == 128
    # Saturn must request the V extension + zfh.
    assert "v" in shape.march_flag
    assert "zfh" in shape.march_flag


def test_config_gcc_and_spike_paths_resolve_to_chipyard_conda_env() -> None:
    cfg = ChipyardConfig(target=ChipyardTarget.GEMMINI)
    assert cfg.gcc_path().name == "riscv64-unknown-elf-gcc"
    assert cfg.spike_path().name == "spike"
    # The defaults must point at the live chipyard conda env so the
    # four-way bench works out-of-the-box on this machine.
    assert cfg.gcc_path() == Path(
        "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
    )


def test_config_defaults_constraint_artifacts_to_none() -> None:
    cfg = ChipyardConfig(target=ChipyardTarget.GEMMINI)
    # Unconstrained (pipeline A) is the default; the runner adapter
    # explicitly opts pipeline C in.
    assert cfg.constraint_artifacts is None
