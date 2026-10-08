"""Constraint slicing tests — pipelines C/D depend on this being honest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.bench.four_way.constraints import (
    RegionConstraints,
    _select_dossier_row,
    build_region_constraints,
)


def test_select_dossier_row_finds_match_in_regions_list() -> None:
    dossier = {
        "regions": [
            {"region_id": "matmul_0", "op_family": "matmul"},
            {"region_id": "matmul_1", "op_family": "matmul"},
        ]
    }
    row = _select_dossier_row(dossier, "matmul_1")
    assert row["region_id"] == "matmul_1"


def test_select_dossier_row_tolerates_id_key_alias() -> None:
    """Some dossier emitters use ``id`` instead of ``region_id``."""
    dossier = [{"id": "matmul_3", "op_family": "matmul"}]
    row = _select_dossier_row(dossier, "matmul_3")
    assert row["id"] == "matmul_3"


def test_select_dossier_row_returns_empty_marker_when_missing() -> None:
    """Missing regions get an honest empty marker, not a fabricated row."""
    dossier = {"regions": [{"region_id": "matmul_0"}]}
    row = _select_dossier_row(dossier, "does_not_exist")
    assert row["region_id"] == "does_not_exist"
    assert "_note" in row  # honest "not found" marker


def test_build_region_constraints_assembles_from_bundle(tmp_path: Path) -> None:
    """Full happy path: dossier + contract + readiness all resolve."""
    bundle = tmp_path / "smolvla_int8"
    (bundle / "kernel_contracts").mkdir(parents=True)

    (bundle / "graph_dossier_v3.json").write_text(json.dumps({
        "regions": [
            {
                "region_id": "matmul_0",
                "op_family": "matmul",
                "dtype": "int8",
                "abstracted_shape": {"M": 128, "N": 64, "K": 64},
                "hot_axis": "K",
            }
        ]
    }))
    (bundle / "kernel_contracts" / "matmul_0.json").write_text(json.dumps({
        "layouts": {"A": "row_major", "B": "col_major"},
        "tile_params": {"BM": 16, "BN": 16, "BK": 16},
    }))
    (bundle / "graph_analysis_readiness_matrix.json").write_text(json.dumps({
        "matmul_0": {"precision_budget": "int8_ok", "working_set_fit": "ok"}
    }))

    rc = build_region_constraints(bundle_dir=bundle, region_id="matmul_0")

    assert isinstance(rc, RegionConstraints)
    assert rc.dossier["op_family"] == "matmul"
    assert rc.contract["tile_params"]["BM"] == 16
    assert rc.readiness is not None
    assert "int8 matmul M=128 N=64 K=64" in rc.problem_spec

    payload = rc.to_payload()
    assert payload["region_id"] == "matmul_0"
    assert payload["dossier"]["dtype"] == "int8"
    assert payload["contract"]["layouts"]["A"] == "row_major"


def test_build_region_constraints_raises_when_dossier_missing(tmp_path: Path) -> None:
    """No dossier → no constraint substrate. Honest error, not silent stub."""
    with pytest.raises(FileNotFoundError) as exc:
        build_region_constraints(bundle_dir=tmp_path, region_id="matmul_0")
    assert "smolvla_int8" in str(exc.value) or "graph_dossier_v3" in str(exc.value)


def test_build_region_constraints_tolerates_missing_contract(tmp_path: Path) -> None:
    """If a region has no contract file, the dossier still drives prompts."""
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "graph_dossier_v3.json").write_text(json.dumps({
        "regions": [{"region_id": "matmul_0", "op_family": "matmul", "dtype": "int8"}]
    }))
    rc = build_region_constraints(bundle_dir=bundle, region_id="matmul_0")
    assert rc.contract == {}
    assert rc.dossier["dtype"] == "int8"


def test_build_region_constraints_finds_alt_layout_02_graph_analysis(tmp_path: Path) -> None:
    """Some bundles emit dossier under 02_graph_analysis/. Find it there too."""
    bundle = tmp_path / "b"
    (bundle / "02_graph_analysis").mkdir(parents=True)
    (bundle / "02_graph_analysis" / "graph_dossier_v3.json").write_text(json.dumps({
        "regions": [{"region_id": "r0", "op_family": "matmul"}]
    }))
    rc = build_region_constraints(bundle_dir=bundle, region_id="r0")
    assert rc.dossier["op_family"] == "matmul"
