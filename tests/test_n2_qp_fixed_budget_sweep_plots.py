from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "n2_qp_fixed_budget_sweep_plots.py"
SPEC = importlib.util.spec_from_file_location("n2_qp_fixed_budget_sweep_plots", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_case_matrix_has_expected_size() -> None:
    case = MODULE.threshold.CASES["n2_cas12_12_R2p50"]
    dense = MODULE.threshold.StructuralConfig("dense", "basis_20", 20, 1, 0, 2)
    thick = MODULE.threshold.StructuralConfig("thick", "restart_12_12_4", 12, 12, 12, 4)
    matrix = MODULE._build_case_matrix(
        case,
        dense,
        thick,
        methods=["dense", "thick"],
        store_grid=[1_000_000, 100_000],
        propagate_grid=[10_000, 1_000, 100],
        fixed_store_keep=65_536,
        fixed_propagate_keep=512,
    )
    assert len(matrix) == 10
    assert {row["method"] for row in matrix} == {"dense", "thick"}
    assert {row["sweep_kind"] for row in matrix} == {"fixed_store_keep", "fixed_propagate_keep"}


def test_execution_order_runs_large_to_small() -> None:
    case = MODULE.threshold.CASES["n2_cas12_12_R2p50"]
    dense = MODULE.threshold.StructuralConfig("dense", "basis_20", 20, 1, 0, 2)
    thick = MODULE.threshold.StructuralConfig("thick", "restart_12_12_4", 12, 12, 12, 4)
    matrix = MODULE._build_case_matrix(
        case,
        dense,
        thick,
        methods=["dense", "thick"],
        store_grid=[1_000_000, 100_000],
        propagate_grid=[10_000, 1_000],
        fixed_store_keep=65_536,
        fixed_propagate_keep=512,
    )
    ordered = MODULE._execution_order(matrix, fixed_store_keep=65_536, fixed_propagate_keep=512)
    assert ordered[0]["sweep_kind"] == "fixed_store_keep"
    assert ordered[0]["budget_value"] == 100_000
    assert ordered[-1]["budget_value"] == 10_000


def test_fixed_budget_text_annotations() -> None:
    assert MODULE._fixed_budget_text("fixed_store_keep", 65_536, 512) == "fixed fixed_propagate_keep = 512"
    assert MODULE._fixed_budget_text("fixed_propagate_keep", 65_536, 512) == "fixed fixed_store_keep = 65536"


def test_run_dir_for_uses_stable_budget_tag() -> None:
    run_dir = MODULE._run_dir_for(sweep_kind="fixed_store_keep", method="dense", budget=1_000_000)
    assert run_dir == Path("runs") / "fixed_store_keep" / "dense" / "1000000"


def test_make_fixed_budget_params_uses_fixed_budget_policy() -> None:
    case = MODULE.threshold.CASES["n2_cas12_12_R2p50"]
    config = MODULE.threshold.StructuralConfig("dense", "basis_40", 40, 1, 0, 2)
    params = MODULE._make_fixed_budget_params(case, config, fixed_store_keep=65_536, fixed_propagate_keep=512)
    assert params.ccik_compression_policy == "fixed_budget"
    assert params.ccik_fixed_store_keep == 65_536
    assert params.ccik_fixed_propagate_keep == 512


def test_load_cached_rows_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "fixed_budget_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MODULE.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "case": "n2_cas12_12_R2p50",
                "reference_kind": "dmrg",
                "reference_label": "DMRG",
                "sweep_kind": "fixed_store_keep",
                "budget_name": "fixed_store_keep",
                "budget_value": "1000000",
                "method": "dense",
                "config_tag": "basis_40",
                "fixed_store_keep": "65536",
                "fixed_propagate_keep": "512",
                "e_total_eh": "-1.23",
                "delta_vs_ref_mEh": "0.1",
                "abs_delta_vs_ref_mEh": "0.1",
                "wall_s": "2.5",
                "basis_size": "40",
                "max_cycle": "1",
                "restart_m_cycle": "0",
                "restart_nroot": "2",
                "failure": "",
            }
        )

    rows = MODULE._load_cached_rows(csv_path)
    assert ("fixed_store_keep", 1_000_000, "dense") in rows
    assert rows[("fixed_store_keep", 1_000_000, "dense")]["config_tag"] == "basis_40"


def test_parse_int_grid_accepts_float_like_tokens() -> None:
    assert MODULE._parse_int_grid("1e6, 1e4", MODULE.DEFAULT_STORE_KEEP_GRID) == [1_000_000, 10_000]


def test_build_case_matrix_can_limit_to_thick_only() -> None:
    case = MODULE.threshold.CASES["n2_cas12_12_R2p50"]
    dense = MODULE.threshold.StructuralConfig("dense", "basis_20", 20, 1, 0, 2)
    thick = MODULE.threshold.StructuralConfig("thick", "restart_12_12_4", 12, 12, 12, 4)
    matrix = MODULE._build_case_matrix(
        case,
        dense,
        thick,
        methods=["thick"],
        store_grid=[16_384],
        propagate_grid=[128],
        fixed_store_keep=65_536,
        fixed_propagate_keep=512,
    )
    assert len(matrix) == 2
    assert {row["method"] for row in matrix} == {"thick"}


def test_refresh_outputs_handles_thick_only_summary(tmp_path: Path) -> None:
    case = MODULE.threshold.CASES["n2_cas12_12_R2p50"]
    reference = {
        "reference_kind": "dmrg",
        "reference_label": "DMRG",
        "total_energy_eh": -1.0,
    }
    calibration = {
        "calibration_mode": "manual",
        "dense": {"chosen": {"tag": "basis_40", "basis_size": 40, "max_cycle": 1, "restart_m_cycle": 0, "restart_nroot": 2}},
        "thick": {"chosen": {"tag": "restart_20_20_6", "basis_size": 20, "max_cycle": 20, "restart_m_cycle": 20, "restart_nroot": 6}},
    }
    rows_by_key = {
        ("fixed_store_keep", 65_536, "thick"): {
            "case": case.name,
            "reference_kind": "dmrg",
            "reference_label": "DMRG",
            "sweep_kind": "fixed_store_keep",
            "budget_name": "fixed_store_keep",
            "budget_value": 65_536,
            "method": "thick",
            "config_tag": "restart_20_20_6",
            "fixed_store_keep": 65_536,
            "fixed_propagate_keep": 512,
            "e_total_eh": -1.0,
            "delta_vs_ref_mEh": 0.1,
            "abs_delta_vs_ref_mEh": 0.1,
            "wall_s": 1.0,
            "basis_size": 20,
            "max_cycle": 20,
            "restart_m_cycle": 20,
            "restart_nroot": 6,
            "failure": None,
        }
    }
    MODULE._refresh_outputs(
        outdir=tmp_path,
        case=case,
        reference=reference,
        calibration=calibration,
        rows_by_key=rows_by_key,
        total_rows=2,
        fixed_store_keep=65_536,
        fixed_propagate_keep=512,
        store_grid=[65_536],
        propagate_grid=[512],
        methods=["thick"],
        csv_path=tmp_path / "fixed_budget_sweep.csv",
        progress_path=tmp_path / "progress.json",
    )
    summary = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert '"methods": [' in summary
    assert '"thick"' in summary
