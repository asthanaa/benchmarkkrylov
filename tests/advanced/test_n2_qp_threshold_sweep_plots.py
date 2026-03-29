from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "advanced" / "n2_qp_threshold_sweep_plots.py"
SPEC = importlib.util.spec_from_file_location("n2_qp_threshold_sweep_plots", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_case_matrix_has_expected_size() -> None:
    case = MODULE.CASES["n2_cas12_12_R2p50"]
    dense = MODULE.StructuralConfig("dense", "basis_20", 20, 1, 0, 2)
    thick = MODULE.StructuralConfig("thick", "restart_12_12_4", 12, 12, 12, 4)
    matrix = MODULE._build_case_matrix(case, dense, thick)
    assert len(matrix) == 36
    assert {row["method"] for row in matrix} == {"dense", "thick"}
    assert {row["sweep_kind"] for row in matrix} == {"store_abs_tol", "propagate_abs_tol"}


def test_execution_order_runs_loose_to_tight() -> None:
    case = MODULE.CASES["n2_cas12_12_R2p50"]
    dense = MODULE.StructuralConfig("dense", "basis_20", 20, 1, 0, 2)
    thick = MODULE.StructuralConfig("thick", "restart_12_12_4", 12, 12, 12, 4)
    matrix = MODULE._build_case_matrix(case, dense, thick)
    ordered = MODULE._execution_order(matrix)
    first = ordered[0]
    last = ordered[-1]
    assert first["sweep_kind"] == "store_abs_tol"
    assert first["threshold_value"] == 1e-2
    assert last["threshold_value"] == 1e-6


def test_fixed_threshold_text_annotations() -> None:
    assert MODULE._fixed_threshold_text("store_abs_tol") == "fixed propagate_abs_tol = 1e-8"
    assert MODULE._fixed_threshold_text("propagate_abs_tol") == "fixed store_abs_tol = 1e-8"


def test_run_dir_for_uses_stable_threshold_tag() -> None:
    case = MODULE.CASES["n2_cas14_14_R2p50"]
    run_dir = MODULE._run_dir_for(case, sweep_kind="store_abs_tol", method="dense", threshold=1.0e-5)
    assert run_dir == Path("runs") / "store_abs_tol" / "dense" / "1e-05"


def test_load_or_compute_reference_prefers_dmrg_when_requested(monkeypatch, tmp_path: Path) -> None:
    case = MODULE.CASES["n2_cas12_12_R2p50"]

    def fake_dmrg(**kwargs):
        return {"reference_kind": "dmrg", "reference_label": "DMRG", "total_energy_eh": -1.23}

    monkeypatch.setattr(MODULE, "_load_or_compute_dmrg_reference", fake_dmrg)

    ref = MODULE._load_or_compute_reference(
        case=case,
        outdir=tmp_path,
        force=False,
        dmrg_memory_gb=64,
        dmrg_threads=16,
        dmrg_conv_tol=1e-8,
        dmrg_python=Path("/tmp/unused"),
        reference_kind="dmrg",
    )
    assert ref["reference_kind"] == "dmrg"


def test_normalize_reference_summary_upgrades_legacy_dmrg_payload() -> None:
    summary = MODULE._normalize_reference_summary(
        {
            "method": "dmrg",
            "total_energy_eh": -1.23,
        }
    )
    assert summary["reference_kind"] == "dmrg"
    assert summary["reference_label"] == "DMRG"


def test_load_or_compute_reference_falls_back_to_dmrg_for_cas14(monkeypatch, tmp_path: Path) -> None:
    case = MODULE.CASES["n2_cas14_14_R2p50"]

    def fake_attempt(*args, **kwargs):
        return {"status": "timeout", "timeout_s": 7200}

    def fake_dmrg(**kwargs):
        return {"reference_kind": "dmrg", "reference_label": "DMRG", "total_energy_eh": -2.34}

    monkeypatch.setattr(MODULE, "_attempt_exact_reference_with_fallback", fake_attempt)
    monkeypatch.setattr(MODULE, "_load_or_compute_dmrg_reference", fake_dmrg)

    ref = MODULE._load_or_compute_reference(
        case=case,
        outdir=tmp_path,
        force=False,
        dmrg_memory_gb=64,
        dmrg_threads=16,
        dmrg_conv_tol=1e-8,
        dmrg_python=Path("/tmp/unused"),
        reference_kind="auto",
    )
    assert ref["reference_kind"] == "dmrg"


def test_load_or_compute_reference_auto_prefers_exact_when_available(monkeypatch, tmp_path: Path) -> None:
    case = MODULE.CASES["n2_cas14_14_R2p50"]

    def fake_attempt(*args, **kwargs):
        return {"reference_kind": "exact_fci", "reference_label": "Exact FCI", "total_energy_eh": -3.45}

    def fail_dmrg(**kwargs):
        raise AssertionError("DMRG should not be used when exact FCI is available in auto mode")

    monkeypatch.setattr(MODULE, "_attempt_exact_reference_with_fallback", fake_attempt)
    monkeypatch.setattr(MODULE, "_load_or_compute_dmrg_reference", fail_dmrg)

    ref = MODULE._load_or_compute_reference(
        case=case,
        outdir=tmp_path,
        force=False,
        dmrg_memory_gb=64,
        dmrg_threads=16,
        dmrg_conv_tol=1e-8,
        dmrg_python=Path("/tmp/unused"),
        reference_kind="auto",
    )
    assert ref["reference_kind"] == "exact_fci"


def test_hf_occ_vector_closed_shell() -> None:
    assert MODULE._hf_occ_vector(6, 8) == [2.0, 2.0, 2.0, 2.0, 0.0, 0.0]


def test_hf_occ_vector_open_shell() -> None:
    assert MODULE._hf_occ_vector(5, 7) == [2.0, 2.0, 2.0, 1.0, 0.0]


def test_select_config_picks_first_rung_with_small_energy_change(monkeypatch, tmp_path: Path) -> None:
    case = MODULE.CASES["n2_cas12_12_R2p50"]
    ladder = MODULE._dense_ladder(case)
    energies = {
        "basis_20": -1.0000,
        "basis_30": -1.00005,
        "basis_40": -1.00007,
    }

    def fake_run_single_qp_job(**kwargs):
        config = kwargs["config"]
        return {
            "config_tag": config.tag,
            "e_total_eh": energies[config.tag],
        }

    monkeypatch.setattr(MODULE, "_run_single_qp_job", fake_run_single_qp_job)
    chosen, results = MODULE._select_config(
        case=case,
        method="dense",
        ladder=ladder,
        reference={"total_energy_eh": -1.1},
        qp_root=Path("/tmp/qp2"),
        outdir=tmp_path,
        omp_threads=24,
        force=False,
    )
    assert chosen.tag == "basis_20"
    assert len(results) == 3


def test_load_cached_rows_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "threshold_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MODULE.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "case": "n2_cas12_12_R2p50",
                "reference_kind": "exact_fci",
                "reference_label": "Exact FCI",
                "sweep_kind": "store_abs_tol",
                "threshold_name": "store_abs_tol",
                "threshold_value": "1e-05",
                "method": "dense",
                "config_tag": "basis_20",
                "store_abs_tol": "1e-05",
                "propagate_abs_tol": "1e-08",
                "e_total_eh": "-1.23",
                "delta_vs_ref_mEh": "0.1",
                "abs_delta_vs_ref_mEh": "0.1",
                "wall_s": "2.5",
                "basis_size": "20",
                "max_cycle": "1",
                "restart_m_cycle": "0",
                "restart_nroot": "2",
                "failure": "",
            }
        )

    rows = MODULE._load_cached_rows(csv_path)
    assert ("store_abs_tol", 1.0e-5, "dense") in rows
    assert rows[("store_abs_tol", 1.0e-5, "dense")]["config_tag"] == "basis_20"


def test_infer_smaller_threshold_failures_marks_lower_thresholds() -> None:
    case = MODULE.CASES["n2_cas12_12_R2p50"]
    dense = MODULE.StructuralConfig("dense", "basis_20", 20, 1, 0, 2)
    thick = MODULE.StructuralConfig("thick", "restart_12_12_4", 12, 12, 12, 4)
    matrix = MODULE._build_case_matrix(case, dense, thick)
    failed_job = next(
        row
        for row in matrix
        if row["sweep_kind"] == "store_abs_tol" and row["method"] == "dense" and row["threshold_value"] == 1e-5
    )
    rows_by_key = {
        ("store_abs_tol", 1.0e-5, "dense"): {
            "case": case.name,
            "reference_kind": "dmrg",
            "reference_label": "DMRG",
            "sweep_kind": "store_abs_tol",
            "threshold_name": "store_abs_tol",
            "threshold_value": 1.0e-5,
            "method": "dense",
            "config_tag": "basis_20",
            "store_abs_tol": 1.0e-5,
            "propagate_abs_tol": 1.0e-8,
            "e_total_eh": None,
            "delta_vs_ref_mEh": None,
            "abs_delta_vs_ref_mEh": None,
            "wall_s": 10.0,
            "basis_size": 20,
            "max_cycle": 1,
            "restart_m_cycle": 0,
            "restart_nroot": 2,
            "failure": "QP CCIK run timed out.",
        }
    }
    ninferred = MODULE._infer_smaller_threshold_failures(
        failed_job=failed_job,
        failure="QP CCIK run timed out.",
        job_specs=matrix,
        rows_by_key=rows_by_key,
        case=case,
        reference={"reference_kind": "dmrg", "reference_label": "DMRG"},
    )
    assert ninferred > 0
    assert ("store_abs_tol", 1.0e-6, "dense") in rows_by_key
    assert rows_by_key[("store_abs_tol", 1.0e-6, "dense")]["failure"].startswith("Inferred monotone failure")
