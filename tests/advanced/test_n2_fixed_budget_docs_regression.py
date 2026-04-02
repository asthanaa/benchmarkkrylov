from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "advanced" / "n2_fixed_budget_docs_regression.py"
SPEC = importlib.util.spec_from_file_location("n2_fixed_budget_docs_regression", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_golden_cas12_row_exists_in_retained_docs_assets() -> None:
    row = MODULE.load_golden_docs_row()
    assert row["system"] == "n2_cas12_12_R2p50"
    assert row["phase"] == "kprop"
    assert row["budget_value"] == 2048
    assert row["fixed_store_keep"] == 65536
    assert row["fixed_propagate_keep"] == 2048
    assert row["source_kind"] == "fresh_qp_donor_seeded"


def test_local_cas12_seed_paths_exist() -> None:
    case = MODULE.resolve_publication_case()
    seed_ezfio, seed_fcidump = MODULE._require_seed_paths(case)
    assert seed_ezfio.exists()
    assert seed_fcidump.exists()


def test_fixed_budget_qp_param_factory_matches_golden_row_high_level_knobs() -> None:
    case = MODULE.resolve_publication_case()
    row = MODULE.load_golden_docs_row()
    params = MODULE._make_qp_params(case, row)
    assert params.ccik_compression_policy == "fixed_budget"
    assert params.ccik_selector_mode == "abs_residual"
    assert params.ccik_fixed_store_keep == 65536
    assert params.ccik_fixed_propagate_keep == 2048
    assert params.ccik_restart_m_cycle == 30
    assert params.ccik_restart_nroot == 8
    assert params.ccik_generator_keep == 2048
    assert params.ccik_max_parents_for_expansion == 2048
    assert params.ccik_selector_keep == 65536
    assert params.ccik_nkeep == 65536
    assert params.ccik_tmp_keep_factor == 4
    assert params.ccik_preserve_prev_support is True
    assert params.ccik_preserve_prev_support_cap == 2048


def test_notebook_bridge_policy_matches_docs_notebook_heuristic() -> None:
    case = MODULE.resolve_publication_case()
    row = MODULE.load_golden_docs_row(case_name="n2_cas12_12_R2p50", phase="kprop", budget_value=512, method="ccik_thick_qp")
    params = MODULE._make_qp_params_with_policy(case, row, bridge_policy="notebook", qp_mode="thick")
    assert params.ccik_generator_keep == 2048
    assert params.ccik_max_parents_for_expansion == 0
    assert params.ccik_selector_keep == 1024
    assert params.ccik_external_keep == 20_000
    assert params.ccik_nkeep == 65_536
    assert params.ccik_preserve_prev_support is True
    assert params.ccik_preserve_prev_support_cap == 2048
    assert params.ccik_restart_m_cycle == 30
    assert params.ccik_restart_nroot == 8


def test_docs_legacy_bridge_policy_matches_d4b2255_mapping() -> None:
    case = MODULE.resolve_publication_case()
    row = MODULE.load_golden_docs_row()
    params = MODULE._make_qp_params_with_policy(case, row, bridge_policy="docs_legacy", qp_mode="thick")
    assert params.ccik_generator_keep == 1
    assert params.ccik_max_parents_for_expansion == 0
    assert params.ccik_selector_keep == 1
    assert params.ccik_external_keep == int(case.external_keep)
    assert params.ccik_external_candidate_factor == 1
    assert params.ccik_nkeep == 1
    assert params.ccik_tmp_keep_factor == 1
    assert params.ccik_preserve_prev_support is False
    assert params.ccik_preserve_prev_support_cap == 0
    assert params.ccik_restart_m_cycle == 30
    assert params.ccik_restart_nroot == 8


def test_qp_baseline_mode_forces_nonrestart_path() -> None:
    case = MODULE.resolve_publication_case()
    row = MODULE.load_golden_docs_row(case_name="n2_cas12_12_R2p50", phase="kprop", budget_value=512, method="ccik_thick_qp")
    params = MODULE._make_qp_params_with_policy(case, row, bridge_policy="current", qp_mode="baseline")
    assert params.ccik_restart_m_cycle == 1
    assert params.ccik_restart_nroot == 0
    assert params.ccik_max_cycle == 30


def test_iteration_error_reporting_writes_reference_energy(tmp_path: Path) -> None:
    ezfio = tmp_path / "n2.ezfio"
    row = MODULE.load_golden_docs_row()
    MODULE._enable_iteration_error_reporting(ezfio, row)
    assert float((ezfio / "ccik" / "ccik_reference_total_energy").read_text(encoding="ascii").strip()) == pytest.approx(
        float(row["reference_total_energy_eh"]),
        abs=1.0e-12,
    )
    assert (ezfio / "ccik" / "ccik_print_iteration_error").read_text(encoding="ascii").strip() == "T"


def test_runtime_and_plugin_fingerprints_include_git_anchors() -> None:
    payload = MODULE._runtime_and_plugin_fingerprints(ROOT / "nonexistent_qp_root")
    assert payload["comparison_mode"] == "bridge-only historical replay"
    assert payload["repo_git_head"]
    commits = {row["commit"] for row in payload["git_anchors"]}
    assert {"d4b2255", "ae53767", "ef3921e", "5a76802"} <= commits


def test_classify_drift_identifies_setup_drift() -> None:
    results = {
        "donor_seeded_qp": {"docs_comparison": {"matches_docs": True, "delta_vs_docs_mEh": 0.0}},
        "fresh_publication_setup_qp": {"docs_comparison": {"matches_docs": False, "delta_vs_docs_mEh": 1.2}},
    }
    summary = MODULE.classify_drift(results)
    assert summary["classification"] == "setup_drift_between_donor_seeded_and_fresh_publication_setup"


def test_run_qp_ccik_uses_qp_set_file_with_runtime_local_ezfio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    ezfio = runtime_parent / "n2.ezfio"
    ezfio.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    rc = tmp_path / "quantum_package.rc"
    ccik = tmp_path / "ccik"
    rc.write_text("", encoding="utf-8")
    ccik.write_text("", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "CCIK_FINAL_TOTAL_ENERGY -1.0\n"
                "CCIK_FINAL_ELECTRONIC_ENERGY -2.0\n"
                "CCIK_FINAL_BASIS 30\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(
        MODULE,
        "discover_qp_paths_from_root",
        lambda _root: SimpleNamespace(quantum_package_rc=rc, ccik_binary=ccik),
    )
    monkeypatch.setattr(MODULE, "qp_parallel_env", lambda _threads: {})
    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        MODULE.pub,
        "_extract_optional",
        lambda _stdout, key: {"CCIK_FINAL_TOTAL_ENERGY": -1.0, "CCIK_FINAL_ELECTRONIC_ENERGY": -2.0, "CCIK_FINAL_BASIS": 30}.get(key),
    )

    payload = MODULE._run_qp_ccik(
        qp_root=tmp_path,
        runtime_parent=runtime_parent,
        ezfio=ezfio,
        log_dir=log_dir,
        omp_threads=1,
    )

    assert payload["failure"] is None
    assert payload["e_total_eh"] == -1.0
    assert captured["cwd"] == str(runtime_parent.resolve())
    assert "qp set_file n2.ezfio" in captured["cmd"][2]
    assert captured["cmd"][2].endswith("ccik n2.ezfio")


def test_small_shared_parity_python_oracle_uses_connected_sd_and_fixed_budget_bridge() -> None:
    params = MODULE._make_small_parity_python_params(steps=8, fixed_store_keep=256, fixed_propagate_keep=64)
    assert params.compression_policy == "fixed_budget"
    assert params.candidate_mode == "connected_sd"
    assert params.generator_keep == 64
    assert params.max_parents_for_expansion == 64
    assert params.include_current_support is True
    assert params.tmp_keep_factor == 4
    assert params.preserve_prev_support is True
    assert params.preserve_prev_support_cap == 64


def test_small_shared_parity_qp_oracle_runs_single_threaded_baseline_bridge() -> None:
    params = MODULE._make_small_parity_qp_params(steps=8, fixed_store_keep=256, fixed_propagate_keep=64)
    assert params.ccik_compression_policy == "fixed_budget"
    assert params.ccik_generator_keep == 64
    assert params.ccik_max_parents_for_expansion == 64
    assert params.ccik_selector_keep == 256
    assert params.ccik_nkeep == 256
    assert params.ccik_tmp_keep_factor == 4
    assert params.ccik_preserve_prev_support is True
    assert params.ccik_preserve_prev_support_cap == 64
    assert params.ccik_restart_m_cycle == 1
    assert params.ccik_restart_nroot == 0
    assert params.ccik_max_cycle == 8


def test_small_shared_parity_python_thick_oracle_uses_connected_sd_restart_shape() -> None:
    params = MODULE._make_small_parity_python_thick_params(
        m_cycle=4,
        ncycles=3,
        nroot=2,
        fixed_store_keep=256,
        fixed_propagate_keep=64,
    )
    assert params.compression_policy == "fixed_budget"
    assert params.candidate_mode == "connected_sd"
    assert params.m_cycle == 4
    assert params.ncycles == 3
    assert params.nroot == 2
    assert params.generator_keep == 64
    assert params.max_parents_for_expansion == 64
    assert params.tmp_keep_factor == 4
    assert params.preserve_prev_support is True
    assert params.preserve_prev_support_cap == 64


def test_small_shared_parity_qp_thick_oracle_uses_restart_shape() -> None:
    params = MODULE._make_small_parity_qp_params(
        steps=8,
        fixed_store_keep=256,
        fixed_propagate_keep=64,
        qp_mode="thick",
        m_cycle=4,
        ncycles=3,
        nroot=2,
    )
    assert params.ccik_compression_policy == "fixed_budget"
    assert params.ccik_generator_keep == 64
    assert params.ccik_max_parents_for_expansion == 64
    assert params.ccik_selector_keep == 256
    assert params.ccik_nkeep == 256
    assert params.ccik_restart_m_cycle == 4
    assert params.ccik_restart_nroot == 2
    assert params.ccik_max_cycle == 3
    assert params.ccik_basis_size == 4


def test_localize_first_mismatch_ignores_projected_noise_when_qnext_matches() -> None:
    row = {
        "step": 2,
        "propagation_support": {"exact_support_match": True, "max_abs_coef_diff": 0.0},
        "parent_support": {"exact_support_match": True, "max_abs_coef_diff": 0.0},
        "projected_hq": {"exact_support_match": False, "max_abs_coef_diff": 2.0e-12},
        "orthogonalized_residual": {"exact_support_match": False, "max_abs_coef_diff": 3.0e-12},
        "q_next": {"exact_support_match": True, "max_abs_coef_diff": 4.0e-13},
        "lowest_ritz_energy": {"matches": True},
    }
    out = MODULE._localize_first_mismatch([row])
    assert out["bug_class"] == "no_mismatch_detected"


def test_qp_thick_driver_checks_convergence_only_at_cycle_boundary() -> None:
    driver = (ROOT / "ccik_qp" / "stable" / "ccik" / "ccik_driver.irp.f").read_text(encoding="utf-8")
    start = driver.index("subroutine ccik_thick_restart_driver")
    end = driver.index("subroutine ccik_baseline_driver")
    thick = driver[start:end]
    assert "prev_cycle_energy" in thick
    assert thick.count("< ccik_conv_tol") == 1
    assert "converged = abs(energy - prev_cycle_energy) < ccik_conv_tol" in thick
    assert "converged = abs(energy - prev_energy) < ccik_conv_tol" not in thick


def test_qp_sparse_compress_preserves_canonical_order_inside_equal_abs_groups() -> None:
    sparse = (ROOT / "ccik_qp" / "stable" / "ccik" / "ccik_sparse.f90").read_text(encoding="utf-8")
    assert "subroutine ccik_stabilize_equal_abs_groups" in sparse
    assert "call ccik_stabilize_equal_abs_groups(abscoef, idx, vec%n)" in sparse
    assert "call ccik_stabilize_equal_abs_groups(absref, idx_ref, ref%n)" in sparse


@pytest.mark.skipif(
    os.environ.get("CKA_RUN_N2_CAS12_DOCS_REGRESSION") != "1",
    reason="set CKA_RUN_N2_CAS12_DOCS_REGRESSION=1 to run the live donor-seeded CAS(12,12) docs regression",
)
def test_live_donor_seeded_qp_matches_retained_docs_row(tmp_path: Path) -> None:
    payload = MODULE.run_regression(
        outdir=tmp_path,
        selected_paths=("donor_seeded_qp",),
        force=True,
    )
    result = payload["results"]["donor_seeded_qp"]
    assert result["run"]["failure"] is None
    assert result["docs_comparison"]["matches_docs"] is True
