from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccik import build_cas_hamiltonian_pyscf, ccik_ground_energy_dense_thick_restart, exact_cas_fci_energy_pyscf, make_mol_pyscf
from ccik.params import CCIKThickRestartParams
from ccik.pyscf_cas import CASSpec

MODULE_PATH = ROOT / "scripts" / "advanced" / "n2_thick_iteration_study.py"
SPEC = importlib.util.spec_from_file_location("n2_thick_iteration_study", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_limit_iteration_rows_respects_cap() -> None:
    rows = [
        {"cum_new_krylov": 10},
        {"cum_new_krylov": 100},
        {"cum_new_krylov": 301},
    ]
    limited = MODULE.limit_iteration_rows(rows, 300)
    assert [row["cum_new_krylov"] for row in limited] == [10, 100]


def test_discover_input_files_recurses_into_nested_dirs(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    (root / "top_case").mkdir(parents=True)
    (root / "nested" / "case_b").mkdir(parents=True)
    (root / "top_case" / "input.json").write_text("{}", encoding="utf-8")
    (root / "nested" / "case_b" / "input.json").write_text("{}", encoding="utf-8")

    discovered = MODULE.discover_input_files(root)

    assert discovered == [
        root / "nested" / "case_b" / "input.json",
        root / "top_case" / "input.json",
    ]


def test_effective_reference_mode_uses_dmrg_for_large_cases() -> None:
    small = {"cas": {"ncas": 10, "nelecas": 10}, "study": {"reference_mode": "auto"}}
    large = {"cas": {"ncas": 14, "nelecas": 16}, "study": {"reference_mode": "auto"}}

    assert MODULE._effective_reference_mode(small) == "exact_fci"
    assert MODULE._effective_reference_mode(large) == "dmrg_then_cipsi"


def test_resolve_qp_reference_falls_back_from_dmrg_to_cipsi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dmrg_calls = []
    cipsi_calls = []

    def _fake_dmrg(**kwargs):
        dmrg_calls.append(kwargs)
        return {
            "success": False,
            "reference_kind": "dmrg",
            "reference_label": "DMRG",
            "failure": "dmrg failed",
            "stop_reason": "subprocess_failed",
        }

    def _fake_cipsi(**kwargs):
        cipsi_calls.append(kwargs)
        return {
            "success": True,
            "reference_kind": "cipsi",
            "reference_label": "CIPSI variational",
            "total_energy_eh": -1.234,
        }

    monkeypatch.setattr(MODULE, "load_or_compute_dmrg_reference_from_fcidump", _fake_dmrg)
    monkeypatch.setattr(MODULE, "load_or_compute_pyscf_selected_ci_reference_from_fcidump", _fake_cipsi)

    setup_dir = tmp_path / "system_setup"
    setup_dir.mkdir()
    ezfio = setup_dir / "n2.ezfio"
    ezfio.mkdir()
    fcidump = setup_dir / "qp_active.FCIDUMP"
    fcidump.write_text("", encoding="utf-8")

    spec = {
        "name": "n2_large",
        "cas": {"ncas": 14, "nelecas": 14},
        "study": {"reference_mode": "dmrg_then_cipsi"},
    }

    chosen, choice = MODULE._resolve_qp_reference(
        spec=spec,
        qp_root=tmp_path / "qp_root",
        ezfio_path=ezfio,
        fcidump_path=fcidump,
        setup_dir=setup_dir,
        h1=np.zeros((1, 1)),
        eri8=np.zeros((1, 1, 1, 1)),
        norb=1,
        nelec=(1, 0),
        ecore=0.0,
        omp_threads=8,
        force=True,
    )

    assert chosen["reference_kind"] == "cipsi"
    assert float(chosen["total_energy_eh"]) == pytest.approx(-1.234)
    assert len(dmrg_calls) == 1
    assert len(cipsi_calls) == 1
    assert choice["effective_reference_mode"] == "dmrg_then_cipsi"
    assert (setup_dir / "reference_choice.json").exists()


def test_plot_run_trace_writes_files(tmp_path: Path) -> None:
    rows = [
        {
            "cycle": 1,
            "basis_size": 2,
            "restart_seed_count": 1,
            "cum_new_krylov": 1,
            "electronic_energy_eh": -1.0,
            "total_energy_eh": -1.0,
            "delta_mHa": -10.0,
            "abs_delta_mHa": 10.0,
        },
        {
            "cycle": 1,
            "basis_size": 3,
            "restart_seed_count": 1,
            "cum_new_krylov": 2,
            "electronic_energy_eh": -1.1,
            "total_energy_eh": -1.1,
            "delta_mHa": -1.0,
            "abs_delta_mHa": 1.0,
        },
    ]
    run_dir = tmp_path / "exact_oracle"
    MODULE.write_trace_outputs(run_dir, rows)
    png_path, pdf_path = MODULE.plot_run_trace(
        run_dir=run_dir,
        rows=rows,
        title="synthetic",
        max_cumulative_iterations=300,
    )
    assert (run_dir / "iteration_trace.csv").exists()
    assert (run_dir / "iteration_trace.json").exists()
    assert png_path.exists()
    assert pdf_path.exists()


def test_exact_on_basis_thick_trace_stays_variational_on_h2() -> None:
    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=CASSpec(ncas=2, nelecas=2, ncore=0))
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, norb, nelec))

    params = CCIKThickRestartParams(
        m_cycle=5,
        ncycles=2,
        nroot=2,
        compression_policy="fixed_budget",
        fixed_store_keep=8,
        fixed_propagate_keep=4,
        candidate_mode="full",
        projected_ritz_mode="exact_on_basis",
        selector_mode="abs_residual",
        verbose=False,
    )
    seen = []

    def _collect(record) -> None:
        seen.append(float(ecore + record.electronic_energy_eh))

    energy = float(
        ecore
        + ccik_ground_energy_dense_thick_restart(
            h1,
            eri8,
            norb,
            nelec,
            params=params,
            iteration_callback=_collect,
        )
    )

    assert seen
    assert all(val >= e_fci_total - 1.0e-10 for val in seen)
    assert energy >= e_fci_total - 1.0e-10


def test_exact_on_basis_selected_ci_thick_stays_variational_on_small_n2() -> None:
    mol = make_mol_pyscf(
        atom="N 0 0 0; N 0 0 1.1",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=CASSpec(ncas=4, nelecas=4, ncore=0))
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, norb, nelec))

    params = CCIKThickRestartParams(
        m_cycle=6,
        ncycles=2,
        nroot=2,
        compression_policy="fixed_budget",
        fixed_store_keep=32,
        fixed_propagate_keep=8,
        candidate_mode="selected_ci",
        projected_ritz_mode="exact_on_basis",
        selector_mode="abs_residual",
        verbose=False,
    )
    seen = []

    def _collect(record) -> None:
        seen.append(float(ecore + record.electronic_energy_eh))

    energy = float(
        ecore
        + ccik_ground_energy_dense_thick_restart(
            h1,
            eri8,
            norb,
            nelec,
            params=params,
            iteration_callback=_collect,
        )
    )

    assert seen
    assert all(val >= e_fci_total - 1.0e-10 for val in seen)
    assert energy >= e_fci_total - 1.0e-10


def test_run_case_writes_two_run_layout(tmp_path: Path) -> None:
    case_dir = tmp_path / "n2_case"
    case_dir.mkdir()
    input_path = case_dir / "input.json"
    payload = {
        "name": "n2_test_case",
        "label": "N2 test case",
        "molecule": {
            "distance_ang": 1.1,
            "basis": "sto-3g",
            "unit": "Angstrom",
            "charge": 0,
            "spin": 0,
            "verbose": 0,
        },
        "cas": {
            "ncas": 4,
            "nelecas": 4,
            "ncore": 0,
        },
        "study": {
            "max_cumulative_iterations": 20,
            "stop_abs_delta_mHa": 1.0e6,
            "error_unit": "mHa",
            "plot_yscale": "log",
        },
        "runs": [
            {
                "name": "exact_oracle",
                "label": "Exact oracle",
                "solver": {
                    "backend": "python",
                    "kind": "ccik_thick",
                    "params": {
                        "m_cycle": 6,
                        "ncycles": 2,
                        "nroot": 2,
                        "compression_policy": "fixed_budget",
                        "fixed_store_keep": 32,
                        "fixed_propagate_keep": 8,
                        "candidate_mode": "full",
                        "projected_ritz_mode": "exact_on_basis",
                        "selector_mode": "abs_residual",
                    },
                },
            },
            {
                "name": "default_selected_ci",
                "label": "Default selected-CI",
                "solver": {
                    "backend": "python",
                    "kind": "ccik_thick",
                    "params": {
                        "m_cycle": 6,
                        "ncycles": 2,
                        "nroot": 2,
                        "compression_policy": "fixed_budget",
                        "fixed_store_keep": 32,
                        "fixed_propagate_keep": 8,
                        "candidate_mode": "selected_ci",
                        "projected_ritz_mode": "exact_on_basis",
                        "selector_mode": "abs_residual",
                    },
                },
            },
        ],
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = MODULE.run_case(input_path, force=True)

    assert (case_dir / "exact_oracle" / "iteration_trace.json").exists()
    assert (case_dir / "exact_oracle" / "summary.json").exists()
    assert (case_dir / "exact_oracle" / "baseline_no_restart_audit.json").exists()
    assert (case_dir / "default_selected_ci" / "iteration_trace.json").exists()
    assert (case_dir / "default_selected_ci" / "summary.json").exists()
    assert (case_dir / "energy_error_vs_iteration_comparison.png").exists()
    assert summary["runs"]["exact_oracle"] is not None
    assert summary["runs"]["default_selected_ci"] is not None
    assert summary["stop_abs_delta_mHa"] == pytest.approx(1.0e6)
    assert summary["runs"]["exact_oracle"]["early_stop_reached"] is True
    assert summary["runs"]["default_selected_ci"]["early_stop_reached"] is True
