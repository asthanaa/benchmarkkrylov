from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf.fci import cistring, direct_spin1

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccik import build_cas_hamiltonian_pyscf, ccik_ground_energy_dense_thick_restart, exact_cas_fci_energy_pyscf, make_mol_pyscf
from ccik.params import CCIKThickRestartParams
from ccik.pyscf_cas import CASSpec
from ccik.selection import _enumerate_connected_sd_support
from ccik.sparse_ci import SparseCIVector, sparse_from_scivector
from ccik.thick_restart import _exactize_support_union, _exactizeplus_support_union

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


def test_qp_baseline_line_regex_parses_seed_iterations() -> None:
    line = "ccik(baseline) iter=  31  total=       -108.800000000000  ref=       -108.861077882421  delta_mEh=       61.077882"

    match = MODULE.QP_BASELINE_LINE_RE.search(line)

    assert match is not None
    assert int(match.group("iter")) == 31
    assert float(match.group("total")) == pytest.approx(-108.8)
    assert float(match.group("delta")) == pytest.approx(61.077882)


def test_make_qp_thick_params_turns_on_existing_wf_for_restart_runs() -> None:
    params = MODULE._make_qp_thick_params(
        run_params={
            "m_cycle": 7,
            "ncycles": 20,
            "nroot": 3,
            "fixed_store_keep": 200000,
            "fixed_propagate_keep": 1000,
            "restart_from_run": "seed_m031",
        },
        reference_total_eh=-108.0,
    )

    assert params.ccik_basis_size == 7
    assert params.ccik_restart_m_cycle == 7
    assert params.ccik_restart_nroot == 3
    assert params.ccik_use_existing_wf is True


def test_make_qp_thick_params_threads_exactized_knobs() -> None:
    params = MODULE._make_qp_thick_params(
        run_params={
            "m_cycle": 10,
            "ncycles": 11,
            "nroot": 3,
            "fixed_store_keep": 1000,
            "fixed_propagate_keep": 1000,
            "exactized_checkpoint_every": 1,
            "exactized_resid_tol": 1.0e-10,
            "exactized_max_iter": 200,
            "exactized_max_subspace": 24,
            "exactized_restart_keep": 4,
        },
        reference_total_eh=-108.0,
    )

    assert params.ccik_exactized_checkpoint_every == 1
    assert params.ccik_exactized_resid_tol == pytest.approx(1.0e-10)
    assert params.ccik_exactized_max_iter == 200
    assert params.ccik_exactized_max_subspace == 24
    assert params.ccik_exactized_restart_keep == 4


def test_make_qp_thick_params_threads_secondortho_knob() -> None:
    params = MODULE._make_qp_thick_params(
        run_params={
            "m_cycle": 31,
            "ncycles": 10,
            "nroot": 3,
            "fixed_store_keep": 10000,
            "fixed_propagate_keep": 500,
            "secondortho": True,
        },
        reference_total_eh=-108.0,
    )

    assert params.ccik_secondortho is True


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


def test_qp_exactized_trace_line_updates_latest_row() -> None:
    row = MODULE._build_qp_trace_row(
        cycle=2,
        iter_idx=5,
        total_energy_eh=-108.75,
        delta_mha=10.0,
        ecore=-5.0,
        restart_m_cycle=10,
        restart_nroot=3,
        elapsed_s=12.5,
    )
    line = (
        "CCIK_TRACE_EXACTIZED cycle=2 iter=5 k=12 support=1356 "
        "electronic=-103.9000000000000000 total=-108.9000000000000000 ref=-108.8610000000000000 "
        "delta_mEh=-39.0000000000000000 solver=davidson iters=14 matvecs=15 converged=T "
        "resid=1.0000000000000000E-08 t_union_s=0.1234000000000000E+00 "
        "t_prep_s=0.2345000000000000E+00 t_solve_s=0.3456000000000000E+00 "
        "t_wall_s=0.7035000000000000E+00 t_cpu_s=0.7021000000000000E+00"
    )

    match = MODULE.QP_EXACTIZED_TRACE_RE.search(line)

    assert match is not None
    MODULE._update_qp_row_with_exactized_trace(row, match)
    assert row["support_union_size"] == 1356
    assert row["exactized_total_energy_eh"] == pytest.approx(-108.9)
    assert row["exactized_delta_mHa"] == pytest.approx(-39.0)
    assert row["exactized_abs_delta_mHa"] == pytest.approx(39.0)
    assert row["exactized_solver_kind"] == "davidson"
    assert row["exactized_davidson_iterations"] == 14
    assert row["exactized_matvec_count"] == 15
    assert row["exactized_converged"] is True
    assert row["exactized_support_build_s"] == pytest.approx(0.1234)
    assert row["exactized_prep_s"] == pytest.approx(0.2345)
    assert row["exactized_solve_s"] == pytest.approx(0.3456)


def test_exactized_checkpoint_matches_dense_restricted_diagonalization() -> None:
    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, _ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=CASSpec(ncas=2, nelecas=2, ncore=0))
    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    strsa = [int(cistring.addr2str(norb, nelec[0], idx)) for idx in range(cistring.num_strings(norb, nelec[0]))]
    strsb = [int(cistring.addr2str(norb, nelec[1], idx)) for idx in range(cistring.num_strings(norb, nelec[1]))]
    support_keys = [(strsa[0], strsb[0]), (strsa[0], strsb[1]), (strsa[1], strsb[0])]

    exactized = _exactize_support_union(
        support_keys=support_keys,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        initial_guess=None,
    )

    manual_H = np.zeros((len(support_keys), len(support_keys)), dtype=float)
    for col, (stra_col, strb_col) in enumerate(support_keys):
        dense = np.zeros((len(strsa), len(strsb)), dtype=float)
        ia = int(cistring.str2addr(norb, nelec[0], int(stra_col)))
        ib = int(cistring.str2addr(norb, nelec[1], int(strb_col)))
        dense[ia, ib] = 1.0
        h_dense = np.asarray(direct_spin1.contract_2e(h2eff, dense, norb, nelec), dtype=float)
        for row, (stra_row, strb_row) in enumerate(support_keys):
            ia_row = int(cistring.str2addr(norb, nelec[0], int(stra_row)))
            ib_row = int(cistring.str2addr(norb, nelec[1], int(strb_row)))
            manual_H[row, col] = float(h_dense[ia_row, ib_row])
    manual_H = 0.5 * (manual_H + manual_H.T)
    ref_energy = float(np.linalg.eigvalsh(manual_H)[0])

    assert exactized["solver_kind"] == "dense_eigh"
    assert int(exactized["subspace_dim"]) == len(support_keys)
    assert float(exactized["electronic_energy_eh"]) == pytest.approx(ref_energy, abs=1.0e-10)


def test_exactizeplus_matches_manual_one_shell_expansion() -> None:
    from pyscf.fci import selected_ci

    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, _ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=CASSpec(ncas=2, nelecas=2, ncore=0))
    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    strsa = [int(cistring.addr2str(norb, nelec[0], idx)) for idx in range(cistring.num_strings(norb, nelec[0]))]
    strsb = [int(cistring.addr2str(norb, nelec[1], idx)) for idx in range(cistring.num_strings(norb, nelec[1]))]
    support_keys = [(strsa[0], strsb[0]), (strsa[0], strsb[1]), (strsa[1], strsb[0])]

    exactized = _exactize_support_union(
        support_keys=support_keys,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        initial_guess=None,
    )
    coeffs = np.asarray(exactized["lowest_eigenvector"], dtype=float)
    ranked = sorted(
        zip(support_keys, coeffs, strict=False),
        key=lambda item: (abs(float(item[1])), int(item[0][0]), int(item[0][1])),
        reverse=True,
    )
    parent_keys = [key for key, _coeff in ranked]

    shell_support = set(support_keys)
    for parent in parent_keys:
        connected = _enumerate_connected_sd_support({parent}, norb)
        sci_parent = SparseCIVector({parent: 1.0}).to_scivector_on_support(connected)
        h_parent = selected_ci.contract_2e(h2eff, sci_parent, norb, nelec)
        shell_support.update((int(stra), int(strb)) for stra, strb in sparse_from_scivector(h_parent).coeffs)
        shell_support.add(parent)

    manual = _exactize_support_union(
        support_keys=sorted(shell_support, key=lambda key: (int(key[0]), int(key[1]))),
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        initial_guess=None,
    )
    exactizeplus = _exactizeplus_support_union(
        support_keys=support_keys,
        exactized_coeffs=coeffs,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        top_parents=500,
    )

    assert int(exactizeplus["parent_count"]) == len(parent_keys)
    assert int(exactizeplus["subspace_dim"]) == int(manual["subspace_dim"])
    assert int(exactizeplus["shell_added"]) == len(shell_support) - len(set(support_keys))
    assert float(exactizeplus["electronic_energy_eh"]) == pytest.approx(float(manual["electronic_energy_eh"]), abs=1.0e-10)


def test_exactized_checkpoints_preserve_standard_trace() -> None:
    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=CASSpec(ncas=2, nelecas=2, ncore=0))

    base_params = {
        "m_cycle": 4,
        "ncycles": 1,
        "nroot": 1,
        "compression_policy": "fixed_budget",
        "fixed_store_keep": 8,
        "fixed_propagate_keep": 8,
        "candidate_mode": "full",
        "projected_ritz_mode": "exact_on_basis",
        "selector_mode": "abs_residual",
        "verbose": False,
    }
    plain_records = []
    exactized_records = []

    def _collect_plain(record) -> None:
        plain_records.append(record)

    def _collect_exactized(record) -> None:
        exactized_records.append(record)

    plain_energy = float(
        ecore
        + ccik_ground_energy_dense_thick_restart(
            h1,
            eri8,
            norb,
            nelec,
            params=CCIKThickRestartParams(**base_params),
            iteration_callback=_collect_plain,
        )
    )
    exactized_energy = float(
        ecore
        + ccik_ground_energy_dense_thick_restart(
            h1,
            eri8,
            norb,
            nelec,
            params=CCIKThickRestartParams(**{**base_params, "exactized_checkpoint_every": 1}),
            iteration_callback=_collect_exactized,
        )
    )

    assert plain_energy == pytest.approx(exactized_energy, abs=1.0e-12)
    assert [record.cum_new_krylov for record in plain_records] == [record.cum_new_krylov for record in exactized_records]
    assert [record.electronic_energy_eh for record in plain_records] == pytest.approx(
        [record.electronic_energy_eh for record in exactized_records],
        abs=1.0e-12,
    )
    assert all(record.support_union_size is not None for record in exactized_records)
    assert any(record.exactized_electronic_energy_eh is not None for record in exactized_records)


def test_exactizeplus_checkpoints_preserve_standard_trace() -> None:
    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=CASSpec(ncas=2, nelecas=2, ncore=0))

    base_params = {
        "m_cycle": 4,
        "ncycles": 1,
        "nroot": 1,
        "compression_policy": "fixed_budget",
        "fixed_store_keep": 8,
        "fixed_propagate_keep": 8,
        "candidate_mode": "full",
        "projected_ritz_mode": "exact_on_basis",
        "selector_mode": "abs_residual",
        "verbose": False,
    }
    plain_records = []
    exactizeplus_records = []

    def _collect_plain(record) -> None:
        plain_records.append(record)

    def _collect_exactizeplus(record) -> None:
        exactizeplus_records.append(record)

    plain_energy = float(
        ecore
        + ccik_ground_energy_dense_thick_restart(
            h1,
            eri8,
            norb,
            nelec,
            params=CCIKThickRestartParams(**base_params),
            iteration_callback=_collect_plain,
        )
    )
    exactizeplus_energy = float(
        ecore
        + ccik_ground_energy_dense_thick_restart(
            h1,
            eri8,
            norb,
            nelec,
            params=CCIKThickRestartParams(
                **{**base_params, "exactized_checkpoint_every": 1, "exactizeplus_enabled": True}
            ),
            iteration_callback=_collect_exactizeplus,
        )
    )

    assert plain_energy == pytest.approx(exactizeplus_energy, abs=1.0e-12)
    assert [record.cum_new_krylov for record in plain_records] == [record.cum_new_krylov for record in exactizeplus_records]
    assert [record.electronic_energy_eh for record in plain_records] == pytest.approx(
        [record.electronic_energy_eh for record in exactizeplus_records],
        abs=1.0e-12,
    )
    assert any(record.exactized_electronic_energy_eh is not None for record in exactizeplus_records)
    assert any(record.exactizeplus_electronic_energy_eh is not None for record in exactizeplus_records)


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


def test_run_case_writes_exactized_checkpoint_fields(tmp_path: Path) -> None:
    case_dir = tmp_path / "n2_exactized_case"
    case_dir.mkdir()
    input_path = case_dir / "input.json"
    payload = {
        "name": "n2_exactized_case",
        "label": "N2 exactized test case",
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
            "max_cumulative_iterations": 12,
            "stop_abs_delta_mHa": 1.0e6,
            "error_unit": "mHa",
            "plot_yscale": "log",
        },
        "runs": [
            {
                "name": "exactized_demo",
                "label": "Exactized demo",
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
                        "exactized_checkpoint_every": 1,
                    },
                },
            }
        ],
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = MODULE.run_case(input_path, force=True)
    trace_rows = json.loads((case_dir / "exactized_demo" / "iteration_trace.json").read_text(encoding="utf-8"))

    assert trace_rows
    assert any(row.get("exactized_abs_delta_mHa") is not None for row in trace_rows)
    assert (case_dir / "exactized_demo" / "energy_error_vs_iteration.png").exists()
    assert summary["runs"]["exactized_demo"]["exactized_trace_points"] >= 1


def test_run_case_writes_exactizeplus_checkpoint_fields(tmp_path: Path) -> None:
    case_dir = tmp_path / "n2_exactizeplus_case"
    case_dir.mkdir()
    input_path = case_dir / "input.json"
    payload = {
        "name": "n2_exactizeplus_case",
        "label": "N2 exactizeplus test case",
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
            "max_cumulative_iterations": 12,
            "stop_abs_delta_mHa": 1.0e6,
            "error_unit": "mHa",
            "plot_yscale": "log",
        },
        "runs": [
            {
                "name": "exactizeplus_demo",
                "label": "Exactizeplus demo",
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
                        "exactized_checkpoint_every": 1,
                        "exactizeplus_enabled": True,
                        "exactizeplus_top_parents": 500,
                    },
                },
            }
        ],
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = MODULE.run_case(input_path, force=True)
    trace_rows = json.loads((case_dir / "exactizeplus_demo" / "iteration_trace.json").read_text(encoding="utf-8"))

    assert trace_rows
    assert any(row.get("exactized_abs_delta_mHa") is not None for row in trace_rows)
    assert any(row.get("exactizeplus_abs_delta_mHa") is not None for row in trace_rows)
    assert (case_dir / "exactizeplus_demo" / "energy_error_vs_iteration.png").exists()
    assert summary["runs"]["exactizeplus_demo"]["exactized_trace_points"] >= 1
    assert summary["runs"]["exactizeplus_demo"]["exactizeplus_trace_points"] >= 1


def test_prepare_case_qp_cached_setup_uses_current_omp_threads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case_dir = tmp_path / "n2_case"
    setup_dir = case_dir / "system_setup"
    setup_dir.mkdir(parents=True)
    ezfio = setup_dir / "n2.ezfio"
    ezfio.mkdir()
    fcidump = setup_dir / "qp_active.FCIDUMP"
    fcidump.write_text("", encoding="utf-8")
    (setup_dir / "system_setup.json").write_text(
        json.dumps(
            {
                "qp_root": str(tmp_path / "qp_root_cached"),
                "ezfio_path": str(ezfio),
                "fcidump_path": str(fcidump),
                "ecore": 0.5,
                "norb": 6,
                "nelec": [3, 3],
                "reference_total_energy_eh": -1.25,
                "reference_summary": {"reference_kind": "dmrg", "reference_label": "DMRG"},
                "effective_reference_mode": "dmrg",
                "omp_threads": 8,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(MODULE, "_select_qp_root", lambda _value: tmp_path / "qp_root_current")
    monkeypatch.setattr(MODULE, "_ensure_qp_study_binaries", lambda _path: None)
    monkeypatch.setattr(MODULE, "discover_qp_paths_from_root", lambda _path: object())

    spec = {
        "label": "N2 cached setup",
        "molecule": {"distance_ang": 2.0, "basis": "6-31g", "unit": "Angstrom", "charge": 0, "spin": 0, "verbose": 0},
        "cas": {"ncas": 4, "nelecas": 4, "ncore": 0},
        "study": {"omp_threads": 16},
    }

    prepared = MODULE._prepare_case_qp(spec, case_dir=case_dir, force=False)

    assert prepared.omp_threads == 16
    assert prepared.qp_root == tmp_path / "qp_root_cached"
