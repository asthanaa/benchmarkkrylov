import json

import pytest

pyscf = pytest.importorskip("pyscf")

from pyscf.fci import direct_spin1

from ccik import build_cas_hamiltonian_pyscf, make_mol_pyscf
from ccik.params import CCIKParams
from ccik.pyscf_cas import CASSpec
from ccik.reference_debug import compare_single_krylov_step, summarize_trace, trace_ccik_run
from ccik.sparse_ci import make_hf_sparse_vector


def _build_h2_cas():
    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    cas = CASSpec(ncas=2, nelecas=2, ncore=0)
    return build_cas_hamiltonian_pyscf(mol, cas=cas)


def _build_h4_chain_cas(r_ang: float = 1.5):
    atom = "; ".join([f"H 0 0 {i * r_ang}" for i in range(4)])
    mol = make_mol_pyscf(atom=atom, basis="sto-3g", unit="Angstrom", charge=0, spin=0, verbose=0)
    cas = CASSpec(ncas=4, nelecas=4, ncore=0)
    return build_cas_hamiltonian_pyscf(mol, cas=cas)


def test_compare_single_krylov_step_full_mode_matches_oracle_on_h2() -> None:
    h1, eri8, _, ncas, nelec = _build_h2_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    params = CCIKParams(candidate_mode="full", selector_mode="abs_hq", exact_hq_diagnostics=True)

    comp = compare_single_krylov_step(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
    )

    assert comp.qnext_l2_diff == pytest.approx(0.0, abs=1e-14)
    assert comp.projected_support_overlap == comp.projected_support_union
    assert comp.qnext_support_overlap == comp.qnext_support_union


def test_trace_ccik_run_writes_jsonl_and_summary(tmp_path) -> None:
    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    trace_path = tmp_path / "trace.jsonl"
    params = CCIKParams(
        m=3,
        candidate_mode="selected_ci",
        selector_mode="abs_residual",
        nadd=16,
        nkeep=32,
        Kv=8,
        generator_keep=8,
        max_parents_for_expansion=8,
        exact_hq_diagnostics=True,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
    )

    traces = trace_ccik_run(h1, eri8, ncas, nelec, params=params, output_path=trace_path)
    summary = summarize_trace(traces)

    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(traces) == summary["n_steps"]
    assert summary["n_steps"] > 0
    assert "qnext_l2_diff" in rows[0]
