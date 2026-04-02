import pytest

pyscf = pytest.importorskip("pyscf")

import numpy as np

from ccik import build_cas_hamiltonian_pyscf, make_mol_pyscf
from ccik.params import CCIKParams, CCIKStochasticParams
from ccik.pyscf_cas import CASSpec
from ccik.selection import build_projection_workspace, make_full_hdiag_map
from ccik.sparse_ci import SparseCIVector, compress_topk_sparse, make_hf_sparse_vector, sparse_from_dense, topk_abs_keys_sparse



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



def test_sparse_dense_roundtrip_preserves_coefficients() -> None:
    h1, eri8, ecore, ncas, nelec = _build_h2_cas()
    _ = (h1, eri8, ecore)
    ci = np.zeros((2, 2), dtype=float)
    ci[0, 0] = 0.8
    ci[1, 0] = -0.6
    sparse = sparse_from_dense(ci, norb=ncas, nelec=nelec)
    dense = sparse.to_dense(norb=ncas, nelec=nelec)
    assert np.allclose(dense, ci)


def test_topk_compression_uses_qp_like_canonical_tie_break() -> None:
    vec = SparseCIVector(
        {
            (5, 5): 1.0,
            (1, 1): -1.0,
            (3, 3): 1.0,
        }
    )

    compressed = compress_topk_sparse(vec, 2)

    assert compressed.support() == {(3, 3), (5, 5)}
    assert topk_abs_keys_sparse(vec, 2) == {(3, 3), (5, 5)}



def test_full_selector_workspace_matches_dense_contract() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    dense_q0 = q0.to_dense(norb=ncas, nelec=nelec)
    dense_v = np.asarray(direct_spin1.contract_2e(h2eff, dense_q0, ncas, nelec), dtype=float)

    params = CCIKParams(selector_mode="full")
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    ws = build_projection_workspace(
        qk=q0,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )

    assert np.allclose(ws.v_proj.to_dense(norb=ncas, nelec=nelec), dense_v)
    assert len(ws.candidate_support) == len(hdiag_full) - len(q0.support())



def test_stochastic_selector_depth_accumulates_candidate_support() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas(r_ang=1.5)
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]

    params_d1 = CCIKStochasticParams(
        selector_mode="stochastic",
        selector_depth=1,
        n_walkers=4000,
        seed=7,
        nadd=12,
        nkeep=50,
        Kv=10,
    )
    params_d2 = CCIKStochasticParams(
        selector_mode="stochastic",
        selector_depth=2,
        n_walkers=4000,
        seed=7,
        nadd=12,
        nkeep=50,
        Kv=10,
    )

    ws1 = build_projection_workspace(
        qk=q0,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params_d1,
        rng=np.random.default_rng(7),
    )
    ws2 = build_projection_workspace(
        qk=q0,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params_d2,
        rng=np.random.default_rng(7),
    )

    assert len(ws2.candidate_support) >= len(ws1.candidate_support)
    assert q0.support().isdisjoint(ws2.candidate_support)
