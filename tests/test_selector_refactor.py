import pytest

pyscf = pytest.importorskip("pyscf")

import numpy as np

from ccik import build_cas_hamiltonian_pyscf, make_mol_pyscf
from ccik.params import CCIKParams
from ccik.propagation import analyze_krylov_step, build_candidate_pool, build_next_krylov_vector, compress_vector
from ccik.pyscf_cas import CASSpec
from ccik.ritz import build_projected_columns, solve_projected_ritz
from ccik.selection import _propagation_source, build_projection_workspace, make_full_hdiag_map, selector_mode_from_params
from ccik.sparse_ci import SparseCIVector, linear_combination_sparse, make_hf_sparse_vector, normalize_sparse, project_sparse_to_support, topk_abs_keys_sparse
from ccik.thick_restart import _compress_with_stage1


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


def test_candidate_pool_respects_generator_limit_cutoff_and_current_support() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    qk = SparseCIVector(
        {
            (15, 15): 0.8,
            (7, 7): 0.2,
            (11, 11): 1e-4,
        }
    )
    params = CCIKParams(
        candidate_mode="selected_ci",
        selector_mode="abs_hq",
        selector_depth=1,
        generator_keep=2,
        generator_abs_coeff_cutoff=1e-3,
        max_parents_for_expansion=1,
        include_current_support=False,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
    )
    pool = build_candidate_pool(
        qk,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
    )

    assert (15, 15) in pool
    assert (7, 7) not in pool
    assert (11, 11) not in pool


def test_pt2_legacy_selector_mode_aliases_to_pt2_residual() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    params = CCIKParams(
        compression_policy="fixed_budget",
        candidate_mode="selected_ci",
        selector_mode="pt2_legacy",
        fixed_store_keep=18,
        fixed_propagate_keep=8,
        selector_depth=1,
        nadd=12,
        nkeep=18,
        Kv=10,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
    )
    q_next, diag, ws = build_next_krylov_vector(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
    )

    assert diag.selector_mode_used == "pt2_residual"
    assert q_next.coeffs
    assert len(ws.candidate_support) > 0


def test_ground_state_pt2_selector_alias_is_preserved() -> None:
    params_direct = CCIKParams(compression_policy="fixed_budget", selector_mode="ground_state_pt2")
    params_alias = CCIKParams(compression_policy="fixed_budget", selector_mode="ritz_pt2")

    assert selector_mode_from_params(params_direct) == "ground_state_pt2"
    assert selector_mode_from_params(params_alias) == "ground_state_pt2"


def test_legacy_selector_shorthands_remain_pt2_legacy() -> None:
    assert selector_mode_from_params(CCIKParams(selector_mode="cipsi")) == "pt2_legacy"
    assert selector_mode_from_params(CCIKParams(selector_mode="selected_ci")) == "pt2_legacy"
    assert selector_mode_from_params(CCIKParams(selector_mode="stochastic")) == "pt2_legacy"


def test_abs_residual_falls_back_to_abs_hq_without_basis() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    params = CCIKParams(candidate_mode="selected_ci", selector_mode="abs_residual", selector_ci_coeff_cutoff=1e-8, selector_cutoff=1e-8)
    _, diag, _ = build_next_krylov_vector(
        q0,
        [],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
    )

    assert diag.selector_mode_used == "abs_hq"


def test_support_preserving_compression_keeps_capped_old_support() -> None:
    vec = SparseCIVector({(1, 1): 0.11, (2, 2): 0.10, (3, 3): 1.0, (4, 4): 0.9})
    out = compress_vector(vec, 2, preserve_support={(1, 1), (2, 2)})
    assert set(out.coeffs) == {(1, 1), (2, 2)}


def test_full_candidate_mode_matches_exact_projected_matvec_on_h2() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    params = CCIKParams(candidate_mode="full", selector_mode="abs_hq", exact_hq_diagnostics=True)
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

    assert ws.exact_v is not None
    assert np.allclose(ws.v_proj.to_dense(norb=ncas, nelec=nelec), ws.exact_v.to_dense(norb=ncas, nelec=nelec))


def test_connected_sd_candidate_mode_matches_exact_projected_matvec_on_h2() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    params = CCIKParams(candidate_mode="connected_sd", selector_mode="abs_hq", exact_hq_diagnostics=True)
    ws = build_projection_workspace(
        qk=q0,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
    )

    assert ws.exact_v is not None
    assert np.allclose(ws.v_proj.to_dense(norb=ncas, nelec=nelec), ws.exact_v.to_dense(norb=ncas, nelec=nelec))


def test_threshold_mode_zero_threshold_matches_exact_projected_matvec_on_h2() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    params = CCIKParams(
        compression_policy="threshold",
        candidate_mode="full",
        selector_mode="abs_hq",
        exact_hq_diagnostics=True,
        store_abs_tol=0.0,
        propagate_abs_tol=0.0,
    )
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

    assert ws.exact_v is not None
    assert np.allclose(ws.v_proj.to_dense(norb=ncas, nelec=nelec), ws.exact_v.to_dense(norb=ncas, nelec=nelec))
    assert ws.propagation_support == q0.support()
    assert ws.propagation_dropped_norm == pytest.approx(0.0)


def test_threshold_mode_monotonically_reduces_propagation_support() -> None:
    qk = normalize_sparse(
        SparseCIVector(
            {
                (15, 15): 0.95,
                (7, 7): 0.20,
                (11, 11): 0.03,
            }
        )
    )
    params_lo = CCIKParams(compression_policy="threshold", candidate_mode="connected_sd", selector_mode="abs_hq", store_abs_tol=0.0, propagate_abs_tol=0.02)
    params_hi = CCIKParams(compression_policy="threshold", candidate_mode="connected_sd", selector_mode="abs_hq", store_abs_tol=0.0, propagate_abs_tol=0.2)

    _, supp_lo, drop_lo, _ = _propagation_source(qk, params_lo)
    _, supp_hi, drop_hi, _ = _propagation_source(qk, params_hi)

    assert len(supp_hi) < len(supp_lo)
    assert drop_hi >= drop_lo


def test_fixed_budget_mode_abs_residual_propagation_uses_topk_support() -> None:
    qk = normalize_sparse(
        SparseCIVector(
            {
                (15, 15): 0.95,
                (7, 7): 0.20,
                (11, 11): 0.03,
            }
        )
    )
    params = CCIKParams(
        compression_policy="fixed_budget",
        fixed_propagate_keep=2,
        fixed_store_keep=8,
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
    )
    qprop, supp, dropped_norm, used_fallback = _propagation_source(qk, params)

    assert used_fallback is False
    assert len(supp) == 2
    assert supp == {(15, 15), (7, 7)}
    assert set(qprop.coeffs) == supp
    assert dropped_norm > 0.0


def test_threshold_mode_propagation_fallback_keeps_largest_entry() -> None:
    qk = normalize_sparse(SparseCIVector({(15, 15): 0.8, (7, 7): 0.6}))
    params = CCIKParams(compression_policy="threshold", candidate_mode="connected_sd", selector_mode="abs_hq", store_abs_tol=0.0, propagate_abs_tol=2.0)
    _, supp, _, used_fallback = _propagation_source(qk, params)

    assert used_fallback is True
    assert len(supp) == 1
    assert (15, 15) in supp


def test_threshold_mode_ignores_topk_knobs_for_dense_step() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    p1 = CCIKParams(
        compression_policy="threshold",
        candidate_mode="full",
        selector_mode="abs_residual",
        store_abs_tol=1e-4,
        propagate_abs_tol=0.0,
        nadd=4,
        nkeep=5,
        Kv=2,
        generator_keep=1,
    )
    p2 = CCIKParams(
        compression_policy="threshold",
        candidate_mode="full",
        selector_mode="abs_residual",
        store_abs_tol=1e-4,
        propagate_abs_tol=0.0,
        nadd=400,
        nkeep=500,
        Kv=200,
        generator_keep=50,
        tmp_keep_factor=9,
    )

    q1, _, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=p1, hdiag_full=hdiag_full)
    q2, _, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=p2, hdiag_full=hdiag_full)

    assert np.allclose(q1.to_dense(norb=ncas, nelec=nelec), q2.to_dense(norb=ncas, nelec=nelec))


def test_fixed_budget_mode_ignores_threshold_knobs_for_dense_step() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    p1 = CCIKParams(
        compression_policy="fixed_budget",
        candidate_mode="full",
        selector_mode="abs_residual",
        fixed_store_keep=8,
        fixed_propagate_keep=4,
        store_abs_tol=1e-4,
        propagate_abs_tol=1e-2,
        nadd=4,
        nkeep=5,
        Kv=2,
        generator_keep=1,
    )
    p2 = CCIKParams(
        compression_policy="fixed_budget",
        candidate_mode="full",
        selector_mode="abs_residual",
        fixed_store_keep=8,
        fixed_propagate_keep=4,
        store_abs_tol=5e-2,
        propagate_abs_tol=4e-1,
        nadd=400,
        nkeep=500,
        Kv=200,
        generator_keep=50,
        tmp_keep_factor=9,
    )

    q1, _, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=p1, hdiag_full=hdiag_full)
    q2, _, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=p2, hdiag_full=hdiag_full)

    assert np.allclose(q1.to_dense(norb=ncas, nelec=nelec), q2.to_dense(norb=ncas, nelec=nelec))


def test_threshold_mode_thick_restart_uses_store_abs_tol() -> None:
    from ccik.params import CCIKThickRestartParams

    vec = normalize_sparse(SparseCIVector({(1, 1): 0.9, (2, 2): 0.08, (3, 3): 0.02}))
    params = CCIKThickRestartParams(compression_policy="threshold", store_abs_tol=0.05)
    out = _compress_with_stage1(vec, 1, params)

    assert set(out.coeffs) == {(1, 1), (2, 2)}


def test_fixed_budget_mode_thick_restart_uses_fixed_store_keep() -> None:
    from ccik.params import CCIKThickRestartParams

    vec = normalize_sparse(SparseCIVector({(1, 1): 0.9, (2, 2): 0.08, (3, 3): 0.02}))
    params = CCIKThickRestartParams(compression_policy="fixed_budget", fixed_store_keep=2)
    out = _compress_with_stage1(vec, 1, params)

    assert set(out.coeffs) == {(1, 1), (2, 2)}


def test_fixed_budget_mode_dense_step_keeps_at_most_fixed_store_support() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    params = CCIKParams(
        compression_policy="fixed_budget",
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        fixed_store_keep=1,
        fixed_propagate_keep=4,
    )

    q1, diag, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=params)

    assert len(q1.coeffs) <= 1
    assert diag.num_final_kept <= 1


def test_ground_state_pt2_requires_fixed_budget_policy() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)

    for policy in ("topk", "threshold"):
        params = CCIKParams(
            compression_policy=policy,
            candidate_mode="full",
            selector_mode="ground_state_pt2",
            store_abs_tol=0.0,
            propagate_abs_tol=0.0,
        )
        with pytest.raises(ValueError, match="fixed_budget"):
            build_next_krylov_vector(
                q0,
                [q0],
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=ncas,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )


def test_pt2_residual_requires_fixed_budget_policy() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)

    for policy in ("topk", "threshold"):
        params = CCIKParams(
            compression_policy=policy,
            candidate_mode="full",
            selector_mode="pt2_residual",
            store_abs_tol=0.0,
            propagate_abs_tol=0.0,
        )
        with pytest.raises(ValueError, match="fixed_budget"):
            build_next_krylov_vector(
                q0,
                [q0],
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=ncas,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )


def test_pt2_residual_rejects_stochastic_candidate_discovery() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    params = CCIKParams(
        compression_policy="fixed_budget",
        candidate_mode="stochastic",
        selector_mode="pt2_residual",
        fixed_store_keep=4,
        fixed_propagate_keep=2,
    )
    with pytest.raises(ValueError, match="deterministic"):
        build_next_krylov_vector(
            q0,
            [q0],
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=ncas,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
            rng=np.random.default_rng(7),
        )


def test_pt2_residual_single_vector_h2_supports_match_manual_formula() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    common = dict(
        compression_policy="fixed_budget",
        candidate_mode="full",
        fixed_store_keep=4,
        fixed_propagate_keep=2,
        orth_tol=1e-12,
        verbose=False,
    )

    q_pt2, _, _, art_pt2 = analyze_krylov_step(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(selector_mode="pt2_residual", **common),
        hdiag_full=hdiag_full,
    )
    probe_ws = build_projection_workspace(
        qk=q0,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(selector_mode="abs_residual", **common),
        hdiag_full=hdiag_full,
    )
    probe_residual = normalize_sparse(project_sparse_to_support(probe_ws.v_proj, set(probe_ws.v_proj.coeffs)))
    probe_residual = probe_ws.v_proj
    from ccik.propagation import orthogonalize_against_basis
    probe_residual = orthogonalize_against_basis(probe_residual, [q0])
    ek = float(sum(q0.coeffs.get(det, 0.0) * probe_ws.v_proj.get(det, 0.0) for det in q0.coeffs))
    manual_scores = {}
    for det, amp in probe_residual.coeffs.items():
        denom = max(abs(ek - probe_ws.hdiag.get(det, ek)), 1e-12)
        manual_scores[det] = (abs(amp) ** 2) / denom
    manual_store_support = {
        key
        for key, _ in sorted(manual_scores.items(), key=lambda item: item[1], reverse=True)[: common["fixed_store_keep"]]
    }
    q_pt2_scores = {}
    for det, amp in q_pt2.coeffs.items():
        denom = max(abs(ek - probe_ws.hdiag.get(det, ek)), 1e-12)
        q_pt2_scores[det] = (abs(amp) ** 2) / denom
    manual_prop_support = {
        key
        for key, _ in sorted(q_pt2_scores.items(), key=lambda item: item[1], reverse=True)[: common["fixed_propagate_keep"]]
    }

    assert art_pt2.score_support == manual_store_support
    assert art_pt2.propagation_score_support == manual_prop_support
    assert q_pt2.delayed_propagation_support == manual_prop_support
    assert q_pt2.coeffs


def test_ground_state_pt2_fixed_budget_support_matches_manual_score_formula() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    common = dict(
        compression_policy="fixed_budget",
        candidate_mode="full",
        fixed_store_keep=4,
        fixed_propagate_keep=4,
        selector_depth=1,
        generator_keep=4,
        max_parents_for_expansion=4,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        orth_tol=1e-12,
        verbose=False,
    )
    q1, _, _ = build_next_krylov_vector(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(selector_mode="abs_residual", **common),
        hdiag_full=hdiag_full,
    )
    params = CCIKParams(selector_mode="ground_state_pt2", **common)
    _, diag, ws, artifacts = analyze_krylov_step(
        q1,
        [q0, q1],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )

    projected = solve_projected_ritz(
        Q=[q0, q1],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )
    coeffs = projected.eigvecs[:, 0]
    pool_columns = build_projected_columns(
        [q0, q1],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
        candidate_pool_override=ws.candidate_pool,
    )
    target = project_sparse_to_support(
        linear_combination_sparse((float(coeffs[i]), pool_columns[i]) for i in range(2)),
        ws.candidate_pool,
    )
    manual_scores = {}
    for det in ws.candidate_pool:
        amp = target.get(det, 0.0)
        if amp == 0.0:
            continue
        denom = max(abs(float(projected.evals[0]) - ws.hdiag.get(det, float(projected.evals[0]))), 1e-12)
        manual_scores[det] = (abs(amp) ** 2) / denom
    manual_support = {
        key
        for key, _ in sorted(manual_scores.items(), key=lambda item: item[1], reverse=True)[: params.fixed_store_keep]
    }

    assert diag.selector_mode_used == "ground_state_pt2"
    assert artifacts.score_support == manual_support


def test_fixed_budget_selector_mode_changes_support_ranking() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    common = dict(
        compression_policy="fixed_budget",
        candidate_mode="full",
        fixed_store_keep=4,
        fixed_propagate_keep=4,
        selector_depth=1,
        generator_keep=4,
        max_parents_for_expansion=4,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        orth_tol=1e-12,
        verbose=False,
    )
    q1, _, _ = build_next_krylov_vector(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(selector_mode="abs_residual", **common),
        hdiag_full=hdiag_full,
    )
    _, _, _, art_abs_hq = analyze_krylov_step(
        q1,
        [q0, q1],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(selector_mode="abs_hq", **common),
        hdiag_full=hdiag_full,
    )
    _, _, _, art_ground = analyze_krylov_step(
        q1,
        [q0, q1],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(selector_mode="pt2_residual", **common),
        hdiag_full=hdiag_full,
    )

    assert art_abs_hq.score_support != art_ground.score_support


def test_pt2_residual_changes_fixed_budget_propagation_support() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    q1, _, _ = build_next_krylov_vector(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(
            compression_policy="fixed_budget",
            candidate_mode="full",
            selector_mode="abs_residual",
            fixed_store_keep=8,
            fixed_propagate_keep=4,
        ),
        hdiag_full=hdiag_full,
    )
    q2, _, _, artifacts = analyze_krylov_step(
        q1,
        [q0, q1],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=CCIKParams(
            compression_policy="fixed_budget",
            candidate_mode="full",
            selector_mode="pt2_residual",
            fixed_store_keep=8,
            fixed_propagate_keep=4,
        ),
        hdiag_full=hdiag_full,
    )

    assert artifacts.propagation_score_support
    assert q2.delayed_propagation_support == artifacts.propagation_score_support
    assert artifacts.propagation_score_support.issubset(q2.support())


def test_pt2_residual_reuses_delayed_propagation_support_on_next_step() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    params = CCIKParams(
        compression_policy="fixed_budget",
        candidate_mode="full",
        selector_mode="pt2_residual",
        fixed_store_keep=8,
        fixed_propagate_keep=4,
    )
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    q1, _, _ = build_next_krylov_vector(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )
    _, diag1, ws1, _ = analyze_krylov_step(
        q1,
        [q0, q1],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )

    assert q1.delayed_propagation_support
    assert ws1.propagation_support == q1.delayed_propagation_support
    assert diag1.delayed_propagation_used_fallback is False


def test_threshold_mode_storage_fallback_keeps_largest_residual_entry() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h2_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=ncas, nelec=nelec)
    params = CCIKParams(
        compression_policy="threshold",
        candidate_mode="full",
        selector_mode="abs_residual",
        store_abs_tol=2.0,
        propagate_abs_tol=0.0,
    )

    q1, diag, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=params, hdiag_full=hdiag_full)

    assert diag.store_used_fallback is True
    assert len(q1.coeffs) == 1


def test_threshold_mode_diagnostics_are_populated_and_sane() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    params = CCIKParams(
        compression_policy="threshold",
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        store_abs_tol=1e-3,
        propagate_abs_tol=1e-2,
    )
    _, diag, _, artifacts = analyze_krylov_step(
        q0,
        [q0],
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=ncas,
        nelec=nelec,
        params=params,
    )

    assert diag.compression_policy_used == "threshold"
    assert diag.propagation_support_size >= 1
    assert diag.propagation_dropped_norm >= 0.0
    assert diag.store_dropped_norm >= 0.0
    assert diag.num_final_kept == len(artifacts.compressed.coeffs)
    assert diag.post_compression_residual_norm <= 1.0 + 1e-12


def test_threshold_mode_monotonically_reduces_stored_support() -> None:
    from pyscf.fci import direct_spin1

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    p_lo = CCIKParams(
        compression_policy="threshold",
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        store_abs_tol=1e-8,
        propagate_abs_tol=0.0,
    )
    p_hi = CCIKParams(
        compression_policy="threshold",
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        store_abs_tol=5e-2,
        propagate_abs_tol=0.0,
    )

    _, diag_lo, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=p_lo)
    _, diag_hi, _ = build_next_krylov_vector(q0, [q0], h1=h1, eri8=eri8, h2eff=h2eff, norb=ncas, nelec=nelec, params=p_hi)

    assert diag_hi.num_final_kept <= diag_lo.num_final_kept
    assert diag_hi.store_dropped_norm >= diag_lo.store_dropped_norm


def test_threshold_mode_candidate_limit_failure_is_explicit(monkeypatch) -> None:
    from pyscf.fci import direct_spin1
    import ccik.selection as selection_mod

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    params = CCIKParams(
        compression_policy="threshold",
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        store_abs_tol=0.0,
        propagate_abs_tol=0.0,
    )
    monkeypatch.setattr(selection_mod, "_THRESHOLD_MAX_CANDIDATES", 1)

    with pytest.raises(RuntimeError, match="internal candidate limit"):
        build_projection_workspace(
            qk=q0,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=ncas,
            nelec=nelec,
            params=params,
        )


def test_threshold_mode_stored_support_limit_failure_is_explicit(monkeypatch) -> None:
    from pyscf.fci import direct_spin1
    import ccik.propagation as propagation_mod

    h1, eri8, _, ncas, nelec = _build_h4_chain_cas()
    q0 = make_hf_sparse_vector(norb=ncas, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, ncas, nelec, fac=0.5)  # type: ignore[arg-type]
    params = CCIKParams(
        compression_policy="threshold",
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        store_abs_tol=0.0,
        propagate_abs_tol=0.0,
    )
    monkeypatch.setattr(propagation_mod, "_THRESHOLD_MAX_STORED", 1)

    with pytest.raises(RuntimeError, match="stored-support limit"):
        build_next_krylov_vector(
            q0,
            [q0],
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=ncas,
            nelec=nelec,
            params=params,
        )
