from __future__ import annotations

from ccik import build_cas_hamiltonian_pyscf, ccik_ground_energy_dense, make_mol_pyscf
from ccik.params import CCIKParams
from ccik.pyscf_cas import CASSpec


def build_h4_chain_cas(r_ang: float = 1.5):
    atom = "; ".join([f"H 0 0 {i * r_ang}" for i in range(4)])
    mol = make_mol_pyscf(atom=atom, basis="sto-3g", unit="Angstrom", charge=0, spin=0, verbose=0)
    cas = CASSpec(ncas=4, nelecas=4, ncore=0)
    return build_cas_hamiltonian_pyscf(mol, cas=cas)


def run_case(selector_mode: str) -> None:
    h1, eri8, ecore, ncas, nelec = build_h4_chain_cas()
    params = CCIKParams(
        m=12,
        nadd=12,
        nkeep=18,
        Kv=10,
        candidate_mode="selected_ci",
        selector_mode=selector_mode,
        selector_depth=1,
        generator_keep=32,
        generator_abs_coeff_cutoff=0.0,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=8,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        exact_hq_diagnostics=True,
        verbose=False,
        orth_tol=1e-12,
    )
    stats: dict[str, object] = {}
    e_var = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=params, stats=stats))
    diag = stats.get("last_step_diagnostics", {})

    print(f"mode={selector_mode}")
    print(f"  variational_energy = {e_var:.12f}")
    print(f"  relative_matvec_error = {diag.get('relative_matvec_error')}")
    print(f"  discarded_matvec_weight = {diag.get('discarded_weight')}")
    print(f"  residual_norm_after_orthogonalization = {diag.get('orthogonalized_residual_norm')}")


def main() -> None:
    for mode in ("pt2_legacy", "abs_hq", "abs_residual"):
        run_case(mode)


if __name__ == "__main__":
    main()
