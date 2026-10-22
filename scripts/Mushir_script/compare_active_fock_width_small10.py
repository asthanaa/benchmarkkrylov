#!/usr/bin/env python
"""Compare saved exact CAS widths with active-orbital Fock estimates.

No SCF, CASSCF, CI, or QRTE runs are launched. The canonical-HF window is an
estimate of the active orbital selection, not the optimized CASSCF subspace.
A separate frozen-reference Fock estimate uses the actual cached CAS integrals.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import run_qrte_casscf_natorb_pi_over_w_small10 as cas
from run_qrte_uhf_determinant_pi_over_w_small10 import determinant_bandwidth

np = cas.np
ROOT = cas.qrte.ROOT


def orbital_estimates(alpha, beta, occupations):
    alpha, beta = (np.asarray(energies, dtype=float) for energies in (alpha, beta))
    width = determinant_bandwidth(alpha, beta, occupations)
    return {
        "alpha_orbital_energies_hartree": alpha.tolist(),
        "beta_orbital_energies_hartree": beta.tolist(),
        "active_spin_occupations": list(occupations),
        "one_electron_range_hartree": float(max(alpha.max(), beta.max()) - min(alpha.min(), beta.min())),
        "lowest_determinant_orbital_sum_hartree": width["lowest_determinant_orbital_sum_hartree"],
        "highest_determinant_orbital_sum_hartree": width["highest_determinant_orbital_sum_hartree"],
        "determinant_width_hartree": width["fock_bandwidth_hartree"],
        "delta_t_pi_over_estimated_w": width["delta_t"],
        "determinant_energy_definition": "fixed active Nalpha/Nbeta orbital-energy sums of one frozen Fock operator; no relaxation of extremal determinants",
    }


def canonical_window(case, reference):
    energies = np.asarray(reference["full_scf_orbital_energies_hartree"], dtype=float)
    occupancies = np.asarray(reference["full_scf_orbital_occupations"], dtype=float)
    if (energies.ndim != 1 or energies.shape != occupancies.shape
            or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(occupancies))):
        raise ValueError("Invalid saved mean-field orbital energies or occupations")
    inactive_electrons = float(occupancies.sum()) - case["active_electrons"]
    ncore = int(round(inactive_electrons / 2))
    if ncore < 0 or abs(inactive_electrons - 2 * ncore) > 1e-8:
        raise ValueError("Inactive electrons do not form a closed-shell core")
    indices = case["active_orbital_indices"]
    if indices:
        indices = [int(i) - case["active_orbital_index_base"] for i in indices]
        selection = "recorded initial canonical HF active orbital indices"
    else:
        indices = list(range(ncore, ncore + case["active_orbitals"]))
        selection = "contiguous canonical HF window after the frozen-core orbitals"
    if (len(indices) != case["active_orbitals"] or len(set(indices)) != len(indices)
            or any(i < 0 or i >= energies.size for i in indices)):
        raise ValueError("Mean-field active orbital indices are invalid")
    if abs(float(occupancies[indices].sum()) - case["active_electrons"]) > 1e-8:
        raise ValueError("Selected mean-field orbitals do not contain the active electrons")
    result = orbital_estimates(energies[indices], energies[indices], case["nelecas"])
    result.update(reference_method=reference["reference_scf"],
                  fock_operator=reference["fock_operator"], ncore=ncore,
                  active_orbital_indices_zero_based=indices,
                  active_selection=selection,
                  active_subspace_matches_optimized_casscf="not established from saved orbital energies",
                  caveat="CASSCF can rotate active orbitals into inactive/virtual space. Saved scalar HF orbital energies cannot reconstruct that optimized subspace.")
    if case["active_orbital_labels"]:
        result["avas_caveat"] = "AVAS labels were recorded; this energy-window baseline does not reconstruct the AVAS projector."
    if ncore == 0 and len(indices) == energies.size:
        result["active_subspace_matches_optimized_casscf"] = True
        result["caveat"] = "The active space is the full molecular orbital space; CASSCF rotations cannot change that subspace."
    if reference["reference_scf"] in {"ROHF", "sfX2C1eROHF"}:
        result["open_shell_caveat"] = "Uses the saved effective Roothaan spatial energies for both spins, not separate physical alpha/beta Fock spectra."
    return result


def frozen_cas_fock(h1, eri, occupations):
    from pyscf.scf.hf import dot_eri_dm

    h1, eri = np.asarray(h1, dtype=float), np.asarray(eri, dtype=float)
    if h1.ndim != 2 or h1.shape[0] != h1.shape[1] or eri.shape != h1.shape * 2:
        raise ValueError("Invalid active-space integrals")
    if len(occupations) != 2 or any(int(n) != n or not 0 <= n <= h1.shape[0] for n in occupations):
        raise ValueError("Invalid active spin occupations")
    if not np.all(np.isfinite(h1)) or not np.all(np.isfinite(eri)):
        raise ValueError("Nonfinite active-space integrals")
    densities = np.array([np.diag(np.arange(h1.shape[0]) < int(n)).astype(float)
                          for n in occupations])
    coulomb, exchange = dot_eri_dm(eri, densities, hermi=1)
    fock = h1[None, :, :] + coulomb.sum(axis=0)[None, :, :] - exchange
    if not np.allclose(fock, fock.transpose(0, 2, 1), rtol=0, atol=1e-10):
        raise ValueError("Frozen reference Fock matrices are not Hermitian")
    spectra = np.linalg.eigvalsh(fock)
    result = orbital_estimates(spectra[0], spectra[1], occupations)
    result.update(reference_method="frozen QRTE reference determinant in the optimized CAS",
                  active_subspace_matches_optimized_casscf=True,
                  alpha_fock_matrix_hartree=fock[0].tolist(), beta_fock_matrix_hartree=fock[1].tolist(),
                  caveat="Not the original molecular SCF spectrum and not a new self-consistent HF solution. Requires the cached CASSCF integrals; only the small one-electron Fock matrices are diagonalized.")
    return result


def compare_case(case, fock_root, exact_root):
    fock_path = fock_root / case["case_id"] / "fock_reference.json"
    exact_path = exact_root / case["case_id"] / "summary.json"
    reference, exact = cas.qrte.load_json(fock_path), cas.qrte.load_json(exact_path)
    if reference["status"] != "completed" or not reference["scf_converged"]:
        raise ValueError("Saved mean-field calculation is not converged")
    if exact["status"] != "completed":
        raise ValueError("Saved exact run is not complete")
    for saved in (reference["fock_profile"]["case"], exact):
        if saved["case_id"] != case["case_id"] or saved["hamiltonian_sha256"] != case["hamiltonian_sha256"]:
            raise ValueError("Saved run and validated Hamiltonian cache provenance differ")
    width = float(exact["active_space_bandwidth_hartree"])
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Invalid saved exact spectral width")
    with np.load(case["hamiltonian_cache_path"], allow_pickle=False) as cache:
        h1 = cache["h1ecas"]
        # Restore PySCF's chemist-index ordering from the QRTE cache convention.
        eri = np.swapaxes(cache["two_mo"], 1, 3)
        frozen = frozen_cas_fock(h1, eri, case["nelecas"])
    canonical = canonical_window(case, reference)
    for estimate in (canonical, frozen):
        ratio = estimate["determinant_width_hartree"] / width
        estimate.update(w_ratio_to_exact=ratio, relative_w_error_percent=100 * (ratio - 1),
                        timestep_ratio_to_exact=1 / ratio)
    return {"case_id": case["case_id"], "hamiltonian_sha256": case["hamiltonian_sha256"],
            "hamiltonian_cache_path": case["hamiltonian_cache_path"],
            "exact_width_hartree": width, "exact_delta_t": float(exact["delta_t"]),
            "source_fock_reference": str(fock_path), "source_exact_summary": str(exact_path),
            "canonical_hf_window": canonical, "frozen_natorb_reference_fock": frozen}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fock-root", type=Path, default=ROOT / "runs/qrte_fock_pi_over_w")
    parser.add_argument("--exact-root", type=Path, default=ROOT / "runs/qrte_casscf_natorb_pi_over_w_small10_exact")
    parser.add_argument("--outdir", type=Path, default=ROOT / "runs/active_fock_width_small10")
    args = parser.parse_args()
    case_args = argparse.Namespace(manifest=cas.qrte.DEFAULT_MANIFEST,
                                   cache_manifest=cas.qrte.DEFAULT_CACHE_MANIFEST,
                                   active_orbital_overrides=ROOT / "manifests/molvqe21_active_orbital_overrides.json",
                                   case_id=None, case_index=None)
    cases = cas.load_cases(case_args)
    results = [compare_case(case, args.fock_root, args.exact_root) for case in cases]
    cas.qrte.write_json(args.outdir / "comparison.json", {
        "methods": "canonical molecular HF window baseline and frozen-reference Fock in the actual optimized CAS",
        "new_molecular_calculations": False, "cases": results})
    columns = ["case_id", "exact_w_hartree", "canonical_one_electron_range_hartree",
               "canonical_determinant_w_hartree", "canonical_w_ratio", "canonical_error_percent",
               "frozen_natorb_determinant_w_hartree", "frozen_natorb_w_ratio", "frozen_natorb_error_percent"]
    with (args.outdir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        print("Case                               Exact W   HF-window W   ratio   CAS-ref Fock W   ratio")
        for result in results:
            canonical, frozen = result["canonical_hf_window"], result["frozen_natorb_reference_fock"]
            writer.writerow([result["case_id"], result["exact_width_hartree"], canonical["one_electron_range_hartree"],
                             canonical["determinant_width_hartree"], canonical["w_ratio_to_exact"], canonical["relative_w_error_percent"],
                             frozen["determinant_width_hartree"], frozen["w_ratio_to_exact"], frozen["relative_w_error_percent"]])
            print(f"{result['case_id']:34s} {result['exact_width_hartree']:8.4f} "
                  f"{canonical['determinant_width_hartree']:11.4f} {canonical['w_ratio_to_exact']:7.3f} "
                  f"{frozen['determinant_width_hartree']:15.4f} {frozen['w_ratio_to_exact']:7.3f}")
    print(f"Saved comparison to {args.outdir}")


if __name__ == "__main__":
    main()
