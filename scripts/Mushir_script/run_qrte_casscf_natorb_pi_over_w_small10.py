#!/usr/bin/env python
"""Run the same ten MolVQE cases with dt = pi / (Emax - E0) of the cached CAS.

Diagonalize the complete interacting CASSCF natural-orbital active-space
Hamiltonian at fixed active Nalpha/Nbeta. QRTE keeps its original reference
determinant, exact Hamiltonian cache, and chemical-accuracy target.
An optional --delta-t overrides pi/W without changing the Hamiltonian or reference.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import run_qrte_fock_pi_over_w as qrte
from run_qrte_uhf_determinant_pi_over_w_small10 import CASE_IDS, load_cases

np = qrte.np
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_casscf_natorb_pi_over_w_small10"
SPECTRUM_SCOPE = "interacting_casscf_natorb_active_fixed_nalpha_nbeta"
SPECTRUM_FILENAME = "active_space_reference.json"
REFERENCE_MATCH_TOLERANCE = 1e-6


def compute_active_space_spectrum(case, case_dir, args):
    from pyscf import fci
    from scipy.linalg import eigvalsh

    cache_path = Path(case["hamiltonian_cache_path"])
    with np.load(cache_path, allow_pickle=False) as cache:
        norb = int(cache["ncas"])
        nelecas = tuple(int(n) for n in cache["nelecas"])
        h1 = np.asarray(cache["h1ecas"], dtype=float)
        # The cache uses PennyLane's two-electron index convention. Restore
        # PySCF ordering before constructing the complete CI Hamiltonian.
        eri = np.swapaxes(np.asarray(cache["two_mo"], dtype=float), 1, 3)
        ecore = float(cache["ecore"])
        reference_energy = float(cache["reference_energy"])
        source = str(cache["hamiltonian_source"].item())
    if (source != "casscf_natorb" or norb != case["active_orbitals"]
            or list(nelecas) != case["nelecas"] or len(nelecas) != 2
            or any(n < 0 or n > norb for n in nelecas)):
        raise ValueError(f"{case['case_id']}: CASSCF/natorb cache sector mismatch")
    if h1.shape != (norb, norb) or eri.shape != (norb,) * 4:
        raise ValueError("Cached active-space integral dimensions are invalid")
    dimension = math.comb(norb, nelecas[0]) * math.comb(norb, nelecas[1])
    start = time.perf_counter()
    # pspace contains the entire CI sector when np equals its full dimension.
    addresses, matrix = fci.direct_spin1.pspace(h1, eri, norb, nelecas, np=dimension)
    if (matrix.shape != (dimension, dimension) or len(addresses) != dimension
            or not np.array_equal(np.sort(addresses), np.arange(dimension))):
        raise RuntimeError("FCI matrix does not contain the complete fixed-occupation sector")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-10):
        raise RuntimeError("Active-space CI Hamiltonian is not Hermitian")
    energies = eigvalsh(matrix) + ecore
    if not np.all(np.isfinite(energies)):
        raise RuntimeError("Active-space diagonalization returned nonfinite eigenvalues")
    ground, highest = float(energies[0]), float(energies[-1])
    difference = ground - reference_energy
    if (not math.isfinite(reference_energy)
            or abs(reference_energy - case["local_reference_energy_hartree"]) > REFERENCE_MATCH_TOLERANCE
            or abs(difference) > REFERENCE_MATCH_TOLERANCE):
        raise RuntimeError(f"{case['case_id']}: sector E0 differs from the cached CASSCF reference "
                           f"by {difference:.12g} Eh; tolerance={REFERENCE_MATCH_TOLERANCE:g} Eh")
    result = {
        "reference_method": "CASSCF/natorb active-space exact CI spectrum",
        "spectrum_scope": SPECTRUM_SCOPE,
        "hamiltonian_cache_path": str(cache_path), "hamiltonian_sha256": case["hamiltonian_sha256"],
        "hamiltonian_source": source,
        "active_orbitals": norb, "active_spin_occupations": list(nelecas),
        "sector_dim": dimension, "n_eigenvalues": int(energies.size),
        "sector_ground_energy_hartree": ground,
        "sector_highest_energy_hartree": highest,
        "sector_energies_hartree": energies.tolist(),
        "ecore_hartree": ecore, "cached_casscf_energy_hartree": reference_energy,
        "sector_ground_minus_cached_reference_hartree": difference,
        "reference_match_tolerance_hartree": REFERENCE_MATCH_TOLERANCE,
        "solver": "pyscf.direct_spin1.pspace (full sector) + scipy.linalg.eigvalsh",
        "symmetry_sector": "fixed active electron number and Sz; no extra total-spin or spatial-irrep projection",
        "elapsed_seconds": time.perf_counter() - start,
    }
    width_from_reference(result)
    return result


def width_from_reference(result):
    ground = float(result["sector_ground_energy_hartree"])
    highest = float(result["sector_highest_energy_hartree"])
    width = highest - ground
    if not math.isfinite(ground) or not math.isfinite(highest) or not math.isfinite(width) or width <= 0:
        raise ValueError(f"Active-space bandwidth must be positive and finite, got {width}")
    return {
        "width_definition": "Emax - E0 of the interacting CASSCF/natorb active-space Hamiltonian",
        "width_reference_method": "CASSCF/natorb active-space exact CI spectrum",
        "spectrum_scope": SPECTRUM_SCOPE,
        "active_space_ground_energy_hartree": ground,
        "active_space_highest_energy_hartree": highest,
        "active_space_bandwidth_hartree": width,
        "sector_dim": int(result["sector_dim"]),
        "active_spin_occupations": result["active_spin_occupations"],
        "delta_t": math.pi / width,
        "delta_t_units": "atomic units of time",
        "delta_t_rule": "pi / (Emax - E0); interacting CASSCF/natorb active-space Hamiltonian",
    }


def parse_args():
    return qrte.parse_args(default_outdir=DEFAULT_OUTDIR, description=__doc__)


def main():
    args = parse_args()
    cases = load_cases(args)
    if args.count_cases:
        print(len(cases))
        return
    if args.dry_run:
        for i, case in enumerate(cases):
            dimension = math.prod(math.comb(case["active_orbitals"], n) for n in case["nelecas"])
            print(f"{i:2d} {case['case_id']:30s} W_source=CASSCF/natorb "
                  f"CAS={case['active_electrons']}e{case['active_orbitals']}o sector_dim={dimension}")
        print(f"Validated {len(cases)} cases; W=active-space Emax-E0; max_iterations={args.n_steps}; "
              f"overlap_tol={args.overlap_tol:g}; evolution={args.evolution_mode}; "
              f"trotter_order={args.trotter_order}; trotter_steps={args.trotter_steps}; "
              f"dt={args.delta_t if args.delta_t is not None else 'pi/W'}; "
              f"early_stop=chemical_accuracy; outdir={args.outdir}")
        return
    failures = 0
    for case in cases:
        try:
            qrte.run_case(case, args, spectrum_builder=compute_active_space_spectrum,
                          width_builder=width_from_reference, spectrum_scope=SPECTRUM_SCOPE,
                          spectrum_filename=SPECTRUM_FILENAME)
        except Exception as exc:
            failures += 1
            print(f"FAIL {case['case_id']}: {exc!r}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
