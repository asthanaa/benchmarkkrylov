#!/usr/bin/env python
"""Second-order QRTE for ten cases with dt=pi/W from saved RHF/ROHF energies.

W is the highest minus lowest fixed-active-Nalpha/Nbeta orbital-energy sum
in the canonical HF active-sized window (or recorded initial indices).
This is an estimate, not the interacting CAS spectrum or a reconstruction of
optimized CASSCF/AVAS orbitals. No SCF or CAS diagonalization is performed.
QRTE still uses the validated CASSCF/natural-orbital Hamiltonian and reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import run_qrte_fock_pi_over_w as qrte
from compare_active_fock_width_small10 import canonical_window, orbital_estimates
from run_qrte_uhf_determinant_pi_over_w_small10 import CASE_IDS, load_cases as load_subset

DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_pi_over_w_small10_trotter2"
FOCK_ROOT = qrte.ROOT / "runs/qrte_fock_pi_over_w"
SPECTRUM_SCOPE = "saved_molecular_hf_active_window_fixed_nalpha_nbeta"
SPECTRUM_FILENAME = "mean_field_active_reference.json"
PROVENANCE_FIELDS = (
    "case_id", "symbols", "geometry_angstrom", "basis", "charge", "spin",
    "active_electrons", "active_orbitals", "nelecas", "active_orbital_indices",
    "active_orbital_index_base", "active_orbital_labels", "reference_settings",
    "hamiltonian_sha256", "hamiltonian_source",
)


def read_saved_reference(case, args):
    path = Path(case["mean_field_reference_path"])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if case.get("mean_field_reference_sha256", digest) != digest:
        raise ValueError(f"{case['case_id']}: saved mean-field spectrum changed after input validation")
    reference = json.loads(data)
    if reference.get("status") != "completed" or reference.get("scf_converged") is not True:
        raise ValueError(f"{case['case_id']}: saved mean-field reference is not completed and converged")
    energy = float(reference["scf_energy_hartree"])
    if not math.isfinite(energy):
        raise ValueError("Saved SCF energy is not finite")
    saved_case = reference["fock_profile"]["case"]
    for field in PROVENANCE_FIELDS:
        if saved_case[field] != case[field]:
            raise ValueError(f"{case['case_id']}: saved mean-field provenance differs for {field}")
    if reference["reference_scf"] != case["reference_settings"]["reference_scf"]:
        raise ValueError("Saved mean-field method differs from the original reference")
    if reference["nelecas"] != case["nelecas"]:
        raise ValueError("Saved mean-field active occupations differ from the validated cache")
    cached_energy = case["cached_scf_energy_hartree"]
    if cached_energy is not None and abs(energy - cached_energy) > args.scf_match_tolerance:
        raise ValueError("Saved SCF energy differs from the cached mean-field reference")
    return reference, digest


def load_cases(args):
    cases = load_subset(args)
    for case in cases:
        case["mean_field_reference_path"] = str(FOCK_ROOT / case["case_id"] / "fock_reference.json")
        reference, digest = read_saved_reference(case, args)
        canonical_window(case, reference)
        # Include the source digest in the shared resume profile.
        case["mean_field_reference_sha256"] = digest
    return cases


def compute_mean_field_spectrum(case, case_dir, args):
    reference, digest = read_saved_reference(case, args)
    return {
        "spectrum_scope": SPECTRUM_SCOPE,
        "source_mean_field_reference_path": case["mean_field_reference_path"],
        "source_mean_field_reference_sha256": digest,
        "source_scf_energy_hartree": reference["scf_energy_hartree"],
        "source_scf_converged": reference["scf_converged"],
        "reference_scf": reference["reference_scf"],
        "active_orbital_estimate": canonical_window(case, reference),
        "new_reference_calculation": False,
    }


def width_from_reference(result):
    estimate = result["active_orbital_estimate"]
    sums = orbital_estimates(estimate["alpha_orbital_energies_hartree"],
                            estimate["beta_orbital_energies_hartree"],
                            estimate["active_spin_occupations"])
    width = sums["determinant_width_hartree"]
    return {
        "width_definition": "highest minus lowest frozen molecular HF orbital-energy sum at fixed active Nalpha/Nbeta",
        "width_reference_method": result["reference_scf"],
        "spectrum_scope": SPECTRUM_SCOPE,
        "fock_bandwidth_hartree": width,
        "mean_field_active_bandwidth_hartree": width,
        "active_spin_occupations": sums["active_spin_occupations"],
        "active_orbital_indices_zero_based": estimate["active_orbital_indices_zero_based"],
        "lowest_determinant_orbital_sum_hartree": sums["lowest_determinant_orbital_sum_hartree"],
        "highest_determinant_orbital_sum_hartree": sums["highest_determinant_orbital_sum_hartree"],
        "mean_field_active_estimate": {**estimate, **sums},
        "source_mean_field_reference_path": result["source_mean_field_reference_path"],
        "source_mean_field_reference_sha256": result["source_mean_field_reference_sha256"],
        "delta_t": math.pi / width,
        "delta_t_units": "atomic units of time",
        "delta_t_rule": "pi / W_MF; saved molecular HF active-window determinant orbital-energy sums",
    }


def parse_args():
    args = qrte.parse_args(default_outdir=DEFAULT_OUTDIR, description=__doc__, default_trotter_order=2)
    if args.delta_t is not None or args.evolution_mode != "trotter" or args.trotter_order != 2:
        raise SystemExit("This workflow requires dt=pi/W_MF, --evolution-mode trotter and --trotter-order 2")
    return args


def run(args):
    cases = load_cases(args)
    if args.count_cases:
        print(len(cases))
        return
    if args.dry_run:
        for case in cases:
            result = compute_mean_field_spectrum(case, None, args)
            width = width_from_reference(result)
            print(f"{CASE_IDS.index(case['case_id']):2d} {case['case_id']:30s} "
                  f"SCF={result['reference_scf']:4s} W_MF={width['fock_bandwidth_hartree']:.12g} Eh "
                  f"dt={width['delta_t']:.12g} au "
                  f"indices={width['active_orbital_indices_zero_based']}")
        print(f"Validated {len(cases)} saved references and Hamiltonian caches; "
              f"max_iterations={args.n_steps}; overlap_tol={args.overlap_tol:g}; "
              f"trotter_order={args.trotter_order}; trotter_steps={args.trotter_steps}; "
              f"early_stop=chemical_accuracy; outdir={args.outdir}")
        print("W is a canonical-HF-window estimate, not the optimized CASSCF/AVAS spectrum; "
              "ROHF uses effective Roothaan spatial energies. No molecular calculations were run.")
        return
    failures = 0
    for case in cases:
        try:
            qrte.run_case(case, args, spectrum_builder=compute_mean_field_spectrum,
                          width_builder=width_from_reference, spectrum_scope=SPECTRUM_SCOPE,
                          spectrum_filename=SPECTRUM_FILENAME)
        except Exception as exc:
            failures += 1
            print(f"FAIL {case['case_id']}: {exc!r}", flush=True)
    if failures:
        raise SystemExit(1)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
