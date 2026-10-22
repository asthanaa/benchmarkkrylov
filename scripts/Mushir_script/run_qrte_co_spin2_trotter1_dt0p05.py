#!/usr/bin/env python
"""Rerun CO spin 2 at dt=0.05 with first-order, one-slice QRTE.

Pin the validated CASSCF/natorb Hamiltonian and stop against its own cached
reference, not the differing MolVQE dataset energy. Reuse the saved exact CAS
spectrum for the reference check; no SCF, CASSCF, or diagonalization is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import run_qrte_casscf_natorb_pi_over_w_small10 as cas

qrte = cas.qrte
CASE_ID = "CO_equi_cc_pVTZ_2"
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_co_spin2_trotter1_dt0p05_correct_reference"
SAVED_SPECTRUM = (qrte.ROOT / "runs/qrte_casscf_natorb_pi_over_w_small10_exact"
                  / CASE_ID / "active_space_reference.json")


def read_saved_spectrum(case):
    data = SAVED_SPECTRUM.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if case.get("saved_cas_spectrum_sha256", digest) != digest:
        raise ValueError("Saved CAS spectrum changed after validation")
    spectrum = json.loads(data)
    if (spectrum.get("status") != "completed"
            or spectrum.get("spectrum_scope") != cas.SPECTRUM_SCOPE
            or spectrum.get("hamiltonian_sha256") != case["hamiltonian_sha256"]
            or spectrum.get("active_orbitals") != case["active_orbitals"]
            or spectrum.get("active_spin_occupations") != case["nelecas"]):
        raise ValueError("Saved CAS spectrum does not match the validated CO Hamiltonian")
    ground = spectrum["sector_ground_energy_hartree"]
    if not math.isfinite(ground) or abs(ground - case["local_reference_energy_hartree"]) > 1e-6:
        raise ValueError("Saved exact-sector ground energy differs from the cached reference")
    cas.width_from_reference(spectrum)
    return spectrum, digest


def load_cases(args):
    if args.case_id is not None and set(args.case_id) != {CASE_ID}:
        raise ValueError(f"This workflow selects only {CASE_ID}")
    if args.case_index not in (None, 0):
        raise ValueError("This single-case workflow only accepts --case-index 0")
    selection = argparse.Namespace(**vars(args))
    selection.case_id, selection.case_index = [CASE_ID], None
    cases = qrte.load_cases(selection)
    case = cases[0]
    raw = qrte.load_json(qrte.resolve_path(case["source_json_path"]))
    if raw["meta"]["casscf"]["energy"] != case["dataset_casscf_energy_hartree"]:
        raise ValueError("Dataset JSON and manifest reference energies differ")
    spectrum, digest = read_saved_spectrum(case)
    case["saved_cas_spectrum_path"] = str(SAVED_SPECTRUM)
    case["saved_cas_spectrum_sha256"] = digest
    case["saved_exact_sector_ground_energy_hartree"] = spectrum["sector_ground_energy_hartree"]
    return cases


def saved_spectrum(case, case_dir, args):
    spectrum, digest = read_saved_spectrum(case)
    return {**spectrum, "source_saved_spectrum_path": str(SAVED_SPECTRUM),
            "source_saved_spectrum_sha256": digest, "new_reference_calculation": False}


def parse_args():
    args = qrte.parse_args(default_outdir=DEFAULT_OUTDIR, description=__doc__)
    if args.delta_t is None:
        args.delta_t = 0.05
    if (args.delta_t != 0.05 or args.evolution_mode != "trotter"
            or args.trotter_order != 1 or args.trotter_steps != 1):
        raise SystemExit("This workflow requires dt=0.05, Trotter order 1, and one slice")
    return args


def main():
    args = parse_args()
    cases = load_cases(args)
    if args.count_cases:
        print(len(cases))
        return
    case = cases[0]
    if args.dry_run:
        cached, dataset = case["local_reference_energy_hartree"], case["dataset_casscf_energy_hartree"]
        print(f"Validated {CASE_ID}: dt=0.05; trotter_order=1; trotter_steps=1; "
              f"max_iterations={args.n_steps}; overlap_tol={args.overlap_tol:g}")
        print(f"Cached reference = {cached:.14f} Eh; dataset CASSCF = {dataset:.14f} Eh; "
              f"dataset minus cache = {1000 * (dataset - cached):.8f} mHa")
        print(f"Early-stop threshold = {cached + qrte.CHEMICAL_ACCURACY:.14f} Eh; "
              "hit requires absolute error <= 1.6 mHa against the cached reference")
        print(f"Saved exact-sector ground energy = {case['saved_exact_sector_ground_energy_hartree']:.14f} Eh")
        print(f"Output: {args.outdir}; no molecular calculations or jobs launched")
        return
    qrte.run_case(case, args, spectrum_builder=saved_spectrum,
                  width_builder=cas.width_from_reference, spectrum_scope=cas.SPECTRUM_SCOPE,
                  spectrum_filename=cas.SPECTRUM_FILENAME)


if __name__ == "__main__":
    main()
