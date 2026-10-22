#!/usr/bin/env python
"""Second-order QRTE for the remaining 17 validated MolVQE systems.

Uses the same saved molecular HF active-window determinant width as small10:
dt=pi/W_MF, fixed active Nalpha/Nbeta, and the original cached QRTE reference.
No new SCF, CASSCF, or interacting CAS diagonalization is performed. Missing
or invalid saved spectra prevent execution; --dry-run reports every case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import run_qrte_mean_field_active_pi_over_w_small10 as mean_field

qrte = mean_field.qrte
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_pi_over_w_remaining17_trotter2"
# Fixed array order: the validated cache manifest's order with small10 removed.
CASE_IDS = (
    "magnesium_porphyrin_ceo",
    "cyclobutadiene_ceo",
    "CH2O_equi_cc_pVTZ_2",
    "CH2O_equi_cc_pVTZ_0",
    "H6_near_equi_cc_pVTZ_2",
    "m_Benzyne_equi_cc_pVTZ_2",
    "hexatriene_equi_cc_pVTZ_2",
    "m_Benzyne_equi_cc_pVTZ_0",
    "FeO",
    "hexatriene_equi_cc_pVTZ_0",
    "H7_near_equi_sto3g_1",
    "fenoco3_minus",
    "feno_2plus",
    "H8_near_equi_sto3g_0",
    "Al3_minus",
    "H9_near_equi_sto3g_1",
    "H10_near_equi_sto3g_0",
)


def inspect_case(case, args):
    """Validate one saved spectrum and compute only orbital-energy sums."""
    path = Path(case["mean_field_reference_path"])
    if not path.is_file():
        return {"case_id": case["case_id"], "status": "missing_saved_reference",
                "error": f"Missing saved converged molecular HF spectrum: {path}"}
    try:
        reference, digest = mean_field.read_saved_reference(case, args)
        mean_field.canonical_window(case, reference)
        case["mean_field_reference_sha256"] = digest
        spectrum = mean_field.compute_mean_field_spectrum(case, None, args)
        width = mean_field.width_from_reference(spectrum)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"case_id": case["case_id"], "status": "invalid_saved_reference",
                "error": str(exc)}
    return {"case_id": case["case_id"], "status": "ready", **width}


def load_cases(args, *, validate_saved=True):
    selection = argparse.Namespace(**vars(args))
    selection.case_id = None
    selection.case_index = None
    by_id = {case["case_id"]: case for case in qrte.load_cases(selection)}
    expected = set(CASE_IDS) | set(mean_field.CASE_IDS)
    if set(by_id) != expected:
        raise ValueError("Validated cache manifest must contain the original 27 studied systems; "
                         f"missing={sorted(expected - set(by_id))}; extra={sorted(set(by_id) - expected)}")
    cases = [by_id[case_id] for case_id in CASE_IDS]
    if args.case_id is not None:
        requested = set(args.case_id)
        missing = requested - set(CASE_IDS)
        if missing:
            raise ValueError(f"Cases outside the remaining 17-system subset: {sorted(missing)}")
        cases = [case for case in cases if case["case_id"] in requested]
    if args.case_index is not None:
        if not 0 <= args.case_index < len(cases):
            raise ValueError(f"--case-index must be in [0, {len(cases) - 1}]")
        cases = [cases[args.case_index]]
    for case in cases:
        case["mean_field_reference_path"] = str(
            mean_field.FOCK_ROOT / case["case_id"] / "fock_reference.json")
    if validate_saved:
        results = [inspect_case(case, args) for case in cases]
        errors = [f"{result['case_id']}: {result['error']}"
                  for result in results if result["status"] != "ready"]
        if errors:
            raise ValueError("Saved mean-field input validation failed:\n" + "\n".join(errors))
    return cases


def parse_args():
    args = qrte.parse_args(default_outdir=DEFAULT_OUTDIR, description=__doc__, default_trotter_order=2)
    if args.delta_t is not None or args.evolution_mode != "trotter" or args.trotter_order != 2:
        raise SystemExit("This workflow requires dt=pi/W_MF, --evolution-mode trotter and --trotter-order 2")
    return args


def main():
    args = parse_args()
    try:
        cases = load_cases(args, validate_saved=not args.dry_run)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.count_cases:
        print(len(cases))
        return
    if args.dry_run:
        results = [inspect_case(case, args) for case in cases]
        for case, result in zip(cases, results):
            prefix = f"{CASE_IDS.index(case['case_id']):2d} {case['case_id']:30s}"
            if result["status"] != "ready":
                print(f"{prefix} {result['status']}: {result['error']}")
            else:
                print(f"{prefix} SCF={case['reference_settings']['reference_scf']:14s} "
                      f"W_MF={result['mean_field_active_bandwidth_hartree']:.12g} Eh "
                      f"dt={result['delta_t']:.12g} au "
                      f"indices={result['active_orbital_indices_zero_based']}")
        ready = sum(result["status"] == "ready" for result in results)
        print(f"Validated {len(cases)} Hamiltonian caches; saved HF inputs ready={ready}/{len(cases)}; "
              f"max_iterations={args.n_steps}; overlap_tol={args.overlap_tol:g}; "
              f"trotter_order=2; trotter_steps={args.trotter_steps}; "
              f"early_stop=chemical_accuracy; outdir={args.outdir}")
        print("W is a canonical-HF-window estimate; ROHF/sfX2C1eROHF use effective "
              "Roothaan spatial energies for both spins. No molecular calculations were run.")
        if ready != len(cases):
            raise SystemExit(1)
        return
    failures = 0
    for case in cases:
        try:
            qrte.run_case(case, args, spectrum_builder=mean_field.compute_mean_field_spectrum,
                          width_builder=mean_field.width_from_reference,
                          spectrum_scope=mean_field.SPECTRUM_SCOPE,
                          spectrum_filename=mean_field.SPECTRUM_FILENAME)
        except Exception as exc:
            failures += 1
            print(f"FAIL {case['case_id']}: {exc!r}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
