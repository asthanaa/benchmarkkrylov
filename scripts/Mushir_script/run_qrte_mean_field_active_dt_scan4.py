#!/usr/bin/env python
"""Scan dt=factor*pi/W_MF for N2 spins 0/2, Al3-minus, and H10/STO-3G.

Factors: 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2. Order 2, one Trotter slice.
--case-index selects a flat scan task (0..27), ordered by system then factor.
Uses saved HF spectra and validated CASSCF/natorb Hamiltonians and references.
No new SCF, CASSCF, or CAS diagonalization is performed.
"""

from __future__ import annotations

import argparse
import copy

import run_qrte_mean_field_active_pi_over_w_small10 as mf

qrte = mf.qrte
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_dt_scan4_trotter2"
CASE_IDS = ("N2_equi_cc_pVTZ_0", "N2_equi_cc_pVTZ_2", "Al3_minus", "H10_near_equi_sto3g_0")
FACTORS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SPECTRUM_SCOPE = "scaled_saved_molecular_hf_active_window_fixed_nalpha_nbeta"


def factor_label(factor):
    return "factor_" + format(factor, "g").replace(".", "p")


def parse_args(argv=None, *, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="trotter"):
    order = 2 if evolution_mode == "trotter" else 1
    args = qrte.parse_args(default_outdir=default_outdir, description=description,
                           default_trotter_order=order, default_evolution_mode=evolution_mode, argv=argv)
    if (args.delta_t is not None or args.evolution_mode != evolution_mode
            or args.trotter_order != order or args.trotter_steps != 1):
        detail = "Trotter order 2, and one slice" if evolution_mode == "trotter" else (
            "sparse exact exponential evolution; unused Trotter API parameters must retain defaults (1, 1)")
        raise SystemExit("This scan requires dt=factor*pi/W_MF, " + detail)
    args.outdir = qrte.resolve_path(args.outdir).resolve()
    return args


def load_tasks(args, *, case_ids=CASE_IDS, factors=FACTORS):
    selection = argparse.Namespace(**vars(args))
    selection.case_id, selection.case_index = list(case_ids), None
    by_id = {case["case_id"]: case for case in qrte.load_cases(selection)}
    if args.case_id is not None and set(args.case_id) - set(case_ids):
        raise ValueError("Cases outside the configured scan: " + str(sorted(set(args.case_id) - set(case_ids))))
    tasks = []
    for cid in case_ids:
        case = by_id[cid]
        case["mean_field_reference_path"] = str(mf.FOCK_ROOT / cid / "fock_reference.json")
        reference, digest = mf.read_saved_reference(case, args)
        mf.canonical_window(case, reference)
        case["mean_field_reference_sha256"] = digest
        width = mf.width_from_reference(mf.compute_mean_field_spectrum(case, None, args))
        for factor in factors:
            task_case = {**case, "dt_scale_factor": factor,
                         "unscaled_mean_field_delta_t": width["delta_t"]}
            tasks.append({"task_index": len(tasks), "case": task_case, "factor": factor,
                          "width": width["mean_field_active_bandwidth_hartree"],
                          "base_delta_t": width["delta_t"], "delta_t": factor * width["delta_t"],
                          "evolution_mode": args.evolution_mode, "trotter_order": args.trotter_order,
                          "trotter_steps": args.trotter_steps,
                          "outdir": args.outdir / factor_label(factor)})
    # Filtering preserves the task IDs of the complete configured grid.
    if args.case_id is not None:
        tasks = [task for task in tasks if task["case"]["case_id"] in args.case_id]
    if args.case_index is not None:
        tasks = [task for task in tasks if task["task_index"] == args.case_index]
        if not tasks:
            raise ValueError(f"--case-index must be a scan task index in [0, {len(case_ids) * len(factors) - 1}]")
    return tasks


def compute_scan_spectrum(case, case_dir, args):
    return {**mf.compute_mean_field_spectrum(case, case_dir, args),
            "spectrum_scope": SPECTRUM_SCOPE, "dt_scale_factor": case["dt_scale_factor"]}


def scan_width(reference):
    width = mf.width_from_reference(reference)
    return {**width, "unscaled_mean_field_delta_t": width["delta_t"],
            "dt_scale_factor": reference["dt_scale_factor"],
            "delta_t": reference["dt_scale_factor"] * width["delta_t"],
            "delta_t_rule": "dt_scale_factor * pi/W_MF; saved HF active-window determinant sums",
            "spectrum_scope": SPECTRUM_SCOPE}


def run_task(task, args):
    options = copy.copy(args)
    options.outdir = task["outdir"]
    qrte.run_case(task["case"], options, spectrum_builder=compute_scan_spectrum,
                  width_builder=scan_width, spectrum_scope=SPECTRUM_SCOPE,
                  spectrum_filename=mf.SPECTRUM_FILENAME)


def main(*, case_ids=CASE_IDS, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="trotter",
         factors=FACTORS, task_loader=None):
    args = parse_args(default_outdir=default_outdir, description=description, evolution_mode=evolution_mode)
    tasks = task_loader(args) if task_loader is not None else load_tasks(args, case_ids=case_ids, factors=factors)
    if args.count_cases:
        print(len(tasks))
        return
    if args.dry_run:
        for task in tasks:
            print(f"{task['task_index']:2d} {task['case']['case_id']:28s} "
                  f"factor={task['factor']:4g} W_MF={task['width']:.12g} Eh "
                  f"dt0={task['base_delta_t']:.12g} dt={task['delta_t']:.12g} au")
        propagation = "order=2; slices=1" if evolution_mode == "trotter" else "exact exponential; sparse expm_multiply; no Trotterization"
        print(f"Validated {len(tasks)} scan tasks; {propagation}; max_iterations={args.n_steps}; "
              f"overlap_tol={args.overlap_tol:g}; chemical_accuracy=1.6 mHa against cached references")
        print(f"Output: {args.outdir}/factor_<scale>/<case_id>/; no calculations or jobs launched")
        return
    failures = 0
    for task in tasks:
        try:
            run_task(task, args)
        except Exception as exc:
            failures += 1
            print(f"FAIL task={task['task_index']} case={task['case']['case_id']} "
                  f"factor={task['factor']}: {exc!r}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
