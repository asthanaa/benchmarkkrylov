"""Run an orbital-optimization refinement workflow around a saved QP-CCIK state.

Default target:
- Cr2
- d = 2.0 Angstrom
- STO-3G
- full active space CAS(48e,36o)

Workflow:
1. Fresh EZFIO + SCF + MO classes
2. Run CCIK and save the final Ritz wavefunction
3. Read the internal PT2-like correction printed by CCIK
4. Run QP orbital optimization on that saved wavefunction
5. Evaluate the optimized saved wavefunction with a lightweight CCIK/PT2 pass
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from qp_ccik_common import (
    ROOT,
    CCIKParams,
    QPPaths,
    discover_qp_paths,
    estimate_thick_storage_slots,
    run_ccik,
    run_orb_opt,
    run_qp_setup,
    set_ccik_params,
)
from cr2_qp_ccik_tune import create_xyz, cr2_spatial_orbitals, make_cr2_atom


DEFAULT_OUTPUT = ROOT / "experiments" / "cr2_qp_ccik_refine"
DEFAULT_TARGET = -2064.503


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage_row(stage: str, *, ccik_total: float | None, ccik_elec: float | None, basis: int | None,
              pt2_total: float | None, pt2_corr: float | None, note: str = "") -> dict[str, object]:
    row: dict[str, object] = {"stage": stage, "note": note}
    row["ccik_total_eh"] = "" if ccik_total is None else f"{ccik_total:.12f}"
    row["ccik_electronic_eh"] = "" if ccik_elec is None else f"{ccik_elec:.12f}"
    row["ccik_basis"] = "" if basis is None else basis
    row["pt2_total_eh"] = "" if pt2_total is None else f"{pt2_total:.12f}"
    row["pt2_correction_eh"] = "" if pt2_corr is None else f"{pt2_corr:.12f}"
    return row


def _extract_optional(text: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)} =\s*([-+0-9.eE]+)", text)
    return None if m is None else float(m.group(1))


def run_stage_ccik(qp: QPPaths, ezfio: Path, stage_dir: Path) -> tuple[float, float, int, float | None, float | None]:
    e_elec, e_tot, basis, out = run_ccik(qp, ezfio)
    (stage_dir / "ccik.log").write_text(out, encoding="utf-8", errors="ignore")
    pt2_corr = _extract_optional(out, "CCIK_FINAL_PT2_CORRECTION")
    e_plus_pt2 = _extract_optional(out, "CCIK_FINAL_E_PLUS_PT2")
    return e_elec, e_tot, basis, e_plus_pt2, pt2_corr


def run_stage_orb_opt(qp: QPPaths, ezfio: Path, stage_dir: Path, *, method: str) -> None:
    out = run_orb_opt(qp, ezfio, optimization_method=method)
    (stage_dir / "orb_opt.log").write_text(out, encoding="utf-8", errors="ignore")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine Cr2 QP-CCIK with PT2 and orbital optimization")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--distance", type=float, default=2.0)
    p.add_argument("--ncas", type=int, default=36)
    p.add_argument("--nelecas", type=int, default=48)
    p.add_argument("--ncore", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--target-total-eh", type=float, default=DEFAULT_TARGET)
    p.add_argument("--orb-opt-method", choices=("diag", "full", "none"), default="diag")

    p.add_argument("--ccik-basis-size", type=int, default=8)
    p.add_argument("--ccik-generator-keep", type=int, default=128)
    p.add_argument("--ccik-generator-fraction", type=float, default=0.0)
    p.add_argument("--ccik-selector-keep", type=int, default=2048)
    p.add_argument("--ccik-external-keep", type=int, default=2048)
    p.add_argument("--ccik-pt2-rank-weight", type=float, default=1.0)
    p.add_argument("--ccik-nkeep", type=int, default=2048)
    p.add_argument("--ccik-total-det-budget", type=int, default=0)
    p.add_argument("--ccik-conv-tol", type=float, default=1e-6)
    p.add_argument("--ccik-ritz-overlap-tol", type=float, default=1e-8)
    p.add_argument("--ccik-ritz-max-drop", type=float, default=0.0)
    p.add_argument("--ccik-max-cycle", type=int, default=8)
    p.add_argument("--ccik-restart-m-cycle", type=int, default=4)
    p.add_argument("--ccik-restart-nroot", type=int, default=1)
    p.add_argument("--ccik-verbose", action="store_true")

    p.add_argument("--eval-ccik-basis-size", type=int, default=1)
    p.add_argument("--eval-ccik-generator-keep", type=int, default=4096)
    p.add_argument("--eval-ccik-generator-fraction", type=float, default=0.0)
    p.add_argument("--eval-ccik-selector-keep", type=int, default=4096)
    p.add_argument("--eval-ccik-external-keep", type=int, default=4096)
    p.add_argument("--eval-ccik-pt2-rank-weight", type=float, default=1.0)
    p.add_argument("--eval-ccik-nkeep", type=int, default=4096)
    p.add_argument("--eval-ccik-conv-tol", type=float, default=1e-6)
    p.add_argument("--eval-ccik-ritz-overlap-tol", type=float, default=1e-8)
    p.add_argument("--eval-ccik-ritz-max-drop", type=float, default=0.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    qp = discover_qp_paths()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    xyz = outdir / "cr2.xyz"
    create_xyz(xyz, make_cr2_atom(args.distance), comment="Cr2")
    nmo_total = cr2_spatial_orbitals(basis=args.basis, r_ang=args.distance)
    if args.ncore + args.ncas > nmo_total:
        raise RuntimeError(
            f"Invalid active space for basis {args.basis}: ncore+ncas={args.ncore + args.ncas}, only {nmo_total} orbitals available."
        )

    ezfio = outdir / "cr2.ezfio"
    if ezfio.exists():
        shutil.rmtree(ezfio)
    run_qp_setup(
        qp,
        basis=args.basis,
        ezfio=ezfio,
        xyz=xyz,
        ncore=args.ncore,
        ncas=args.ncas,
        nmo_total=nmo_total,
    )

    params = CCIKParams(
        ccik_basis_size=args.ccik_basis_size,
        ccik_generator_keep=args.ccik_generator_keep,
        ccik_generator_fraction=args.ccik_generator_fraction,
        ccik_selector_keep=args.ccik_selector_keep,
        ccik_external_keep=args.ccik_external_keep,
        ccik_pt2_rank_weight=args.ccik_pt2_rank_weight,
        ccik_nkeep=args.ccik_nkeep,
        ccik_total_det_budget=args.ccik_total_det_budget,
        ccik_conv_tol=args.ccik_conv_tol,
        ccik_ritz_overlap_tol=args.ccik_ritz_overlap_tol,
        ccik_ritz_max_drop=args.ccik_ritz_max_drop,
        ccik_save_final_wf=True,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=True,
        ccik_max_cycle=args.ccik_max_cycle,
        ccik_restart_m_cycle=args.ccik_restart_m_cycle,
        ccik_restart_nroot=args.ccik_restart_nroot,
        ccik_verbose=args.ccik_verbose,
    )
    eval_params = CCIKParams(
        ccik_basis_size=args.eval_ccik_basis_size,
        ccik_generator_keep=args.eval_ccik_generator_keep,
        ccik_generator_fraction=args.eval_ccik_generator_fraction,
        ccik_selector_keep=args.eval_ccik_selector_keep,
        ccik_external_keep=args.eval_ccik_external_keep,
        ccik_pt2_rank_weight=args.eval_ccik_pt2_rank_weight,
        ccik_nkeep=args.eval_ccik_nkeep,
        ccik_total_det_budget=0,
        ccik_conv_tol=args.eval_ccik_conv_tol,
        ccik_ritz_overlap_tol=args.eval_ccik_ritz_overlap_tol,
        ccik_ritz_max_drop=args.eval_ccik_ritz_max_drop,
        ccik_save_final_wf=False,
        ccik_use_existing_wf=True,
        ccik_do_post_pt2=True,
        ccik_max_cycle=1,
        ccik_restart_m_cycle=0,
        ccik_restart_nroot=2,
        ccik_verbose=args.ccik_verbose,
    )
    set_ccik_params(qp, ezfio, params)

    rows: list[dict[str, object]] = []
    stages = [
        ("stage1_ccik", "initial CCIK on RHF orbitals"),
        ("stage2_orbopt", f"orbital optimization method={args.orb_opt_method}"),
        ("stage3_eval", "lightweight CCIK/PT2 evaluation of the optimized saved wavefunction"),
    ]

    stage_dir = outdir / stages[0][0]
    stage_dir.mkdir(parents=True, exist_ok=True)
    e_elec, e_tot, basis, e_plus_pt2, pt2_corr = run_stage_ccik(qp, ezfio, stage_dir)
    rows.append(stage_row(stages[0][0], ccik_total=e_tot, ccik_elec=e_elec, basis=basis, pt2_total=e_plus_pt2, pt2_corr=pt2_corr, note=stages[0][1]))

    stage_dir = outdir / stages[1][0]
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_stage_orb_opt(qp, ezfio, stage_dir, method=args.orb_opt_method)
    rows.append(stage_row(stages[1][0], ccik_total=None, ccik_elec=None, basis=None, pt2_total=None, pt2_corr=None, note=stages[1][1]))

    stage_dir = outdir / stages[2][0]
    stage_dir.mkdir(parents=True, exist_ok=True)
    set_ccik_params(qp, ezfio, eval_params)
    e_elec2, e_tot2, basis2, e_plus_pt22, pt2_corr2 = run_stage_ccik(qp, ezfio, stage_dir)
    rows.append(stage_row(stages[2][0], ccik_total=e_tot2, ccik_elec=e_elec2, basis=basis2, pt2_total=e_plus_pt22, pt2_corr=pt2_corr2, note=stages[2][1]))

    results_csv = outdir / "results.csv"
    write_rows(results_csv, rows)

    summary = {
        "system": "Cr2",
        "distance_ang": args.distance,
        "basis": args.basis,
        "active_space": [args.ncas, args.nelecas],
        "target_total_eh": args.target_total_eh,
        "estimated_storage_slots": estimate_thick_storage_slots(params),
        "params": params.__dict__,
        "eval_params": eval_params.__dict__,
        "rows": rows,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")

    lines = [
        "Cr2 QP-CCIK refinement workflow",
        f"target_total_eh = {args.target_total_eh:.12f}",
        f"estimated_storage_slots = {estimate_thick_storage_slots(params)}",
        "",
    ]
    for row in rows:
        line = (
            f"{row['stage']}: ccik_total={row['ccik_total_eh']} pt2_total={row['pt2_total_eh']} "
            f"pt2_corr={row['pt2_correction_eh']} basis={row['ccik_basis']} note={row['note']}"
        )
        lines.append(line)
    (outdir / "NOTES.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
