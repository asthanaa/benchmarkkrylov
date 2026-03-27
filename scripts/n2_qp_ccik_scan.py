"""Run N2 scans with the Quantum Package ccik plugin and plot PES/error curves.

This driver supports:

- exact PySCF CASCI references for tractable active spaces
- a higher-budget QP-ccik benchmark reference when exact FCI is not practical
- direct execution of the compiled QP plugin binary
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf

from n2_utils import make_n2_atom, safe_log_err
from qp_ccik_common import (
    ROOT,
    CCIKParams,
    derived_thick_nkeep,
    discover_qp_paths,
    run_ccik,
    run_qp_setup,
    set_ccik_params,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "qp_ccik_n2_scan"


def n2_spatial_orbitals(*, basis: str) -> int:
    mol = gto.M(atom=make_n2_atom(1.1), basis=basis, unit="Angstrom", charge=0, spin=0, verbose=0)
    return int(mol.nao_nr())


def create_n2_xyz(path: Path, r_ang: float) -> None:
    path.write_text(
        f"2\nN2 {r_ang:.6f} A\nN 0 0 {-r_ang/2.0:.8f}\nN 0 0 {r_ang/2.0:.8f}\n",
        encoding="ascii",
    )


def exact_cas_reference(*, r_ang: float, basis: str, ncas: int, nelecas: int, ncore: int) -> float:
    mol = gto.M(atom=make_n2_atom(r_ang), basis=basis, unit="Angstrom", charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10, verbose=0)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore
    mc.fcisolver = fci.direct_spin1.FCI()
    mc.fcisolver.conv_tol = 1e-12
    mc.fcisolver.max_cycle = 200
    mc.fcisolver.max_space = 50
    e_tot, *_ = mc.kernel()
    return float(e_tot)


def make_params_from_args(args: argparse.Namespace, *, prefix: str = "") -> CCIKParams:
    return CCIKParams(
        ccik_basis_size=int(getattr(args, f"{prefix}ccik_basis_size")),
        ccik_generator_keep=int(getattr(args, f"{prefix}ccik_generator_keep")),
        ccik_generator_fraction=float(getattr(args, f"{prefix}ccik_generator_fraction")),
        ccik_selector_keep=int(getattr(args, f"{prefix}ccik_selector_keep")),
        ccik_external_keep=int(getattr(args, f"{prefix}ccik_external_keep")),
        ccik_pt2_rank_weight=float(getattr(args, f"{prefix}ccik_pt2_rank_weight", 1.0)),
        ccik_nkeep=int(getattr(args, f"{prefix}ccik_nkeep")),
        ccik_total_det_budget=int(getattr(args, f"{prefix}ccik_total_det_budget")),
        ccik_conv_tol=float(getattr(args, f"{prefix}ccik_conv_tol")),
        ccik_ritz_overlap_tol=float(getattr(args, f"{prefix}ccik_ritz_overlap_tol", 1e-8)),
        ccik_ritz_max_drop=float(getattr(args, f"{prefix}ccik_ritz_max_drop", 0.0)),
        ccik_save_final_wf=bool(getattr(args, f"{prefix}ccik_save_final_wf", True)),
        ccik_use_existing_wf=bool(getattr(args, f"{prefix}ccik_use_existing_wf", False)),
        ccik_do_post_pt2=bool(getattr(args, f"{prefix}ccik_do_post_pt2", False)),
        ccik_max_cycle=int(getattr(args, f"{prefix}ccik_max_cycle")),
        ccik_restart_m_cycle=int(getattr(args, f"{prefix}ccik_restart_m_cycle")),
        ccik_restart_nroot=int(getattr(args, f"{prefix}ccik_restart_nroot")),
        ccik_verbose=bool(getattr(args, f"{prefix}ccik_verbose")),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run N2 scans with the Quantum Package ccik backend")
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--Rmin", type=float, default=1.0)
    p.add_argument("--Rmax", type=float, default=2.5)
    p.add_argument("--npoints", type=int, default=5)
    p.add_argument("--ncas", type=int, required=True)
    p.add_argument("--nelecas", type=int, required=True)
    p.add_argument("--ncore", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--keep-ezfio", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument("--reference-mode", choices=("exact", "benchmark", "none"), default="exact")

    p.add_argument("--ccik-basis-size", type=int, default=16)
    p.add_argument("--ccik-generator-keep", type=int, default=512)
    p.add_argument("--ccik-generator-fraction", type=float, default=0.0)
    p.add_argument("--ccik-selector-keep", type=int, default=8192)
    p.add_argument("--ccik-external-keep", type=int, default=8192)
    p.add_argument("--ccik-pt2-rank-weight", type=float, default=1.0)
    p.add_argument("--ccik-nkeep", type=int, default=8192)
    p.add_argument("--ccik-total-det-budget", type=int, default=0)
    p.add_argument("--ccik-conv-tol", type=float, default=1e-6)
    p.add_argument("--ccik-ritz-overlap-tol", type=float, default=1e-8)
    p.add_argument("--ccik-ritz-max-drop", type=float, default=0.0)
    p.add_argument("--ccik-save-final-wf", action="store_true", default=True)
    p.add_argument("--ccik-do-post-pt2", action="store_true", default=False)
    p.add_argument("--ccik-max-cycle", type=int, default=1)
    p.add_argument("--ccik-restart-m-cycle", type=int, default=0)
    p.add_argument("--ccik-restart-nroot", type=int, default=2)
    p.add_argument("--ccik-verbose", action="store_true")

    p.add_argument("--ref-ccik-basis-size", type=int, default=32)
    p.add_argument("--ref-ccik-generator-keep", type=int, default=4096)
    p.add_argument("--ref-ccik-generator-fraction", type=float, default=0.0)
    p.add_argument("--ref-ccik-selector-keep", type=int, default=65536)
    p.add_argument("--ref-ccik-external-keep", type=int, default=65536)
    p.add_argument("--ref-ccik-pt2-rank-weight", type=float, default=1.0)
    p.add_argument("--ref-ccik-nkeep", type=int, default=65536)
    p.add_argument("--ref-ccik-total-det-budget", type=int, default=0)
    p.add_argument("--ref-ccik-conv-tol", type=float, default=1e-7)
    p.add_argument("--ref-ccik-ritz-overlap-tol", type=float, default=1e-8)
    p.add_argument("--ref-ccik-ritz-max-drop", type=float, default=5.0)
    p.add_argument("--ref-ccik-save-final-wf", action="store_true", default=True)
    p.add_argument("--ref-ccik-do-post-pt2", action="store_true", default=False)
    p.add_argument("--ref-ccik-max-cycle", type=int, default=1)
    p.add_argument("--ref-ccik-restart-m-cycle", type=int, default=0)
    p.add_argument("--ref-ccik-restart-nroot", type=int, default=2)
    p.add_argument("--ref-ccik-verbose", action="store_true")
    return p.parse_args(argv)


def write_results_csv(path: Path, rows: list[dict[str, object]], *, include_reference: bool) -> None:
    fieldnames = [
        "R_ang",
        "basis",
        "ncas",
        "nelecas",
        "ncore",
        "qp_ccik_electronic_eh",
        "qp_ccik_total_eh",
        "basis_size",
    ]
    if include_reference:
        fieldnames.extend(["reference_total_eh", "abs_err_eh", "reference_mode"])
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]], *, reference_mode: str, params: CCIKParams) -> None:
    err_vals = [float(row["abs_err_eh"]) for row in rows if "abs_err_eh" in row]
    derived_nkeep = None
    if params.ccik_total_det_budget > 0 and params.ccik_restart_m_cycle > 1 and params.ccik_max_cycle > 1:
        derived_nkeep = derived_thick_nkeep(params)
    lines = [
        "Quantum Package CCIK N2 scan",
        f"reference_mode = {reference_mode}",
        "",
        "main params:",
        f"  basis_size = {params.ccik_basis_size}",
        f"  generator_keep = {params.ccik_generator_keep}",
        f"  generator_fraction = {params.ccik_generator_fraction}",
        f"  selector_keep = {params.ccik_selector_keep}",
        f"  external_keep = {params.ccik_external_keep}",
        f"  nkeep = {params.ccik_nkeep}",
        f"  total_det_budget = {params.ccik_total_det_budget}",
        f"  conv_tol = {params.ccik_conv_tol}",
        f"  restart_m_cycle = {params.ccik_restart_m_cycle}",
        f"  restart_nroot = {params.ccik_restart_nroot}",
        f"  outer_cycles = {params.ccik_max_cycle}",
        "",
    ]
    if derived_nkeep is not None:
        lines.extend(
            [
                f"derived_thick_nkeep = {derived_nkeep}",
                "",
            ]
        )
    if err_vals:
        lines.extend(
            [
                f"max_abs_err_eh = {max(err_vals):.6e}",
                f"mean_abs_err_eh = {float(np.mean(err_vals)):.6e}",
                f"n_below_1e-3 = {sum(v < 1e-3 for v in err_vals)} / {len(err_vals)}",
                "",
            ]
        )
    for row in rows:
        if "abs_err_eh" in row:
            lines.append(
                f"R={row['R_ang']}  E={row['qp_ccik_total_eh']}  ref={row['reference_total_eh']}  "
                f"err={row['abs_err_eh']}  basis={row['basis_size']}"
            )
        else:
            lines.append(f"R={row['R_ang']}  E={row['qp_ccik_total_eh']}  basis={row['basis_size']}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def plot_results(out_prefix: Path, rows: list[dict[str, object]], *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Matplotlib is required for plotting.") from exc

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "legend.frameon": False,
        }
    )

    R = np.asarray([float(row["R_ang"]) for row in rows], dtype=float)
    e_alg = np.asarray([float(row["qp_ccik_total_eh"]) for row in rows], dtype=float)
    has_reference = "reference_total_eh" in rows[0]

    if has_reference:
        e_ref = np.asarray([float(row["reference_total_eh"]) for row in rows], dtype=float)
        err = np.asarray([float(row["abs_err_eh"]) for row in rows], dtype=float)
        fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
        ax[0].plot(R, e_ref, "ko-", label="Reference")
        ax[0].plot(R, e_alg, "C0o-", label="QP-CCIK")
        ax[0].set_xlabel("R (Angstrom)")
        ax[0].set_ylabel("Energy (Eh)")
        ax[0].set_title(title)
        ax[0].legend()
        ax[1].plot(R, safe_log_err(err), "C1o-")
        ax[1].axhline(1e-3, color="0.5", ls="--", lw=1.0)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("R (Angstrom)")
        ax[1].set_ylabel("|Delta E| (Eh)")
        ax[1].set_title("Absolute Error")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.8), constrained_layout=True)
        ax.plot(R, e_alg, "C0o-", label="QP-CCIK")
        ax.set_xlabel("R (Angstrom)")
        ax.set_ylabel("Energy (Eh)")
        ax.set_title(title)
        ax.legend()

    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".pdf"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    qp = discover_qp_paths()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    nmo_total = n2_spatial_orbitals(basis=args.basis)
    if args.ncore + args.ncas > nmo_total:
        raise ValueError(
            f"Invalid active space for basis {args.basis}: ncore+ncas={args.ncore + args.ncas}, only {nmo_total} orbitals available."
        )

    points = np.linspace(float(args.Rmin), float(args.Rmax), int(args.npoints))
    params = make_params_from_args(args)
    ref_params = make_params_from_args(args, prefix="ref_")
    rows: list[dict[str, object]] = []

    for r_ang in points:
        tag = f"{r_ang:.2f}".replace(".", "p")
        xyz = out_dir / f"n2_{tag}.xyz"
        ezfio = out_dir / f"n2_{tag}.ezfio"
        if ezfio.exists():
            shutil.rmtree(ezfio)
        create_n2_xyz(xyz, float(r_ang))
        run_qp_setup(
            qp,
            basis=args.basis,
            ezfio=ezfio,
            xyz=xyz,
            ncore=args.ncore,
            ncas=args.ncas,
            nmo_total=nmo_total,
        )

        ref_total: float | None = None
        if args.reference_mode == "exact":
            ref_total = exact_cas_reference(
                r_ang=float(r_ang),
                basis=args.basis,
                ncas=args.ncas,
                nelecas=args.nelecas,
                ncore=args.ncore,
            )
        elif args.reference_mode == "benchmark":
            set_ccik_params(qp, ezfio, ref_params)
            _, ref_total, _, _ = run_ccik(qp, ezfio)

        set_ccik_params(qp, ezfio, params)
        e_elec, e_tot, basis_size, _ = run_ccik(qp, ezfio)
        row: dict[str, object] = {
            "R_ang": f"{float(r_ang):.2f}",
            "basis": args.basis,
            "ncas": args.ncas,
            "nelecas": args.nelecas,
            "ncore": args.ncore,
            "qp_ccik_electronic_eh": f"{e_elec:.12f}",
            "qp_ccik_total_eh": f"{e_tot:.12f}",
            "basis_size": basis_size,
        }
        if ref_total is not None:
            row["reference_total_eh"] = f"{ref_total:.12f}"
            row["abs_err_eh"] = f"{abs(e_tot - ref_total):.6e}"
            row["reference_mode"] = args.reference_mode
            print(
                f"R={float(r_ang):.2f}  ref={ref_total:.12f}  qp={e_tot:.12f}  "
                f"err={abs(e_tot - ref_total):.6e}  basis={basis_size}",
                flush=True,
            )
        else:
            print(f"R={float(r_ang):.2f}  qp={e_tot:.12f}  basis={basis_size}", flush=True)
        rows.append(row)

        if not args.keep_ezfio:
            shutil.rmtree(ezfio)

    csv_path = out_dir / "results.csv"
    write_results_csv(csv_path, rows, include_reference=("reference_total_eh" in rows[0]))
    write_summary(out_dir / "summary.txt", rows, reference_mode=args.reference_mode, params=params)

    if not args.skip_plot:
        title = f"N2 {args.basis} CAS({args.nelecas},{args.ncas})"
        plot_results(out_dir / "n2_qp_ccik_pes_and_error", rows, title=title)

    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
