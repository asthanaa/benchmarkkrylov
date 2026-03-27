"""Tune a Quantum Package CCIK single-point calculation for Cr2.

Target system by default:
- Cr2
- d = 2.0 Angstrom
- STO-3G
- full active space CAS(48e,36o)

This script does not rely on an exact reference. Instead it records the best
variational total energy found under a determinant-storage ceiling.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from pyscf import gto

from qp_ccik_common import (
    ROOT,
    CCIKParams,
    bash,
    derived_thick_nkeep,
    discover_qp_paths,
    estimate_thick_storage_slots,
    run_qp_setup,
    set_ccik_params,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "cr2_qp_ccik_sto3g_d2p0"
DEFAULT_TARGET = -2064.503


def make_cr2_atom(r_ang: float) -> str:
    z = float(r_ang) / 2.0
    return f"""
    Cr 0.0 0.0 {-z}
    Cr 0.0 0.0 {+z}
    """


def create_xyz(path: Path, atom: str, *, comment: str) -> None:
    lines = [line.strip() for line in atom.splitlines() if line.strip()]
    path.write_text(
        f"{len(lines)}\n{comment}\n" + "\n".join(lines) + "\n",
        encoding="ascii",
    )


def cr2_spatial_orbitals(*, basis: str, r_ang: float) -> int:
    mol = gto.M(atom=make_cr2_atom(r_ang), basis=basis, unit="Angstrom", charge=0, spin=0, verbose=0)
    return int(mol.nao_nr())


def parse_ccik_output(text: str) -> dict[str, object]:
    import re

    out: dict[str, object] = {
        "final_total_eh": "",
        "final_electronic_eh": "",
        "final_basis": "",
    }
    m = re.search(r"CCIK_FINAL_TOTAL_ENERGY =\s*([-+0-9.Ee]+)", text)
    if m:
        out["final_total_eh"] = f"{float(m.group(1)):.12f}"
    m = re.search(r"CCIK_FINAL_ELECTRONIC_ENERGY =\s*([-+0-9.Ee]+)", text)
    if m:
        out["final_electronic_eh"] = f"{float(m.group(1)):.12f}"
    m = re.search(r"CCIK_FINAL_BASIS =\s*(\d+)", text)
    if m:
        out["final_basis"] = int(m.group(1))
    return out


def run_trial(
    *,
    qp_binary: Path,
    template_ezfio: Path,
    trial_dir: Path,
    params: CCIKParams,
    timeout_s: int,
    target_energy: float,
) -> dict[str, object]:
    ezfio = trial_dir / template_ezfio.name
    if ezfio.exists():
        shutil.rmtree(ezfio)
    shutil.copytree(template_ezfio, ezfio)

    qp = discover_qp_paths()
    set_ccik_params(qp, ezfio, params)
    (trial_dir / "params.json").write_text(json.dumps(params.__dict__, indent=2) + "\n", encoding="ascii")

    log_path = trial_dir / "run.log"
    start = time.time()
    status = "finished"
    try:
        cmd = (
            f"source {qp.quantum_package_rc} && "
            f"qp set_file {ezfio} >/dev/null && "
            f"{qp_binary} {ezfio}"
        )
        proc = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            env=os.environ.copy(),
        )
        log_text = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            status = f"exit_{proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_text = stdout + ("\n" + stderr if stderr else "")
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")
    wall_s = time.time() - start

    row: dict[str, object] = {
        "status": status,
        "wall_s": f"{wall_s:.2f}",
        "ccik_basis_size": params.ccik_basis_size,
        "ccik_generator_keep": params.ccik_generator_keep,
        "ccik_generator_fraction": params.ccik_generator_fraction,
        "ccik_selector_keep": params.ccik_selector_keep,
        "ccik_external_keep": params.ccik_external_keep,
        "ccik_nkeep": params.ccik_nkeep,
        "ccik_total_det_budget": params.ccik_total_det_budget,
        "ccik_conv_tol": params.ccik_conv_tol,
        "ccik_max_cycle": params.ccik_max_cycle,
        "ccik_restart_m_cycle": params.ccik_restart_m_cycle,
        "ccik_restart_nroot": params.ccik_restart_nroot,
        "derived_nkeep": derived_thick_nkeep(params) or "",
        "estimated_storage_slots": estimate_thick_storage_slots(params) or "",
        "target_total_eh": f"{target_energy:.12f}",
    }
    row.update(parse_ccik_output(log_text))
    if row["final_total_eh"]:
        row["delta_vs_target_eh"] = f"{float(row['final_total_eh']) - target_energy:.6e}"
        row["better_than_target"] = int(float(row["final_total_eh"]) < target_energy)
    else:
        row["delta_vs_target_eh"] = ""
        row["better_than_target"] = 0
    return row


def build_trials(args: argparse.Namespace) -> list[CCIKParams]:
    trials: list[CCIKParams] = []
    for basis_size, gen_keep, gen_frac, selector_keep, ext_keep, nkeep, conv_tol, max_cycle, m_cycle, nroot in itertools.product(
        args.ccik_basis_sizes,
        args.ccik_generator_keeps,
        args.ccik_generator_fractions,
        args.ccik_selector_keeps,
        args.ccik_external_keeps,
        args.ccik_nkeeps,
        args.ccik_conv_tols,
        args.ccik_max_cycles,
        args.ccik_restart_m_cycles,
        args.ccik_restart_nroots,
    ):
        params = CCIKParams(
            ccik_basis_size=int(basis_size),
            ccik_generator_keep=int(gen_keep),
            ccik_generator_fraction=float(gen_frac),
            ccik_selector_keep=int(selector_keep),
            ccik_external_keep=int(ext_keep),
            ccik_pt2_rank_weight=1.0,
            ccik_nkeep=int(nkeep),
            ccik_total_det_budget=int(args.ccik_total_det_budget),
            ccik_conv_tol=float(conv_tol),
            ccik_ritz_overlap_tol=1e-8,
            ccik_ritz_max_drop=0.0,
            ccik_save_final_wf=True,
            ccik_use_existing_wf=False,
            ccik_do_post_pt2=False,
            ccik_max_cycle=int(max_cycle),
            ccik_restart_m_cycle=int(m_cycle),
            ccik_restart_nroot=int(nroot),
            ccik_verbose=bool(args.ccik_verbose),
        )
        slots = estimate_thick_storage_slots(params)
        if slots is not None and slots > args.storage_slot_limit:
            continue
        if m_cycle > 1 and nroot >= m_cycle:
            continue
        trials.append(params)

    def sort_key(p: CCIKParams) -> tuple[float, int, int, int, int, int]:
        slots = estimate_thick_storage_slots(p) or 0
        return (
            float(slots),
            p.ccik_selector_keep,
            p.ccik_external_keep,
            p.ccik_nkeep,
            p.ccik_restart_m_cycle,
            p.ccik_restart_nroot,
        )

    return sorted(trials, key=sort_key)


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]], *, args: argparse.Namespace, nmo_total: int) -> None:
    best_finished = [row for row in rows if row.get("status") == "finished" and row.get("final_total_eh")]
    best_line = "none"
    if best_finished:
        best = min(best_finished, key=lambda row: float(row["final_total_eh"]))
        best_line = (
            f"best_total_eh = {best['final_total_eh']}  delta_vs_target_eh = {best['delta_vs_target_eh']}  "
            f"selector_keep = {best['ccik_selector_keep']}  external_keep = {best['ccik_external_keep']}  "
            f"nkeep = {best['ccik_nkeep']}  m_cycle = {best['ccik_restart_m_cycle']}  "
            f"nroot = {best['ccik_restart_nroot']}  max_cycle = {best['ccik_max_cycle']}"
        )
    lines = [
        "Cr2 QP-CCIK single-point tuning",
        f"geometry = Cr2 at {args.distance:.6f} Angstrom",
        f"basis = {args.basis}",
        f"nmo_total = {nmo_total}",
        f"active_space = ({args.ncas}o,{args.nelecas}e)",
        f"ncore = {args.ncore}",
        f"storage_slot_limit = {args.storage_slot_limit}",
        f"ccik_total_det_budget = {args.ccik_total_det_budget}",
        f"target_total_eh = {args.target_total_eh:.12f}",
        f"n_trials = {len(rows)}",
        best_line,
        "",
    ]
    for row in rows:
        lines.append(
            f"status={row['status']} E={row.get('final_total_eh','')} dE={row.get('delta_vs_target_eh','')} "
            f"sel={row['ccik_selector_keep']} ext={row['ccik_external_keep']} nkeep={row['ccik_nkeep']} "
            f"gen={row['ccik_generator_keep']} genfrac={row['ccik_generator_fraction']} "
            f"mcycle={row['ccik_restart_m_cycle']} nroot={row['ccik_restart_nroot']} "
            f"cycles={row['ccik_max_cycle']} slots={row.get('estimated_storage_slots','')} wall={row['wall_s']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune a Cr2 QP-CCIK single-point under a determinant budget")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--distance", type=float, default=2.0)
    p.add_argument("--ncas", type=int, default=36)
    p.add_argument("--nelecas", type=int, default=48)
    p.add_argument("--ncore", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--target-total-eh", type=float, default=DEFAULT_TARGET)
    p.add_argument("--storage-slot-limit", type=int, default=10_000_000)
    p.add_argument("--ccik-total-det-budget", type=int, default=0)
    p.add_argument("--keep-template-ezfio", action="store_true")
    p.add_argument("--ccik-verbose", action="store_true")

    p.add_argument("--ccik-basis-sizes", type=int, nargs="+", default=[16])
    p.add_argument("--ccik-generator-keeps", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--ccik-generator-fractions", type=float, nargs="+", default=[0.0])
    p.add_argument("--ccik-selector-keeps", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    p.add_argument("--ccik-external-keeps", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    p.add_argument("--ccik-nkeeps", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    p.add_argument("--ccik-conv-tols", type=float, nargs="+", default=[1e-6])
    p.add_argument("--ccik-max-cycles", type=int, nargs="+", default=[6, 10])
    p.add_argument("--ccik-restart-m-cycles", type=int, nargs="+", default=[8, 10, 12])
    p.add_argument("--ccik-restart-nroots", type=int, nargs="+", default=[2, 3])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    qp = discover_qp_paths()
    nmo_total = cr2_spatial_orbitals(basis=args.basis, r_ang=args.distance)
    if args.ncore + args.ncas > nmo_total:
        raise ValueError(
            f"Invalid active space for basis {args.basis}: ncore+ncas={args.ncore + args.ncas}, only {nmo_total} orbitals available."
        )

    atom = make_cr2_atom(args.distance)
    xyz = out_dir / "cr2.xyz"
    template_ezfio = out_dir / "cr2_template.ezfio"
    if template_ezfio.exists():
        shutil.rmtree(template_ezfio)
    create_xyz(xyz, atom, comment=f"Cr2 d={args.distance:.6f} A")
    run_qp_setup(
        qp,
        basis=args.basis,
        ezfio=template_ezfio,
        xyz=xyz,
        ncore=args.ncore,
        ncas=args.ncas,
        nmo_total=nmo_total,
    )

    trials = build_trials(args)
    rows: list[dict[str, object]] = []
    for idx, params in enumerate(trials, start=1):
        trial_name = (
            f"t{idx:03d}_sel{params.ccik_selector_keep}_ext{params.ccik_external_keep}_"
            f"nk{params.ccik_nkeep}_gen{params.ccik_generator_keep}_"
            f"mc{params.ccik_restart_m_cycle}_nr{params.ccik_restart_nroot}_cy{params.ccik_max_cycle}"
        )
        trial_dir = out_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        row = run_trial(
            qp_binary=qp.ccik_binary,
            template_ezfio=template_ezfio,
            trial_dir=trial_dir,
            params=params,
            timeout_s=args.timeout_s,
            target_energy=args.target_total_eh,
        )
        rows.append(row)
        if row["final_total_eh"]:
            print(
                f"[{idx}/{len(trials)}] {row['status']} E={row['final_total_eh']} dE={row['delta_vs_target_eh']} "
                f"sel={params.ccik_selector_keep} ext={params.ccik_external_keep} nk={params.ccik_nkeep} "
                f"gen={params.ccik_generator_keep} mc={params.ccik_restart_m_cycle} nr={params.ccik_restart_nroot} "
                f"cy={params.ccik_max_cycle} slots={row['estimated_storage_slots']} wall={row['wall_s']}",
                flush=True,
            )
        else:
            print(
                f"[{idx}/{len(trials)}] {row['status']} sel={params.ccik_selector_keep} ext={params.ccik_external_keep} "
                f"nk={params.ccik_nkeep} gen={params.ccik_generator_keep} mc={params.ccik_restart_m_cycle} "
                f"nr={params.ccik_restart_nroot} cy={params.ccik_max_cycle} slots={row['estimated_storage_slots']} wall={row['wall_s']}",
                flush=True,
            )
        write_results(out_dir / "results.csv", rows)
        write_summary(out_dir / "summary.txt", rows, args=args, nmo_total=nmo_total)

    if not args.keep_template_ezfio and template_ezfio.exists():
        shutil.rmtree(template_ezfio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
