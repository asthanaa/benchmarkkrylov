"""Sequential fixed-orbital Cr2 campaign: CIPSI first, then CCIK/CCIK-thick."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cr2_qp_ccik_tune import create_xyz, cr2_spatial_orbitals, make_cr2_atom  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    CCIKParams,
    QPPaths,
    discover_qp_paths,
    ensure_ezfio_defaults,
    set_ccik_params,
)


DEFAULT_CIPSI_SEED = (
    _ROOT
    / "experiments"
    / "cr2_qp_campaign_20260314"
    / "cipsi_qprun_from_plain9797_ndet2000000_20260320"
    / "cr2.ezfio"
)
DEFAULT_CIPSI_LOG = (
    _ROOT
    / "experiments"
    / "cr2_qp_campaign_20260314"
    / "cipsi_qprun_from_plain9797_ndet2000000_20260320"
    / "qp_run.log"
)
DEFAULT_N2_PY_GATE = _ROOT / "experiments" / "n2_accuracy_gate_20260320" / "gate.json"
DEFAULT_N2_QP_GATE = _ROOT / "experiments" / "n2_qp_exact_gate_20260320" / "gate.json"
DEFAULT_CR2_DONOR = (
    _ROOT
    / "experiments"
    / "cr2_qp_campaign_20260314"
    / "repro_w0_f4_rel"
    / "cr2.ezfio"
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_gate(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing {label} gate file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not bool(data.get("overall_passed", False)):
        raise RuntimeError(f"{label} gate did not pass: {path}")
    return data


def _parse_cipsi_summaries(text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    current: dict[str, float | int] | None = None
    for line in text.splitlines():
        m = re.search(r"Summary at N_det =\s*(\d+)", line)
        if m:
            current = {"n_det": int(m.group(1))}
            rows.append(current)
            continue
        if current is None:
            continue
        m = re.search(r"# E\s+(-?\d+\.\d+)", line)
        if m and "E+PT2" not in line:
            current["e_var"] = float(m.group(1))
        m = re.search(r"# E\+PT2\s+(-?\d+\.\d+)", line)
        if m:
            current["e_pt2"] = float(m.group(1))
    return rows


def _parse_ccik_summary(text: str) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "e_var": None,
        "e_elec": None,
        "e_pt2": None,
        "pt2_corr": None,
        "basis": None,
        "saved_det": None,
    }
    patterns = {
        "e_var": r"CCIK_FINAL_TOTAL_ENERGY =\s*([-+0-9.Ee]+)",
        "e_elec": r"CCIK_FINAL_ELECTRONIC_ENERGY =\s*([-+0-9.Ee]+)",
        "e_pt2": r"CCIK_FINAL_E_PLUS_PT2 =\s*([-+0-9.Ee]+)",
        "pt2_corr": r"CCIK_FINAL_PT2_CORRECTION =\s*([-+0-9.Ee]+)",
        "basis": r"CCIK_FINAL_BASIS =\s*(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            out[key] = int(m.group(1)) if key == "basis" else float(m.group(1))
    saved = re.findall(r"\* Saved determinants\s+(\d+)", text)
    if saved:
        out["saved_det"] = int(saved[-1])
    return out


def _short_ezfio_link(target: Path, short_name: str) -> Path:
    short = Path("/private/tmp") / short_name
    if short.exists() or short.is_symlink():
        short.unlink()
    short.symlink_to(target)
    return short


def _run_qp_fci(qp: QPPaths, ezfio: Path, log_path: Path, *, n_det_max: int, use_existing_wf: bool) -> tuple[int, float]:
    short = _short_ezfio_link(ezfio, f"{ezfio.stem}_{n_det_max}.ezfio")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "24")
    env.setdefault("QP_NIC", "lo0")
    setup_cmd = (
        f"cd {qp.qp_root} && "
        "set +u && "
        f"source {qp.quantum_package_rc} && "
        "set -u && "
        f"export OMP_NUM_THREADS={env['OMP_NUM_THREADS']} && "
        f"export QP_NIC={env['QP_NIC']} && "
        f"export PATH={qp.qp_root}/localbin:$PATH && "
        f"qp set_file {short} && "
        f"qp set determinants n_det_max {n_det_max} && "
        f"qp set determinants read_wf {'true' if use_existing_wf else 'false'}"
    )
    subprocess.run(["/bin/bash", "-lc", setup_cmd], check=True, text=True, env=env)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [
                "/bin/bash",
                "-lc",
                (
                    f"cd {qp.qp_root} && "
                    "set +u && "
                    f"source {qp.quantum_package_rc} && "
                    "set -u && "
                    f"export OMP_NUM_THREADS={env['OMP_NUM_THREADS']} && "
                    f"export QP_NIC={env['QP_NIC']} && "
                    f"export PATH={qp.qp_root}/localbin:$PATH && "
                    f"qp_run fci {short}"
                ),
            ],
            text=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    wall = float(time.perf_counter() - t0)
    short.unlink(missing_ok=True)
    return proc.returncode, wall


def _run_ccik_job(qp: QPPaths, ezfio: Path, log_path: Path, params: CCIKParams, *, n_det_max: int) -> tuple[int, float]:
    short = _short_ezfio_link(ezfio, f"{ezfio.stem}_ccik.ezfio")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "24")
    env.setdefault("QP_NIC", "lo0")
    set_ccik_params(qp, ezfio, params)
    subprocess.run(
        [
            "/bin/bash",
            "-lc",
            (
                f"source {qp.quantum_package_rc} && "
                f"qp set_file {short} && "
                f"qp set determinants n_det_max {n_det_max} && "
                f"qp set determinants read_wf {'true' if params.ccik_use_existing_wf else 'false'}"
            ),
        ],
        check=True,
        text=True,
        env=env,
    )
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            ["/bin/bash", "-lc", f"source {qp.quantum_package_rc} && {qp.ccik_binary} {short}"],
            text=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    wall = float(time.perf_counter() - t0)
    short.unlink(missing_ok=True)
    return proc.returncode, wall


def _existing_best_cipsi(log_path: Path) -> dict[str, float | int]:
    rows = _parse_cipsi_summaries(log_path.read_text(errors="ignore"))
    if not rows:
        raise RuntimeError(f"No CIPSI summaries found in {log_path}")
    last = rows[-1]
    return {"n_det": int(last["n_det"]), "e_var": float(last["e_var"]), "e_pt2": float(last["e_pt2"])}


def _fresh_cr2_template(qp: QPPaths, template_dir: Path, donor_ezfio: Path) -> Path:
    if not donor_ezfio.exists():
        raise RuntimeError(f"Missing Cr2 donor EZFIO: {donor_ezfio}")
    template_ezfio = template_dir / "cr2_template.ezfio"
    if template_ezfio.exists():
        shutil.rmtree(template_ezfio)
    shutil.copytree(donor_ezfio, template_ezfio)
    ensure_ezfio_defaults(template_ezfio)
    return template_ezfio


def _ccik_branch_configs(branch: str) -> list[tuple[str, CCIKParams]]:
    common = dict(
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_include_current_support=True,
        ccik_external_candidate_factor=8,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_tmp_keep_factor=4,
        ccik_preserve_prev_support=True,
        ccik_preserve_prev_support_cap=2048,
        ccik_total_det_budget=0,
        ccik_conv_tol=1e-12,
        ccik_ritz_overlap_tol=1e-8,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=True,
        ccik_do_post_pt2=True,
        ccik_verbose=False,
    )
    if branch == "ccik":
        return [
            (
                "ccik_r1_keep32768_m40",
                CCIKParams(
                    ccik_basis_size=40,
                    ccik_generator_keep=2048,
                    ccik_max_parents_for_expansion=2048,
                    ccik_selector_keep=32768,
                    ccik_external_keep=32768,
                    ccik_nkeep=32768,
                    ccik_use_existing_wf=False,
                    ccik_max_cycle=1,
                    ccik_restart_m_cycle=0,
                    ccik_restart_nroot=2,
                    **common,
                ),
            ),
            (
                "ccik_r2_keep65536_m40",
                CCIKParams(
                    ccik_basis_size=40,
                    ccik_generator_keep=4096,
                    ccik_max_parents_for_expansion=4096,
                    ccik_selector_keep=65536,
                    ccik_external_keep=65536,
                    ccik_nkeep=65536,
                    ccik_use_existing_wf=True,
                    ccik_max_cycle=1,
                    ccik_restart_m_cycle=0,
                    ccik_restart_nroot=2,
                    **common,
                ),
            ),
            (
                "ccik_r3_keep65536_m60",
                CCIKParams(
                    ccik_basis_size=60,
                    ccik_generator_keep=4096,
                    ccik_max_parents_for_expansion=4096,
                    ccik_selector_keep=65536,
                    ccik_external_keep=65536,
                    ccik_nkeep=65536,
                    ccik_use_existing_wf=True,
                    ccik_max_cycle=1,
                    ccik_restart_m_cycle=0,
                    ccik_restart_nroot=2,
                    **common,
                ),
            ),
            (
                "ccik_r4_keep131072_m60",
                CCIKParams(
                    ccik_basis_size=60,
                    ccik_generator_keep=8192,
                    ccik_max_parents_for_expansion=8192,
                    ccik_selector_keep=131072,
                    ccik_external_keep=131072,
                    ccik_nkeep=131072,
                    ccik_use_existing_wf=True,
                    ccik_max_cycle=1,
                    ccik_restart_m_cycle=0,
                    ccik_restart_nroot=2,
                    **common,
                ),
            ),
        ]
    return [
        (
            "thick_r1_keep32768_mc40_c40_nr10",
            CCIKParams(
                ccik_basis_size=24,
                ccik_generator_keep=2048,
                ccik_max_parents_for_expansion=2048,
                ccik_selector_keep=32768,
                ccik_external_keep=32768,
                ccik_nkeep=32768,
                ccik_use_existing_wf=False,
                ccik_max_cycle=40,
                ccik_restart_m_cycle=40,
                ccik_restart_nroot=10,
                **common,
            ),
        ),
        (
            "thick_r2_keep65536_mc40_c40_nr10",
            CCIKParams(
                ccik_basis_size=24,
                ccik_generator_keep=4096,
                ccik_max_parents_for_expansion=4096,
                ccik_selector_keep=65536,
                ccik_external_keep=65536,
                ccik_nkeep=65536,
                ccik_use_existing_wf=True,
                ccik_max_cycle=40,
                ccik_restart_m_cycle=40,
                ccik_restart_nroot=10,
                **common,
            ),
        ),
        (
            "thick_r3_keep65536_mc50_c60_nr10",
            CCIKParams(
                ccik_basis_size=24,
                ccik_generator_keep=4096,
                ccik_max_parents_for_expansion=4096,
                ccik_selector_keep=65536,
                ccik_external_keep=65536,
                ccik_nkeep=65536,
                ccik_use_existing_wf=True,
                ccik_max_cycle=60,
                ccik_restart_m_cycle=50,
                ccik_restart_nroot=10,
                **common,
            ),
        ),
        (
            "thick_r4_keep131072_mc50_c60_nr10",
            CCIKParams(
                ccik_basis_size=24,
                ccik_generator_keep=8192,
                ccik_max_parents_for_expansion=8192,
                ccik_selector_keep=131072,
                ccik_external_keep=131072,
                ccik_nkeep=131072,
                ccik_use_existing_wf=True,
                ccik_max_cycle=60,
                ccik_restart_m_cycle=50,
                ccik_restart_nroot=10,
                **common,
            ),
        ),
        (
            "thick_r5_keep131072_mc50_c60_nr12",
            CCIKParams(
                ccik_basis_size=24,
                ccik_generator_keep=8192,
                ccik_max_parents_for_expansion=8192,
                ccik_selector_keep=131072,
                ccik_external_keep=131072,
                ccik_nkeep=131072,
                ccik_use_existing_wf=True,
                ccik_max_cycle=60,
                ccik_restart_m_cycle=50,
                ccik_restart_nroot=12,
                **common,
            ),
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-orbital Cr2 campaign")
    p.add_argument("--outdir", type=Path, default=_ROOT / "experiments" / "cr2_fixed_orbital_campaign_20260320")
    p.add_argument("--cipsi-seed-ezfio", type=Path, default=DEFAULT_CIPSI_SEED)
    p.add_argument("--cipsi-seed-log", type=Path, default=DEFAULT_CIPSI_LOG)
    p.add_argument("--n2-py-gate", type=Path, default=DEFAULT_N2_PY_GATE)
    p.add_argument("--n2-qp-gate", type=Path, default=DEFAULT_N2_QP_GATE)
    p.add_argument("--cr2-donor-ezfio", type=Path, default=DEFAULT_CR2_DONOR)
    p.add_argument("--cipsi-rungs", type=int, nargs="+", default=[5_000_000, 10_000_000])
    p.add_argument("--ccik-n-det-max", type=int, default=10_000_000)
    p.add_argument("--improvement-stop-eh", type=float, default=5.0e-4)
    p.add_argument("--nroot-bump-threshold-eh", type=float, default=1.0e-3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    py_gate = _parse_gate(args.n2_py_gate, "N2 Python")
    qp_gate = _parse_gate(args.n2_qp_gate, "N2 QP")
    qp = discover_qp_paths()

    summary_rows: list[dict[str, object]] = []

    # Stage 3A: CIPSI ladder
    cipsi_dir = outdir / "cipsi"
    cipsi_dir.mkdir(parents=True, exist_ok=True)
    base_cipsi = _existing_best_cipsi(args.cipsi_seed_log)
    summary_rows.append(
        {
            "stage": "cipsi_base",
            "label": "existing_2513196",
            "n_det": base_cipsi["n_det"],
            "e_var": base_cipsi["e_var"],
            "e_pt2": base_cipsi["e_pt2"],
            "wall_s": "",
            "status": "existing",
        }
    )
    prev_e_var = float(base_cipsi["e_var"])
    run_10m = False
    current_seed = args.cipsi_seed_ezfio
    for idx, n_det_max in enumerate(args.cipsi_rungs, start=1):
        if idx > 1 and not run_10m:
            break
        rung_dir = cipsi_dir / f"cipsi_{n_det_max}"
        rung_dir.mkdir(parents=True, exist_ok=True)
        ezfio = rung_dir / "cr2.ezfio"
        if ezfio.exists():
            shutil.rmtree(ezfio)
        shutil.copytree(current_seed, ezfio)
        log_path = rung_dir / "qp_run.log"
        rc, wall_s = _run_qp_fci(qp, ezfio, log_path, n_det_max=n_det_max, use_existing_wf=True)
        parsed = _parse_cipsi_summaries(log_path.read_text(errors="ignore"))
        if not parsed:
            raise RuntimeError(f"No CIPSI summaries found in {log_path}")
        best = parsed[-1]
        e_var = float(best["e_var"])
        improvement = prev_e_var - e_var
        summary_rows.append(
            {
                "stage": "cipsi",
                "label": f"n_det_max_{n_det_max}",
                "n_det": int(best["n_det"]),
                "e_var": e_var,
                "e_pt2": float(best["e_pt2"]),
                "wall_s": wall_s,
                "status": f"rc_{rc}",
                "improvement_vs_prev_eh": improvement,
            }
        )
        current_seed = ezfio
        if idx == 1:
            run_10m = improvement > args.improvement_stop_eh
        prev_e_var = e_var

    # Stage 3B: calibration baseline vs thick
    ccik_dir = outdir / "ccik"
    ccik_dir.mkdir(parents=True, exist_ok=True)
    template_dir = ccik_dir / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_ezfio = _fresh_cr2_template(qp, template_dir, args.cr2_donor_ezfio)

    calibration_specs = {
        "ccik": _ccik_branch_configs("ccik")[0][1],
        "ccik_thick": _ccik_branch_configs("ccik_thick")[0][1],
    }
    calibration_results: dict[str, dict[str, object]] = {}
    for branch, params in calibration_specs.items():
        branch_dir = ccik_dir / f"calibration_{branch}"
        branch_dir.mkdir(parents=True, exist_ok=True)
        ezfio = branch_dir / "cr2.ezfio"
        if ezfio.exists():
            shutil.rmtree(ezfio)
        shutil.copytree(template_ezfio, ezfio)
        log_path = branch_dir / "ccik.log"
        rc, wall_s = _run_ccik_job(qp, ezfio, log_path, params, n_det_max=args.ccik_n_det_max)
        parsed = _parse_ccik_summary(log_path.read_text(errors="ignore"))
        row = {
            "stage": "ccik_calibration",
            "label": branch,
            "e_var": parsed["e_var"],
            "e_pt2": parsed["e_pt2"],
            "pt2_corr": parsed["pt2_corr"],
            "basis": parsed["basis"],
            "saved_det": parsed["saved_det"],
            "wall_s": wall_s,
            "status": f"rc_{rc}",
        }
        calibration_results[branch] = row
        summary_rows.append(row)

    e_base = float(calibration_results["ccik"]["e_var"])
    e_thick = float(calibration_results["ccik_thick"]["e_var"])
    chosen_branch = "ccik_thick" if (e_thick <= e_base + 5.0e-4) else "ccik"

    # Stage 3C: production ladder
    ladder = _ccik_branch_configs(chosen_branch)
    best_e = None
    small_improvement_streak = 0
    bump_nroot = False
    previous_ezfio: Path | None = None
    for label, params in ladder:
        if chosen_branch == "ccik_thick" and label.endswith("nr12") and not bump_nroot:
            continue
        rung_dir = ccik_dir / label
        rung_dir.mkdir(parents=True, exist_ok=True)
        ezfio = rung_dir / "cr2.ezfio"
        if ezfio.exists():
            shutil.rmtree(ezfio)
        seed = template_ezfio if previous_ezfio is None else previous_ezfio
        shutil.copytree(seed, ezfio)
        effective_params = params
        if previous_ezfio is None and effective_params.ccik_use_existing_wf:
            effective_params = CCIKParams(**{**effective_params.__dict__, "ccik_use_existing_wf": False})
        log_path = rung_dir / "ccik.log"
        rc, wall_s = _run_ccik_job(qp, ezfio, log_path, effective_params, n_det_max=args.ccik_n_det_max)
        parsed = _parse_ccik_summary(log_path.read_text(errors="ignore"))
        e_var = float(parsed["e_var"])
        improvement = None if best_e is None else best_e - e_var
        row = {
            "stage": "ccik_production",
            "label": label,
            "branch": chosen_branch,
            "e_var": e_var,
            "e_pt2": parsed["e_pt2"],
            "pt2_corr": parsed["pt2_corr"],
            "basis": parsed["basis"],
            "saved_det": parsed["saved_det"],
            "wall_s": wall_s,
            "status": f"rc_{rc}",
            "improvement_vs_prev_best_eh": improvement,
        }
        summary_rows.append(row)

        if best_e is None:
            best_e = e_var
        else:
            delta = best_e - e_var
            if delta > 0.0:
                best_e = e_var
            if delta < args.improvement_stop_eh:
                small_improvement_streak += 1
            else:
                small_improvement_streak = 0
            if chosen_branch == "ccik_thick" and "keep65536" in label and delta > args.nroot_bump_threshold_eh:
                bump_nroot = True
            if small_improvement_streak >= 2:
                previous_ezfio = ezfio
                break
        previous_ezfio = ezfio

    _write_csv(outdir / "campaign_results.csv", summary_rows)
    payload = {
        "n2_python_gate": py_gate,
        "n2_qp_gate": qp_gate,
        "chosen_branch": chosen_branch,
        "rows": summary_rows,
    }
    (outdir / "campaign_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Cr2 Fixed-Orbital Campaign",
        "",
        f"- Chosen CCIK branch: `{chosen_branch}`",
        f"- N2 Python gate passed: `{py_gate['overall_passed']}`",
        f"- N2 QP gate passed: `{qp_gate['overall_passed']}`",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- `{row['stage']}` / `{row['label']}`: E_var={row.get('e_var')} E+PT2={row.get('e_pt2')} "
            f"wall={row.get('wall_s')} status={row.get('status')}"
        )
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"chosen_branch": chosen_branch, "n_rows": len(summary_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
