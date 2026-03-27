"""One-point N2 CAS(12,12) method comparison on a donor-backed QP Hamiltonian.

This script keeps the Hamiltonian fixed by exporting FCIDUMP from a prepared
Quantum Package EZFIO, then runs:

- exact FCI on the exported FCIDUMP
- block2 DMRG on the same FCIDUMP
- QP CIPSI from a copied donor EZFIO
- QP CCIK baseline and thick-restart from copied donor EZFIOs
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qp_ccik_common import CCIKParams, QPPaths, discover_qp_paths, ensure_ezfio_defaults, set_ccik_params  # noqa: E402


DEFAULT_DONOR = ROOT / "experiments" / "n2_qp_ccik_cas12_12_R2p50_b" / "n2_2p50.ezfio"
DEFAULT_OUTDIR = ROOT / "experiments" / "n2_cas12_method_compare_20260324"
DEFAULT_BLOCK2_PYTHON = ROOT / "experiments" / "block2_cr2_d2p0_sto3g" / ".venv" / "bin" / "python"
DEFAULT_QP_ROOT = Path("/private/tmp/qp2_fresh_20260320")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run N2 CAS(12,12) DMRG/CIPSI/CCIK comparison")
    p.add_argument("--donor-ezfio", type=Path, default=DEFAULT_DONOR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--qp-root", type=Path, default=DEFAULT_QP_ROOT)
    p.add_argument("--block2-python", type=Path, default=DEFAULT_BLOCK2_PYTHON)
    p.add_argument("--cipsi-n-det-max", type=int, default=2_000_000)
    p.add_argument("--ccik-n-det-max", type=int, default=2_000_000)
    p.add_argument("--dmrg-memory-gb", type=int, default=64)
    p.add_argument("--dmrg-threads", type=int, default=16)
    p.add_argument("--dmrg-conv-tol", type=float, default=1e-8)
    p.add_argument("--keep-going", action="store_true", help="Continue after a method failure and record the error")
    return p.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bash(cmd: str, *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-lc", cmd],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd,
    )


def _qp_env(qp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["QP_ROOT"] = str(qp_root)
    env.setdefault("OMP_NUM_THREADS", "24")
    env.setdefault("QP_NIC", "lo0")
    return env


def _generate_qp_fcidump(qp: QPPaths, ezfio: Path, outdir: Path) -> Path:
    from pyscf.tools import fcidump as pyscf_fcidump

    fcidump_binary = qp.qp_root / "src" / "tools" / "fcidump"
    env = _qp_env(qp.qp_root)
    short = _short_ezfio_link(ezfio, "n2_cas12_prepare.ezfio")
    _bash(f"source {qp.quantum_package_rc} && {fcidump_binary} {short}", env=env, cwd=short.parent)
    raw = Path(str(short) + ".FCIDUMP")
    copied = outdir / "qp_active.FCIDUMP"
    shutil.copy2(raw, copied)
    raw.unlink(missing_ok=True)
    short.unlink(missing_ok=True)
    text = copied.read_text(encoding="utf-8")
    text = text.replace(" ISYM=0,", " ISYM=1,")
    copied.write_text(text, encoding="utf-8")
    data = pyscf_fcidump.read(str(copied), verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec + ms2) // 2
    nelecb = nelec - neleca
    _write_json(
        outdir / "qp_fcidump_summary.json",
        {
            "fcidump_path": str(copied),
            "norb": norb,
            "nelec": [int(neleca), int(nelecb)],
            "ecore": float(data.get("ECORE", 0.0)),
        },
    )
    return copied


def _run_exact_fci(fcidump_path: Path, outdir: Path) -> dict[str, Any]:
    from pyscf import ao2mo, fci
    from pyscf.tools import fcidump as pyscf_fcidump

    t0 = time.perf_counter()
    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec_tot = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec_tot + ms2) // 2
    nelecb = nelec_tot - neleca
    ecore = float(data.get("ECORE", 0.0))
    h1 = data["H1"]
    eri8 = ao2mo.restore(8, data["H2"], norb)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    solver.max_cycle = 200
    solver.max_space = 100
    e_elec, _ = solver.kernel(h1, eri8, norb, (int(neleca), int(nelecb)))
    wall_s = float(time.perf_counter() - t0)
    result = {
        "method": "exact_fci",
        "fcidump_path": str(fcidump_path),
        "norb": norb,
        "nelec": [int(neleca), int(nelecb)],
        "electronic_energy_eh": float(e_elec),
        "total_energy_eh": float(e_elec + ecore),
        "ecore_eh": ecore,
        "wall_s": wall_s,
    }
    _write_json(outdir / "exact_fci_summary.json", result)
    return result


def _run_dmrg_same_fcidump(
    fcidump_path: Path,
    outdir: Path,
    *,
    memory_gb: int,
    threads: int,
    conv_tol: float,
) -> dict[str, Any]:
    from pyblock2.gsdmrg import GSDMRG
    from pyscf.tools import fcidump as pyscf_fcidump

    outdir.mkdir(parents=True, exist_ok=True)
    runtime_dir = outdir / "runtime"
    scratch_dir = outdir / "scratch"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "dmrg.log"
    summary_path = outdir / "dmrg_summary.json"

    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    ecore = float(data.get("ECORE", 0.0))
    t0 = time.perf_counter()

    bond_dims = [250, 500, 1000, 1500, 2000, 2500, 3000, 3000]
    noises = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0, 0.0, 0.0]
    dav_thrds = [1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9]

    with log_path.open("w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        os.environ["TMPDIR"] = str(scratch_dir)
        dmrg = GSDMRG(
            scratch=str(scratch_dir),
            memory=float(memory_gb) * 1e9,
            omp_threads=int(threads),
            verbose=3,
            print_statistics=True,
        )
        dmrg.init_hamiltonian_fcidump("c1", str(fcidump_path))
        e_total = float(
            dmrg.dmrg(
                bond_dims=bond_dims,
                noises=noises,
                dav_thrds=dav_thrds,
                occs=None,
                n_steps=32,
                conv_tol=float(conv_tol),
                cutoff=1e-14,
            )
        )
        del dmrg

    wall_s = float(time.perf_counter() - t0)
    summary = {
        "method": "dmrg",
        "fcidump_path": str(fcidump_path),
        "total_energy_eh": e_total,
        "electronic_energy_eh": float(e_total - ecore),
        "ecore_eh": ecore,
        "bond_dims": bond_dims,
        "noises": noises,
        "dav_thrds": dav_thrds,
        "threads": int(threads),
        "memory_gb": int(memory_gb),
        "conv_tol": float(conv_tol),
        "wall_s": wall_s,
        "log_path": str(log_path),
    }
    _write_json(summary_path, summary)
    return summary


def _copy_donor(donor: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(donor, target)
    ensure_ezfio_defaults(target)
    return target


def _short_ezfio_link(target: Path, short_name: str) -> Path:
    short = Path("/private/tmp") / short_name
    if short.exists() or short.is_symlink():
        short.unlink()
    short.symlink_to(target)
    return short


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


def _run_qp_fci(qp: QPPaths, ezfio: Path, log_path: Path, *, n_det_max: int, use_existing_wf: bool) -> dict[str, Any]:
    env = _qp_env(qp.qp_root)
    short = _short_ezfio_link(ezfio, f"{ezfio.stem}_{n_det_max}.ezfio")
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
    wall_s = float(time.perf_counter() - t0)
    short.unlink(missing_ok=True)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = _parse_cipsi_summaries(text)
    final_energy_match = re.findall(r"\* Energy of state\s+\d+\s+([-+0-9.Ee]+)", text)
    saved_det_match = re.findall(r"\* Saved determinants\s+(\d+)", text)
    result: dict[str, Any] = {
        "method": "cipsi",
        "n_det_max": int(n_det_max),
        "rc": int(proc.returncode),
        "wall_s": wall_s,
        "log_path": str(log_path),
        "checkpoint_count": len(rows),
    }
    if rows:
        last = rows[-1]
        result.update(
            {
                "final_n_det": int(last["n_det"]),
                "e_var": float(last["e_var"]),
                "e_pt2": float(last["e_pt2"]),
            }
        )
    if final_energy_match:
        result["final_davidson_energy"] = float(final_energy_match[-1])
        if result.get("e_var") is None or float(result["final_davidson_energy"]) < float(result["e_var"]):
            result["e_var"] = float(result["final_davidson_energy"])
    if saved_det_match:
        result["saved_det"] = int(saved_det_match[-1])
    return result


def _run_ccik_job(qp: QPPaths, ezfio: Path, log_path: Path, params: CCIKParams, *, n_det_max: int) -> dict[str, Any]:
    env = _qp_env(qp.qp_root)
    short = _short_ezfio_link(ezfio, f"{ezfio.stem}_ccik.ezfio")
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
    wall_s = float(time.perf_counter() - t0)
    short.unlink(missing_ok=True)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    summary = _parse_ccik_summary(text)
    result: dict[str, Any] = {
        "method": "ccik",
        "rc": int(proc.returncode),
        "wall_s": wall_s,
        "log_path": str(log_path),
        **summary,
    }
    return result


def _baseline_params() -> CCIKParams:
    return CCIKParams(
        ccik_basis_size=40,
        ccik_generator_keep=2048,
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_max_parents_for_expansion=2048,
        ccik_include_current_support=True,
        ccik_selector_keep=65536,
        ccik_external_keep=65536,
        ccik_external_candidate_factor=6,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_nkeep=65536,
        ccik_tmp_keep_factor=4,
        ccik_preserve_prev_support=True,
        ccik_preserve_prev_support_cap=4096,
        ccik_total_det_budget=0,
        ccik_conv_tol=1e-12,
        ccik_ritz_overlap_tol=1e-8,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=True,
        ccik_use_existing_wf=True,
        ccik_do_post_pt2=True,
        ccik_max_cycle=1,
        ccik_restart_m_cycle=0,
        ccik_restart_nroot=2,
        ccik_verbose=False,
    )


def _thick_params() -> CCIKParams:
    return CCIKParams(
        ccik_basis_size=24,
        ccik_generator_keep=2048,
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_max_parents_for_expansion=2048,
        ccik_include_current_support=True,
        ccik_selector_keep=65536,
        ccik_external_keep=65536,
        ccik_external_candidate_factor=6,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_nkeep=65536,
        ccik_tmp_keep_factor=4,
        ccik_preserve_prev_support=True,
        ccik_preserve_prev_support_cap=4096,
        ccik_total_det_budget=0,
        ccik_conv_tol=1e-12,
        ccik_ritz_overlap_tol=1e-8,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=True,
        ccik_use_existing_wf=True,
        ccik_do_post_pt2=True,
        ccik_max_cycle=20,
        ccik_restart_m_cycle=20,
        ccik_restart_nroot=6,
        ccik_verbose=False,
    )


def _summary_rows(results: dict[str, Any], exact_total: float | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in ("exact_fci", "dmrg", "cipsi", "ccik_baseline", "ccik_thick"):
        payload = results.get(key)
        if not isinstance(payload, dict):
            continue
        row: dict[str, object] = {
            "label": key,
            "method": payload.get("method", key),
            "total_energy_eh": payload.get("total_energy_eh", payload.get("e_var")),
            "electronic_energy_eh": payload.get("electronic_energy_eh", payload.get("e_elec")),
            "e_pt2": payload.get("e_pt2"),
            "final_n_det": payload.get("final_n_det", payload.get("saved_det")),
            "basis": payload.get("basis"),
            "wall_s": payload.get("wall_s"),
            "rc": payload.get("rc"),
        }
        if exact_total is not None and row["total_energy_eh"] is not None:
            row["delta_vs_exact_mEh"] = 1000.0 * (float(row["total_energy_eh"]) - float(exact_total))
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.donor_ezfio.exists():
        raise RuntimeError(f"Missing donor EZFIO: {args.donor_ezfio}")
    if not args.block2_python.exists():
        raise RuntimeError(f"Missing block2 Python interpreter: {args.block2_python}")

    os.environ["QP_ROOT"] = str(args.qp_root)
    qp = discover_qp_paths()

    results: dict[str, Any] = {
        "system": "N2",
        "basis": "6-31g",
        "distance_angstrom": 2.50,
        "active_space": {"ncore": 1, "ncas": 12, "nelecas": 12},
        "donor_ezfio": str(args.donor_ezfio.resolve()),
        "qp_root": str(args.qp_root.resolve()),
        "block2_python": str(args.block2_python.resolve()),
    }

    def run_step(name: str, fn) -> None:
        try:
            results[name] = fn()
            _write_json(args.outdir / "results_partial.json", results)
        except Exception as exc:
            results[name] = {"error": str(exc), "type": type(exc).__name__}
            _write_json(args.outdir / "results_partial.json", results)
            if not args.keep_going:
                raise

    prepared_ezfio = _copy_donor(args.donor_ezfio.resolve(), args.outdir / "prepared" / "n2.ezfio")
    results["prepared_ezfio"] = str(prepared_ezfio)
    fcidump_path = _generate_qp_fcidump(qp, prepared_ezfio, args.outdir)
    results["fcidump_path"] = str(fcidump_path)
    _write_json(args.outdir / "results_partial.json", results)

    run_step("exact_fci", lambda: _run_exact_fci(fcidump_path, args.outdir))
    run_step(
        "dmrg",
        lambda: _run_dmrg_same_fcidump(
            fcidump_path,
            args.outdir / "dmrg",
            memory_gb=args.dmrg_memory_gb,
            threads=args.dmrg_threads,
            conv_tol=args.dmrg_conv_tol,
        ),
    )

    cipsi_dir = args.outdir / "cipsi"
    cipsi_ezfio = _copy_donor(prepared_ezfio, cipsi_dir / "n2.ezfio")
    run_step(
        "cipsi",
        lambda: _run_qp_fci(
            qp,
            cipsi_ezfio,
            cipsi_dir / "qp_run.log",
            n_det_max=args.cipsi_n_det_max,
            use_existing_wf=True,
        ),
    )

    ccik_dir = args.outdir / "ccik_baseline"
    ccik_ezfio = _copy_donor(prepared_ezfio, ccik_dir / "n2.ezfio")
    run_step(
        "ccik_baseline",
        lambda: _run_ccik_job(
            qp,
            ccik_ezfio,
            ccik_dir / "ccik.log",
            _baseline_params(),
            n_det_max=args.ccik_n_det_max,
        ),
    )

    thick_dir = args.outdir / "ccik_thick"
    thick_ezfio = _copy_donor(prepared_ezfio, thick_dir / "n2.ezfio")
    run_step(
        "ccik_thick",
        lambda: _run_ccik_job(
            qp,
            thick_ezfio,
            thick_dir / "ccik.log",
            _thick_params(),
            n_det_max=args.ccik_n_det_max,
        ),
    )

    exact_total = None
    if isinstance(results.get("exact_fci"), dict):
        exact_total = results["exact_fci"].get("total_energy_eh")
    rows = _summary_rows(results, exact_total if isinstance(exact_total, (int, float)) else None)
    _write_csv(args.outdir / "summary.csv", rows)
    _write_json(args.outdir / "results.json", results)

    lines = [
        "# N2 CAS(12,12) Method Comparison",
        "",
        f"- donor EZFIO: `{args.donor_ezfio}`",
        f"- FCIDUMP: `{fcidump_path}`",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        label = str(row["label"])
        method = str(row["method"])
        total = row.get("total_energy_eh")
        delta = row.get("delta_vs_exact_mEh")
        wall = row.get("wall_s")
        line = f"- `{label}` / `{method}`: E = {total}"
        if delta is not None:
            line += f", Delta_exact = {delta} mEh"
        if wall is not None:
            line += f", wall = {wall:.1f} s"
        lines.append(line)
    (args.outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
