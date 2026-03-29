"""Sweep QP-CCIK hard-point settings and record progress/accuracy.

This script is intended for difficult single-point cases where we want to know:

- whether thick-restart reaches cycle-level Ritz output at all
- whether the run finishes within a practical timeout
- how the first projected-Hq call breaks down by cost
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "experiments" / "n2_qp_ccik_cas14_14_hard_budget1e7" / "n2_2p50.ezfio"
DEFAULT_OUTPUT = ROOT / "experiments" / "n2_qp_ccik_cas14_14_selector_sweep"


def bash(cmd: str) -> None:
    subprocess.run(["/bin/bash", "-lc", cmd], check=True, text=True)


def qp_root() -> Path:
    return Path(os.environ.get("QP_ROOT", "/tmp/qp2")).resolve()


def qp_rc() -> Path:
    return qp_root() / "quantum_package.rc"


def qp_binary() -> Path:
    return qp_root() / "plugins" / "ccik_qp" / "stable" / "ccik" / "ccik"


def qp_set_ccik(ezfio: Path, params: dict[str, object]) -> None:
    assignments = " && ".join(f"qp set ccik {k} {v}" for k, v in params.items())
    cmd = (
        f"cd {qp_root()} && "
        f"source {qp_rc()} >/dev/null 2>&1 && "
        f"cd '{ROOT}' && "
        f"qp set_file '{ezfio}' && "
        f"{assignments}"
    )
    bash(cmd)


def _to_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return data


def parse_first_hq_done(log_text: str) -> dict[str, object]:
    pat = re.compile(
        r"ccik\(hq\) done\s+call=\s*(\d+)\s+buf=\s*(\d+)\s+n_ext=\s*(\d+)\s+n_total=\s*(\d+)\s+\|vec\|=\s*(\d+)"
        r"\s+select_s=\s*([-+0-9.Ee]+)\s+sort_s=\s*([-+0-9.Ee]+)\s+int_s=\s*([-+0-9.Ee]+)"
        r"\s+ext_s=\s*([-+0-9.Ee]+)\s+assign_s=\s*([-+0-9.Ee]+)"
    )
    m = pat.search(log_text)
    if not m:
        return {}
    return {
        "hq_call": int(m.group(1)),
        "hq_buf": int(m.group(2)),
        "hq_n_ext": int(m.group(3)),
        "hq_n_total": int(m.group(4)),
        "hq_vec_size": int(m.group(5)),
        "hq_select_s": float(m.group(6)),
        "hq_sort_s": float(m.group(7)),
        "hq_int_s": float(m.group(8)),
        "hq_ext_s": float(m.group(9)),
        "hq_assign_s": float(m.group(10)),
    }


def parse_final_total(log_text: str) -> float | None:
    m = re.search(r"CCIK_FINAL_TOTAL_ENERGY =\s*([-+0-9.Ee]+)", log_text)
    return None if m is None else float(m.group(1))


def parse_final_basis(log_text: str) -> int | None:
    m = re.search(r"CCIK_FINAL_BASIS =\s*(\d+)", log_text)
    return None if m is None else int(m.group(1))


def count_ritz_lines(log_text: str) -> int:
    return len(re.findall(r"ccik\(thick\) cycle=\s*\d+.*ritz=", log_text))


def run_trial(
    *,
    template_ezfio: Path,
    trial_dir: Path,
    params: dict[str, object],
    timeout_s: int,
) -> dict[str, object]:
    ezfio = trial_dir / "n2_2p50.ezfio"
    if ezfio.exists():
        shutil.rmtree(ezfio)
    shutil.copytree(template_ezfio, ezfio)

    qp_set_ccik(ezfio, params)

    log_path = trial_dir / "run.log"
    start = time.time()
    status = "finished"
    try:
        proc = subprocess.run(
            [str(qp_binary()), str(ezfio)],
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        log_text = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            status = f"exit_{proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        out = _to_text(exc.stdout)
        err = _to_text(exc.stderr)
        log_text = out + ("\n" + err if err else "")
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")
    wall_s = time.time() - start

    row: dict[str, object] = {
        "selector_keep": params["ccik_selector_keep"],
        "external_keep": params["ccik_external_keep"],
        "nkeep": params["ccik_nkeep"],
        "status": status,
        "wall_s": f"{wall_s:.2f}",
        "reached_ritz": int(count_ritz_lines(log_text) > 0),
        "ritz_lines": count_ritz_lines(log_text),
        "final_total_eh": "",
        "final_basis": "",
        "abs_err_eh": "",
    }
    final_total = parse_final_total(log_text)
    if final_total is not None:
        row["final_total_eh"] = f"{final_total:.12f}"
    final_basis = parse_final_basis(log_text)
    if final_basis is not None:
        row["final_basis"] = final_basis
    row.update(parse_first_hq_done(log_text))
    return row


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep QP-CCIK hard-point selector settings")
    p.add_argument("--template-ezfio", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--selector-values", type=int, nargs="+", required=True)
    p.add_argument("--external-values", type=int, nargs="*", default=None)
    p.add_argument("--timeout-s", type=int, default=180)
    p.add_argument("--reference-total-eh", type=float, default=-108.774883260748)
    p.add_argument("--ccik-basis-size", type=int, default=16)
    p.add_argument("--ccik-generator-keep", type=int, default=512)
    p.add_argument("--ccik-generator-fraction", type=float, default=0.0)
    p.add_argument("--ccik-nkeep", type=int, default=0)
    p.add_argument("--ccik-total-det-budget", type=int, default=0)
    p.add_argument("--ccik-conv-tol", type=float, default=1e-6)
    p.add_argument("--ccik-max-cycle", type=int, default=6)
    p.add_argument("--ccik-restart-m-cycle", type=int, default=10)
    p.add_argument("--ccik-restart-nroot", type=int, default=5)
    p.add_argument("--ccik-verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.external_values is None or len(args.external_values) == 0:
        pairs = [(v, v) for v in args.selector_values]
    else:
        if len(args.external_values) != len(args.selector_values):
            raise ValueError("external-values must match selector-values length")
        pairs = list(zip(args.selector_values, args.external_values, strict=True))

    rows: list[dict[str, object]] = []
    for selector_keep, external_keep in pairs:
        trial_name = f"sel{selector_keep}_ext{external_keep}"
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        nkeep = args.ccik_nkeep if args.ccik_nkeep > 0 else max(selector_keep, external_keep)
        params = {
            "ccik_basis_size": args.ccik_basis_size,
            "ccik_generator_keep": args.ccik_generator_keep,
            "ccik_generator_fraction": args.ccik_generator_fraction,
            "ccik_selector_keep": selector_keep,
            "ccik_external_keep": external_keep,
            "ccik_nkeep": nkeep,
            "ccik_total_det_budget": args.ccik_total_det_budget,
            "ccik_conv_tol": args.ccik_conv_tol,
            "ccik_max_cycle": args.ccik_max_cycle,
            "ccik_restart_m_cycle": args.ccik_restart_m_cycle,
            "ccik_restart_nroot": args.ccik_restart_nroot,
            "ccik_verbose": str(bool(args.ccik_verbose)).lower(),
        }
        row = run_trial(
            template_ezfio=args.template_ezfio,
            trial_dir=trial_dir,
            params=params,
            timeout_s=args.timeout_s,
        )
        if row.get("final_total_eh"):
            err = abs(float(row["final_total_eh"]) - args.reference_total_eh)
            row["abs_err_eh"] = f"{err:.6e}"
        rows.append(row)
        print(row, flush=True)

    write_results(output_dir / "results.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
