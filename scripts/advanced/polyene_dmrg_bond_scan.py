#!/usr/bin/env python3
"""Run a resumable DMRG bond-dimension scan for the polyene reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from reference_backends import load_or_compute_dmrg_reference_from_fcidump  # noqa: E402


DEFAULT_DMRG_PYTHON = ROOT / ".conda" / "cka-dmrg-py312" / "bin" / "python"
DEFAULT_BOND_DIMS = (2000, 4000, 6000, 8000, 10000, 12000)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer bond dimension")
    return values


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one floating-point value")
    return values


def _row_from_summary(*, summary: dict[str, Any], max_bond_dim: int, source: str, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "max_bond_dim": int(max_bond_dim),
        "bond_dims": ",".join(str(int(x)) for x in summary.get("bond_dims", [max_bond_dim])),
        "total_energy_eh": None if summary.get("total_energy_eh") is None else float(summary["total_energy_eh"]),
        "electronic_energy_eh": (
            None if summary.get("electronic_energy_eh") is None else float(summary["electronic_energy_eh"])
        ),
        "ecore_eh": None if summary.get("ecore_eh") is None else float(summary["ecore_eh"]),
        "wall_s": None if summary.get("wall_s") is None else float(summary["wall_s"]),
        "largest_file_mb": None if summary.get("largest_file_mb") is None else float(summary["largest_file_mb"]),
        "peak_total_scratch_gb": (
            None if summary.get("peak_total_scratch_gb") is None else float(summary["peak_total_scratch_gb"])
        ),
        "log_path": summary.get("log_path"),
        "summary_path": summary.get("summary_path"),
        "failure": summary.get("failure"),
        "stop_reason": summary.get("stop_reason"),
        "delta_vs_previous_mHa": None,
        "delta_vs_m1000_mHa": None,
    }


def _update_deltas(rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row.get("status") == "success" and row.get("total_energy_eh") is not None]
    baseline = next((row for row in successful if int(row["max_bond_dim"]) == 1000), None)
    previous: dict[str, Any] | None = None
    for row in successful:
        if previous is not None:
            row["delta_vs_previous_mHa"] = 1000.0 * (
                float(row["total_energy_eh"]) - float(previous["total_energy_eh"])
            )
        if baseline is not None:
            row["delta_vs_m1000_mHa"] = 1000.0 * (
                float(row["total_energy_eh"]) - float(baseline["total_energy_eh"])
            )
        previous = row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "source",
        "max_bond_dim",
        "bond_dims",
        "total_energy_eh",
        "electronic_energy_eh",
        "ecore_eh",
        "delta_vs_previous_mHa",
        "delta_vs_m1000_mHa",
        "wall_s",
        "largest_file_mb",
        "peak_total_scratch_gb",
        "log_path",
        "summary_path",
        "failure",
        "stop_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path_pdf: Path, path_png: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "success" and row.get("total_energy_eh") is not None]
    if not ok_rows:
        return
    x = [int(row["max_bond_dim"]) for row in ok_rows]
    e = [float(row["total_energy_eh"]) for row in ok_rows]
    deltas = [
        None if row.get("delta_vs_previous_mHa") is None else abs(float(row["delta_vs_previous_mHa"]))
        for row in ok_rows
    ]
    delta_x = [xx for xx, yy in zip(x, deltas) if yy is not None and yy > 0.0]
    delta_y = [yy for yy in deltas if yy is not None and yy > 0.0]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.2), constrained_layout=True)
    axes[0].plot(x, e, color="#1F4E79", marker="o", linewidth=1.6, markersize=4.0)
    axes[0].set_xlabel("Maximum DMRG bond dimension")
    axes[0].set_ylabel("Total energy (Eh)")
    axes[0].grid(True, alpha=0.25)

    if delta_x:
        axes[1].plot(delta_x, delta_y, color="#A23B00", marker="s", linewidth=1.6, markersize=4.0)
        axes[1].set_yscale("log")
    axes[1].set_xlabel("Maximum DMRG bond dimension")
    axes[1].set_ylabel("|Energy change from previous| (mHa)")
    axes[1].grid(True, which="both", alpha=0.25)

    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=350)
    plt.close(fig)


def _write_outputs(outdir: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    _update_deltas(rows)
    _write_csv(outdir / "dmrg_bond_scan.csv", rows)
    (outdir / "dmrg_bond_scan.json").write_text(
        json.dumps({**payload, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_plot(outdir / "dmrg_bond_convergence.pdf", outdir / "dmrg_bond_convergence.png", rows)


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fcidump = args.fcidump.resolve()
    baseline_summary = _load_json(args.baseline_summary.resolve())
    rows: list[dict[str, Any]] = [
        _row_from_summary(summary=baseline_summary, max_bond_dim=1000, source="existing_m1000", status="success")
    ]
    payload: dict[str, Any] = {
        "fcidump_path": str(fcidump),
        "baseline_summary_path": str(args.baseline_summary.resolve()),
        "memory_gb": int(args.memory_gb),
        "threads": int(args.threads),
        "conv_tol": float(args.conv_tol),
        "n_steps": int(args.n_steps),
        "max_file_mb": int(args.max_file_mb),
        "max_total_scratch_gb": float(args.max_total_scratch_gb),
        "scratch_root": str(args.scratch_root.resolve()),
        "requested_bond_dims": [int(x) for x in args.bond_dims],
    }
    _write_outputs(outdir, rows, payload)

    for max_bond_dim in args.bond_dims:
        run_dir = outdir / "dmrg_runs" / f"M{int(max_bond_dim):05d}"
        schedule = tuple(int(max_bond_dim) for _ in range(int(args.schedule_repeats)))
        noises = tuple(float(x) for x in args.noises[: len(schedule)])
        dav_thrds = tuple(float(x) for x in args.dav_thrds[: len(schedule)])
        summary = load_or_compute_dmrg_reference_from_fcidump(
            fcidump_path=fcidump,
            outdir=run_dir,
            force=bool(args.force),
            memory_gb=int(args.memory_gb),
            threads=int(args.threads),
            conv_tol=float(args.conv_tol),
            dmrg_python=args.dmrg_python.resolve(),
            max_file_mb=int(args.max_file_mb),
            max_total_scratch_gb=float(args.max_total_scratch_gb),
            scratch_root=args.scratch_root.resolve(),
            bond_dims=schedule,
            noises=noises,
            dav_thrds=dav_thrds,
            n_steps=int(args.n_steps),
        )
        summary["summary_path"] = str(run_dir / "reference_dmrg_summary.json")
        status = "success" if bool(summary.get("success")) else "failed"
        rows.append(
            _row_from_summary(summary=summary, max_bond_dim=int(max_bond_dim), source="computed", status=status)
        )
        _write_outputs(outdir, rows, payload)
        if status != "success":
            break
    return {**payload, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcidump", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bond-dims", type=_parse_int_list, default=DEFAULT_BOND_DIMS)
    parser.add_argument("--memory-gb", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=36)
    parser.add_argument("--conv-tol", type=float, default=1.0e-9)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--schedule-repeats", type=int, default=4)
    parser.add_argument("--max-file-mb", type=int, default=131072)
    parser.add_argument("--max-total-scratch-gb", type=float, default=900.0)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp/cka_dmrg"))
    parser.add_argument("--dmrg-python", type=Path, default=DEFAULT_DMRG_PYTHON)
    parser.add_argument(
        "--noises",
        type=_parse_float_list,
        default=None,
        help="Comma-separated DMRG noise schedule; defaults to 1e-5,1e-6,1e-8,0",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.noises is None:
        args.noises = (1.0e-5, 1.0e-6, 1.0e-8, 0.0)
    args.dav_thrds = (1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9)
    if len(args.noises) < int(args.schedule_repeats):
        raise SystemExit("--noises schedule is shorter than --schedule-repeats")
    payload = run_scan(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
