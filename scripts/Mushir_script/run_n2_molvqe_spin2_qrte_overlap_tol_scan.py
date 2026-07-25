#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Scan QRTE generalized-eigenproblem overlap tolerances for MolVQE N2 spin 2.

The default case is the original MolVQE dataset row:
``N2_equi_cc_pVTZ_2``.  The QRTE settings are fixed at dt=0.05 and 100
maximum Krylov iterations, with early stopping at chemical accuracy enabled by
default.  Only ``overlap_tol`` is scanned.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np


CHEMICAL_ACCURACY_HARTREE = 1.6e-3
BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_ID = "N2_equi_cc_pVTZ_2"
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb"
DEFAULT_OUTDIR = (
    BENCHMARK_ROOT
    / "runs"
    / "qrte_natorb_dt0.05_n2_equi_spin2_overlap_tol_scan"
)
DEFAULT_OVERLAP_TOLS = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)

SUMMARY_FIELDNAMES = [
    "case_id",
    "overlap_tol",
    "overlap_tol_label",
    "status",
    "hit_chemical_accuracy",
    "hit_iteration",
    "n_steps_completed",
    "n_steps_requested",
    "stopped_early",
    "final_min_history_energy_hartree",
    "final_error_vs_casscf_hartree",
    "min_error_vs_casscf_hartree",
    "casscf_energy_hartree",
    "delta_t",
    "chemical_accuracy_hartree",
    "summary_path",
    "history_path",
    "error",
]


def _set_thread_limits() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _load_manifest_row(path: Path, case_id: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == case_id:
                return row
    raise SystemExit(f"Case id not found in manifest: {case_id}")


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path)


def _overlap_tol_label(overlap_tol: float) -> str:
    return f"overlap_tol_{overlap_tol:.0e}"


def _completed(summary_path: Path, overlap_tol: float) -> bool:
    if not summary_path.exists():
        return False
    try:
        payload = _load_json(summary_path)
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "completed":
        return False
    return abs(float(payload.get("overlap_tol", -1.0)) - float(overlap_tol)) <= 1e-20


def _first_hit(min_history: np.ndarray, casscf_energy: float) -> int | None:
    for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
        if float(energy) - float(casscf_energy) <= CHEMICAL_ACCURACY_HARTREE:
            return idx
    return None


def _write_history_csv(path: Path, min_history: np.ndarray, casscf_energy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "min_energy_hartree",
                "error_vs_casscf_hartree",
                "within_chemical_accuracy",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
            error = float(energy) - float(casscf_energy)
            writer.writerow(
                {
                    "iteration": idx,
                    "min_energy_hartree": float(energy),
                    "error_vs_casscf_hartree": error,
                    "within_chemical_accuracy": str(error <= CHEMICAL_ACCURACY_HARTREE).lower(),
                }
            )


def _read_history_errors(path: Path) -> list[float]:
    if not path.exists():
        return []
    errors: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            errors.append(float(row["error_vs_casscf_hartree"]))
    return errors


def _summary_csv_row(
    summary_path: Path,
    history_path: Path,
    *,
    case_id: str,
    overlap_tol: float,
    casscf_energy: float,
    delta_t: float,
) -> dict[str, Any]:
    label = _overlap_tol_label(overlap_tol)
    if not summary_path.exists():
        return {
            "case_id": case_id,
            "overlap_tol": overlap_tol,
            "overlap_tol_label": label,
            "status": "missing",
            "hit_chemical_accuracy": "",
            "hit_iteration": "",
            "n_steps_completed": "",
            "n_steps_requested": "",
            "stopped_early": "",
            "final_min_history_energy_hartree": "",
            "final_error_vs_casscf_hartree": "",
            "min_error_vs_casscf_hartree": "",
            "casscf_energy_hartree": casscf_energy,
            "delta_t": delta_t,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "summary_path": _relative(summary_path),
            "history_path": _relative(history_path),
            "error": "",
        }

    summary = _load_json(summary_path)
    errors = _read_history_errors(history_path)
    final_energy = summary.get("final_min_history_energy_hartree")
    final_error = "" if final_energy is None else float(final_energy) - float(casscf_energy)
    min_error = "" if not errors else min(errors)
    return {
        "case_id": case_id,
        "overlap_tol": overlap_tol,
        "overlap_tol_label": label,
        "status": summary.get("status", ""),
        "hit_chemical_accuracy": summary.get("hit_chemical_accuracy", ""),
        "hit_iteration": "" if summary.get("hit_iteration") is None else summary.get("hit_iteration"),
        "n_steps_completed": summary.get("n_steps_completed", ""),
        "n_steps_requested": summary.get("n_steps_requested", ""),
        "stopped_early": summary.get("stopped_early", ""),
        "final_min_history_energy_hartree": "" if final_energy is None else final_energy,
        "final_error_vs_casscf_hartree": final_error,
        "min_error_vs_casscf_hartree": min_error,
        "casscf_energy_hartree": summary.get("casscf_energy_hartree", casscf_energy),
        "delta_t": summary.get("delta_t", delta_t),
        "chemical_accuracy_hartree": summary.get(
            "chemical_accuracy_hartree",
            CHEMICAL_ACCURACY_HARTREE,
        ),
        "summary_path": _relative(summary_path),
        "history_path": _relative(history_path),
        "error": summary.get("error", ""),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _import_qcant(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))
    import QCANT  # noqa: PLC0415

    return QCANT


def _run_overlap_tol(
    row: dict[str, str],
    overlap_tol: float,
    args: argparse.Namespace,
    QCANT,
) -> str:
    case_id = row["case_id"]
    label = _overlap_tol_label(overlap_tol)
    case_dir = args.outdir / case_id / label
    summary_path = case_dir / "summary.json"
    history_path = case_dir / "min_energy_history.csv"

    if _completed(summary_path, overlap_tol) and not args.force:
        print(f"SKIP completed {case_id} overlap_tol={overlap_tol:.0e}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    symbols = _json_loads(row["symbols"])
    geometry_angstrom = np.asarray(_json_loads(row["geometry_angstrom"]), dtype=float)
    casscf_energy = float(row["casscf_energy_hartree"])
    early_stop_energy = (
        casscf_energy + CHEMICAL_ACCURACY_HARTREE
        if args.early_stop_chemical_accuracy
        else None
    )
    start = time.perf_counter()

    input_snapshot = {
        "case_id": case_id,
        "source_name": row["source_name"],
        "source_json_path": row["source_json_path"],
        "symbols": symbols,
        "geometry_bohr": _json_loads(row["geometry_bohr"]),
        "geometry_angstrom": geometry_angstrom.tolist(),
        "basis": row["basis"],
        "charge": int(row["charge"]),
        "spin": int(row["spin"]),
        "active_electrons": int(row["active_electrons"]),
        "active_orbitals": int(row["active_orbitals"]),
        "casscf_energy_hartree": casscf_energy,
        "h0.25": float(row["h0.25"]),
        "reference_source": "MolVQE manifest CASSCF energy",
    }
    run_config = {
        "case_id": case_id,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "delta_t": float(args.delta_t),
        "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
        "early_stop_energy_hartree": early_stop_energy,
        "evolution_mode": args.evolution_mode,
        "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
        "hamiltonian_source": args.hamiltonian_source,
        "max_workers": int(args.max_workers),
        "method": "qrte",
        "n_steps": int(args.n_steps),
        "overlap_tol": float(overlap_tol),
        "overlap_tol_label": label,
        "qcant_root": str(args.qcant_root.resolve()),
        "reference_source": "MolVQE manifest CASSCF energy",
        "trotter_steps": int(args.trotter_steps),
    }
    _write_json(case_dir / "input_snapshot.json", input_snapshot)
    _write_json(case_dir / "run_config.json", run_config)

    try:
        energies, _basis_states, _times, min_history = QCANT.qrte_qulacs(
            symbols=symbols,
            geometry=geometry_angstrom,
            delta_t=float(args.delta_t),
            n_steps=int(args.n_steps),
            active_electrons=int(row["active_electrons"]),
            active_orbitals=int(row["active_orbitals"]),
            basis=row["basis"],
            charge=int(row["charge"]),
            spin=int(row["spin"]),
            overlap_tol=float(overlap_tol),
            return_min_energy_history=True,
            evolution_mode=args.evolution_mode,
            trotter_steps=int(args.trotter_steps),
            max_workers=int(args.max_workers),
            early_stop_energy=early_stop_energy,
            hamiltonian_cache_dir=str(args.hamiltonian_cache_dir),
            hamiltonian_source=args.hamiltonian_source,
        )

        energies = np.asarray(energies, dtype=float)
        min_history = np.asarray(min_history, dtype=float)
        hit_iteration = _first_hit(min_history, casscf_energy)
        n_steps_completed = int(min_history.size)
        stopped_early = bool(
            args.early_stop_chemical_accuracy
            and hit_iteration is not None
            and n_steps_completed < int(args.n_steps)
        )
        _write_history_csv(history_path, min_history, casscf_energy)

        summary = {
            "status": "completed",
            "case_id": case_id,
            "method": "qrte",
            "reference_source": "MolVQE manifest CASSCF energy",
            "casscf_energy_hartree": casscf_energy,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(args.delta_t),
            "overlap_tol": float(overlap_tol),
            "overlap_tol_label": label,
            "hit_iteration": hit_iteration,
            "hit_chemical_accuracy": hit_iteration is not None,
            "n_steps": n_steps_completed,
            "n_steps_requested": int(args.n_steps),
            "n_steps_completed": n_steps_completed,
            "stopped_early": stopped_early,
            "early_stop_energy_hartree": early_stop_energy,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": args.hamiltonian_source,
            "basis": row["basis"],
            "charge": int(row["charge"]),
            "spin": int(row["spin"]),
            "active_electrons": int(row["active_electrons"]),
            "active_orbitals": int(row["active_orbitals"]),
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(
            f"DONE {case_id} overlap_tol={overlap_tol:.0e} hit={hit_iteration}",
            flush=True,
        )
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case_id,
            "method": "qrte",
            "reference_source": "MolVQE manifest CASSCF energy",
            "casscf_energy_hartree": casscf_energy,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(args.delta_t),
            "overlap_tol": float(overlap_tol),
            "overlap_tol_label": label,
            "n_steps_requested": int(args.n_steps),
            "early_stop_energy_hartree": early_stop_energy,
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": args.hamiltonian_source,
            "basis": row["basis"],
            "charge": int(row["charge"]),
            "spin": int(row["spin"]),
            "active_electrons": int(row["active_electrons"]),
            "active_orbitals": int(row["active_orbitals"]),
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL {case_id} overlap_tol={overlap_tol:.0e}: {exc!r}", flush=True)
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument(
        "--hamiltonian-source",
        choices=("casci", "casscf", "casscf_natorb", "casscf-natural", "molecular"),
        default="casscf_natorb",
    )
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument(
        "--overlap-tol",
        action="append",
        type=float,
        default=None,
        help="Tolerance to scan. Repeat to override the default 1e-6..1e-10 grid.",
    )
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--no-early-stop-chemical-accuracy",
        action="store_false",
        dest="early_stop_chemical_accuracy",
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    overlap_tols = list(args.overlap_tol or DEFAULT_OVERLAP_TOLS)
    if any(value <= 0 for value in overlap_tols):
        raise SystemExit("All --overlap-tol values must be positive")

    summary_csv = args.summary_csv or (args.outdir / "overlap_tol_scan_summary.csv")
    row = _load_manifest_row(args.manifest, args.case_id)
    if args.case_id == DEFAULT_CASE_ID and int(row["spin"]) != 2:
        raise SystemExit(f"{DEFAULT_CASE_ID} manifest row unexpectedly has spin={row['spin']}")

    if args.dry_run:
        print(f"case_id={row['case_id']}")
        print(f"delta_t={float(args.delta_t):g}")
        print(f"n_steps={int(args.n_steps)}")
        print(f"early_stop_chemical_accuracy={bool(args.early_stop_chemical_accuracy)}")
        print("overlap_tols=" + ",".join(f"{value:.0e}" for value in overlap_tols))
        print(f"outdir={args.outdir}")
        print(f"summary_csv={summary_csv}")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)

    QCANT = _import_qcant(args.qcant_root)
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    csv_rows: list[dict[str, Any]] = []
    casscf_energy = float(row["casscf_energy_hartree"])
    for overlap_tol in overlap_tols:
        status = _run_overlap_tol(row, float(overlap_tol), args, QCANT)
        counts[status] = counts.get(status, 0) + 1
        label = _overlap_tol_label(float(overlap_tol))
        case_dir = args.outdir / row["case_id"] / label
        csv_rows.append(
            _summary_csv_row(
                case_dir / "summary.json",
                case_dir / "min_energy_history.csv",
                case_id=row["case_id"],
                overlap_tol=float(overlap_tol),
                casscf_energy=casscf_energy,
                delta_t=float(args.delta_t),
            )
        )
        _write_summary_csv(summary_csv, csv_rows)

    print(json.dumps(counts, sort_keys=True), flush=True)
    print(f"Wrote {_relative(summary_csv)}", flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
