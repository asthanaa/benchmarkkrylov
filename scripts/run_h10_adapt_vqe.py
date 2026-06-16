#!/usr/bin/env python
"""Run the missing H10 ADAPT-VQE point with QCANT/Qulacs.

This driver is intentionally case-specific: it uses the canonical H10 row from
the MolVQE-21 Fig. 1b manifest, streams every ADAPT iteration to disk, and stops
as soon as chemical accuracy relative to the stored CASSCF energy is reached.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

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
CASE_ID = "H10_near_equi_sto3g_0"
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "adapt_h10_qcant"

HISTORY_FIELDNAMES = [
    "iteration",
    "energy_hartree",
    "error_vs_casscf_hartree",
    "within_chemical_accuracy",
    "adapt_max_gradient",
    "selected_pool_index",
    "selected_excitation",
    "optimizer_method",
    "optimizer_nit",
    "optimizer_success",
    "optimizer_grad_norm",
]


class ReachedChemicalAccuracy(RuntimeError):
    """Raised internally after the callback records the first H10 hit."""


class InterruptedRun(RuntimeError):
    """Raised when Slurm sends an interrupt/terminate signal."""


def _set_thread_limits() -> None:
    for thread_var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(thread_var, "1")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), sort_keys=True))
        handle.write("\n")


def _append_history_csv(path: Path, snapshot: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "iteration": int(snapshot["iteration"]),
                "energy_hartree": float(snapshot["energy"]),
                "error_vs_casscf_hartree": float(snapshot["error_vs_casscf_hartree"]),
                "within_chemical_accuracy": str(bool(snapshot["within_chemical_accuracy"])).lower(),
                "adapt_max_gradient": float(snapshot["adapt_max_gradient"]),
                "selected_pool_index": int(snapshot["selected_pool_index"]),
                "selected_excitation": json.dumps(snapshot["selected_excitation"], separators=(",", ":")),
                "optimizer_method": str(snapshot["optimizer_method"]),
                "optimizer_nit": int(snapshot["optimizer_nit"]),
                "optimizer_success": str(bool(snapshot["optimizer_success"])).lower(),
                "optimizer_grad_norm": float(snapshot["optimizer_grad_norm"]),
            }
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _load_manifest_row(path: Path, case_id: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == case_id:
                return row
    raise SystemExit(f"Could not find {case_id!r} in {path}")


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def _archive_incomplete_outputs(outdir: Path) -> Path | None:
    candidates = [
        "iteration_history.csv",
        "iteration_history.jsonl",
        "params.json",
        "params.npy",
        "progress.json",
        "selected_excitations.json",
        "summary.json",
        "traceback.txt",
    ]
    existing = [outdir / name for name in candidates if (outdir / name).exists()]
    if not existing:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = outdir / "incomplete_backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.replace(backup_dir / path.name)
    return backup_dir


def _import_qcant(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))
    try:
        import qulacs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Qulacs is required for this ADAPT-VQE run; install QCANT with the [qulacs] extra.") from exc
    import QCANT

    return QCANT


def _prepare_inputs(row: dict[str, str]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "source_name": row["source_name"],
        "source_json_path": row["source_json_path"],
        "symbols": _json_loads(row["symbols"]),
        "geometry_bohr": _json_loads(row["geometry_bohr"]),
        "geometry_angstrom": _json_loads(row["geometry_angstrom"]),
        "basis": row["basis"],
        "charge": int(row["charge"]),
        "spin": int(row["spin"]),
        "active_electrons": int(row["active_electrons"]),
        "active_orbitals": int(row["active_orbitals"]),
        "casscf_energy_hartree": float(row["casscf_energy_hartree"]),
        "h0.25": float(row["h0.25"]),
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "target_energy_hartree": float(row["casscf_energy_hartree"]) + CHEMICAL_ACCURACY_HARTREE,
        "manifest_adapt_chem_accuracy_iteration": row["adapt_chem_accuracy_iteration"] or None,
        "manifest_has_adapt_chem_hit": row["has_adapt_chem_hit"] == "true",
    }


def _write_final_artifacts(
    outdir: Path,
    snapshots: list[dict[str, Any]],
    params: Any,
    excitations: Any,
) -> None:
    if params is None and snapshots:
        params = snapshots[-1].get("params")
    if excitations is None and snapshots:
        excitations = snapshots[-1].get("ash_excitation")

    if params is not None:
        params_array = np.asarray(params, dtype=float)
        np.save(outdir / "params.npy", params_array)
        _write_json(outdir / "params.json", {"params": params_array.tolist()})
    if excitations is not None:
        _write_json(outdir / "selected_excitations.json", {"ash_excitation": excitations})


def _make_summary(
    *,
    status: str,
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
    snapshots: list[dict[str, Any]],
    start: float,
    error: str | None = None,
) -> dict[str, Any]:
    hit_snapshot = next((item for item in snapshots if item.get("within_chemical_accuracy")), None)
    final_snapshot = snapshots[-1] if snapshots else None
    final_energy = None if final_snapshot is None else float(final_snapshot["energy"])
    final_error = None if final_snapshot is None else float(final_snapshot["error_vs_casscf_hartree"])

    summary: dict[str, Any] = {
        "status": status,
        "case_id": inputs["case_id"],
        "method": "adapt_vqe_qulacs",
        "h0.25": float(inputs["h0.25"]),
        "casscf_energy_hartree": float(inputs["casscf_energy_hartree"]),
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "target_energy_hartree": float(inputs["target_energy_hartree"]),
        "hit_iteration": None if hit_snapshot is None else int(hit_snapshot["iteration"]),
        "hit_chemical_accuracy": hit_snapshot is not None,
        "n_iterations": len(snapshots),
        "n_iterations_completed": len(snapshots),
        "n_iterations_requested": int(config["adapt_it"]),
        "stopped_early": hit_snapshot is not None and len(snapshots) < int(config["adapt_it"]),
        "final_energy_hartree": final_energy,
        "final_error_vs_casscf_hartree": final_error,
        "elapsed_seconds": float(time.perf_counter() - start),
        "max_workers": int(config["max_workers"]),
        "pool_type": str(config["pool_type"]),
        "pool_sample_size": config["pool_sample_size"],
        "pool_seed": config["pool_seed"],
        "parallel_gradients": bool(config["parallel_gradients"]),
        "gradient_chunk_size": config["gradient_chunk_size"],
        "optimizer_method": str(config["optimizer_method"]),
        "optimizer_maxiter": int(config["optimizer_maxiter"]),
        "hamiltonian_source": str(config["hamiltonian_source"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
    }
    if error is not None:
        summary["error"] = error
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--case-id", default=CASE_ID)
    parser.add_argument("--adapt-it", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--pool-type", default="fermionic_sd")
    parser.add_argument("--pool-sample-size", type=int, default=None)
    parser.add_argument("--pool-seed", type=int, default=None)
    parser.add_argument("--gradient-chunk-size", type=int, default=None)
    parser.add_argument("--optimizer-method", default="BFGS")
    parser.add_argument("--optimizer-maxiter", type=int, default=500)
    parser.add_argument("--hamiltonian-cutoff", type=float, default=1e-20)
    parser.add_argument("--hamiltonian-source", default="casci")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    if args.adapt_it <= 0:
        raise SystemExit("--adapt-it must be positive")

    outdir = args.outdir.resolve()
    summary_path = outdir / "summary.json"
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed ADAPT {args.case_id} at {summary_path}", flush=True)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    backup_dir = _archive_incomplete_outputs(outdir)
    if backup_dir is not None:
        print(f"Archived previous ADAPT {args.case_id} outputs to {backup_dir}", flush=True)

    row = _load_manifest_row(args.manifest, args.case_id)
    inputs = _prepare_inputs(row)
    config = {
        "method": "adapt_vqe_qulacs",
        "adapt_it": int(args.adapt_it),
        "max_workers": int(args.max_workers),
        "pool_type": str(args.pool_type),
        "pool_sample_size": args.pool_sample_size,
        "pool_seed": args.pool_seed,
        "parallel_gradients": True,
        "gradient_chunk_size": args.gradient_chunk_size,
        "optimizer_method": str(args.optimizer_method),
        "optimizer_maxiter": int(args.optimizer_maxiter),
        "hamiltonian_cutoff": float(args.hamiltonian_cutoff),
        "hamiltonian_source": str(args.hamiltonian_source),
        "early_stop_chemical_accuracy": True,
        "early_stop_energy_hartree": float(inputs["target_energy_hartree"]),
        "qcant_root": str(args.qcant_root.resolve()),
        "manifest": str(args.manifest.resolve()),
    }
    _write_json(outdir / "input_snapshot.json", inputs)
    _write_json(outdir / "run_config.json", config)

    interrupted_signal: dict[str, int | None] = {"signum": None}

    def _signal_handler(signum: int, _frame: Any) -> None:
        interrupted_signal["signum"] = int(signum)
        raise InterruptedRun(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    QCANT = _import_qcant(args.qcant_root)
    snapshots: list[dict[str, Any]] = []
    start = time.perf_counter()

    def _iteration_callback(snapshot: Mapping[str, Any]) -> None:
        plain = _to_jsonable(snapshot)
        energy = float(plain["energy"])
        error = energy - float(inputs["casscf_energy_hartree"])
        plain["error_vs_casscf_hartree"] = error
        plain["within_chemical_accuracy"] = error <= CHEMICAL_ACCURACY_HARTREE
        plain["target_energy_hartree"] = float(inputs["target_energy_hartree"])
        snapshots.append(plain)
        _append_jsonl(outdir / "iteration_history.jsonl", plain)
        _append_history_csv(outdir / "iteration_history.csv", plain)
        _write_json(
            outdir / "progress.json",
            _make_summary(status="running", inputs=inputs, config=config, snapshots=snapshots, start=start),
        )
        print(
            "ITER "
            f"{plain['iteration']} "
            f"E={energy:.12f} "
            f"err={error:.6e} "
            f"hit={plain['within_chemical_accuracy']}",
            flush=True,
        )
        if plain["within_chemical_accuracy"]:
            raise ReachedChemicalAccuracy(f"{args.case_id} ADAPT-VQE reached chemical accuracy")

    params = None
    excitations = None
    try:
        params, excitations, _energies, history = QCANT.adapt_vqe_qulacs(
            symbols=inputs["symbols"],
            geometry=np.asarray(inputs["geometry_angstrom"], dtype=float),
            adapt_it=int(args.adapt_it),
            basis=str(inputs["basis"]),
            charge=int(inputs["charge"]),
            spin=int(inputs["spin"]),
            active_electrons=int(inputs["active_electrons"]),
            active_orbitals=int(inputs["active_orbitals"]),
            hamiltonian_cutoff=float(args.hamiltonian_cutoff),
            hamiltonian_source=str(args.hamiltonian_source),
            pool_type=str(args.pool_type),
            pool_sample_size=args.pool_sample_size,
            pool_seed=args.pool_seed,
            parallel_gradients=True,
            max_workers=int(args.max_workers),
            gradient_chunk_size=args.gradient_chunk_size,
            optimizer_method=str(args.optimizer_method),
            optimizer_maxiter=int(args.optimizer_maxiter),
            return_history=True,
            iteration_callback=_iteration_callback,
        )
        if not snapshots:
            for snapshot in history:
                _iteration_callback(snapshot)
        _write_final_artifacts(outdir, snapshots, params, excitations)
        summary = _make_summary(status="completed", inputs=inputs, config=config, snapshots=snapshots, start=start)
        _write_json(summary_path, summary)
        print(f"DONE ADAPT {args.case_id} hit={summary['hit_iteration']}", flush=True)
    except ReachedChemicalAccuracy:
        _write_final_artifacts(outdir, snapshots, params, excitations)
        summary = _make_summary(status="completed", inputs=inputs, config=config, snapshots=snapshots, start=start)
        _write_json(summary_path, summary)
        print(f"DONE ADAPT {args.case_id} hit={summary['hit_iteration']}", flush=True)
    except InterruptedRun as exc:
        _write_final_artifacts(outdir, snapshots, params, excitations)
        summary = _make_summary(status="interrupted", inputs=inputs, config=config, snapshots=snapshots, start=start, error=str(exc))
        summary["signal"] = interrupted_signal["signum"]
        _write_json(summary_path, summary)
        raise SystemExit(130) from exc
    except Exception as exc:  # noqa: BLE001
        _write_final_artifacts(outdir, snapshots, params, excitations)
        tb = traceback.format_exc()
        (outdir / "traceback.txt").write_text(tb, encoding="utf-8")
        summary = _make_summary(status="failed", inputs=inputs, config=config, snapshots=snapshots, start=start, error=repr(exc))
        summary["traceback_path"] = str(outdir / "traceback.txt")
        _write_json(summary_path, summary)
        print(f"FAIL ADAPT {args.case_id}: {exc!r}", flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
