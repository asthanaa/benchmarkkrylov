#!/usr/bin/env python
"""Pack many restartable QCANT benchmark cases into one Slurm allocation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
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


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb"
DRIVER = BENCHMARK_ROOT / "scripts" / "run_krylov_benchmark.py"

_STOP_REQUESTED = False
_RUNNING: list["RunningCase"] = []


@dataclass
class RunningCase:
    row_index: int
    case_id: str
    process: subprocess.Popen[str]
    stdout_handle: Any
    stderr_handle: Any
    started_at: float
    summary_path: Path


def _set_thread_limits() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for idx, row in enumerate(rows):
        row["_manifest_index"] = str(idx)
    return rows


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def _case_dir(row: dict[str, str], outdir: Path) -> Path:
    return outdir / row["case_id"]


def _summary_path(row: dict[str, str], outdir: Path) -> Path:
    return _case_dir(row, outdir) / "summary.json"


def _basis_weight(basis: str) -> int:
    normalized = basis.lower()
    if "sto" in normalized and "3g" in normalized:
        return 1
    if "6-31" in normalized:
        return 2
    if "cc-pvdz" in normalized:
        return 3
    if "cc-pvtz" in normalized:
        return 4
    if "def2-tzvp" in normalized:
        return 5
    return 4


def _small_first_key(row: dict[str, str]) -> tuple[float, int, int, int, str]:
    symbols = _json_loads(row["symbols"])
    n_atoms = len(symbols)
    active_orbitals = int(row["active_orbitals"])
    active_electrons = int(row["active_electrons"])
    size_score = (
        n_atoms * _basis_weight(row["basis"])
        + 4 * active_orbitals
        + 2 * active_electrons
    )
    return (size_score, n_atoms, active_orbitals, active_electrons, row["case_id"])


def _select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = rows
    if args.row_index is not None:
        allowed_indexes = set(args.row_index)
        selected = [row for row in selected if int(row["_manifest_index"]) in allowed_indexes]
    if args.case_id:
        allowed_cases = set(args.case_id)
        selected = [row for row in selected if row["case_id"] in allowed_cases]
    if args.exclude_case_id:
        excluded_cases = set(args.exclude_case_id)
        selected = [row for row in selected if row["case_id"] not in excluded_cases]
    if args.order == "small-first":
        selected = sorted(selected, key=_small_first_key)
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    return selected


def _read_summary_status(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return "unreadable"
    status = payload.get("status")
    return str(status) if status is not None else None


def _build_child_command(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(DRIVER),
        "--method",
        args.method,
        "--manifest",
        str(args.manifest),
        "--outdir",
        str(args.outdir),
        "--qcant-root",
        str(args.qcant_root),
        "--row-index",
        row["_manifest_index"],
        "--n-steps",
        str(args.n_steps),
        "--delta-t",
        str(args.delta_t),
        "--epsilon",
        str(args.epsilon),
        "--evolution-mode",
        args.evolution_mode,
        "--trotter-steps",
        str(args.trotter_steps),
        "--max-workers",
        str(args.max_workers),
        "--hamiltonian-source",
        args.hamiltonian_source,
    ]
    if args.hamiltonian_cache_dir is not None:
        cmd.extend(["--hamiltonian-cache-dir", str(args.hamiltonian_cache_dir)])
    if args.early_stop_chemical_accuracy:
        cmd.append("--early-stop-chemical-accuracy")
    if args.force:
        cmd.append("--force")
    return cmd


def _start_case(row: dict[str, str], args: argparse.Namespace) -> RunningCase:
    case_id = row["case_id"]
    case_dir = _case_dir(row, args.outdir)
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "summary.json"
    cmd = _build_child_command(row, args)
    launch_record = {
        "case_id": case_id,
        "command": cmd,
        "launcher": "run_krylov_pool.py",
        "manifest_index": int(row["_manifest_index"]),
        "method": args.method,
        "pool_concurrency": int(args.concurrency),
        "pool_id": args.pool_id,
        "case_max_workers": int(args.max_workers),
        "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
        "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
        "hamiltonian_source": args.hamiltonian_source,
        "started_at_epoch": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _write_json(case_dir / "pool_launch.json", launch_record)

    stdout_handle = (case_dir / "pool_stdout.log").open("w", encoding="utf-8")
    stderr_handle = (case_dir / "pool_stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        cwd=str(BENCHMARK_ROOT),
    )
    running = RunningCase(
        row_index=int(row["_manifest_index"]),
        case_id=case_id,
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        started_at=time.perf_counter(),
        summary_path=summary_path,
    )
    _RUNNING.append(running)
    print(f"START {args.method} row={running.row_index} case={case_id} pid={process.pid}", flush=True)
    return running


def _finish_case(running: RunningCase, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    rc = running.process.returncode
    running.stdout_handle.close()
    running.stderr_handle.close()
    if running in _RUNNING:
        _RUNNING.remove(running)

    elapsed = time.perf_counter() - running.started_at
    status = _read_summary_status(running.summary_path)
    if rc == 0 and status == "completed":
        outcome = "completed"
    elif rc == 0 and status is None:
        outcome = "unknown"
    elif status == "completed":
        outcome = "completed"
    elif status == "failed":
        outcome = "failed"
    else:
        outcome = "failed"

    event = {
        "case_id": running.case_id,
        "elapsed_seconds": elapsed,
        "manifest_index": running.row_index,
        "outcome": outcome,
        "returncode": rc,
        "summary_status": status,
    }
    print(
        f"END {args.method} row={running.row_index} case={running.case_id} "
        f"outcome={outcome} rc={rc} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return outcome, event


def _terminate_children(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Signal {signum} received; terminating {len(_RUNNING)} active cases.", flush=True)
    for running in list(_RUNNING):
        if running.process.poll() is None:
            running.process.terminate()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at_epoch"] = time.time()
    _write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qrte", "qkud"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--exclude-case-id", action="append", default=None)
    parser.add_argument("--row-index", action="append", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1, help="QCANT workers per molecule.")
    parser.add_argument("--concurrency", type=int, default=1, help="Molecules to run at once inside this job.")
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument(
        "--hamiltonian-source",
        choices=("casci", "casscf", "casscf_natorb", "casscf-natural", "molecular"),
        default="casscf_natorb",
    )
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--order", choices=("manifest", "small-first"), default="small-first")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pool-id", default=os.environ.get("SLURM_JOB_ID"))
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    signal.signal(signal.SIGTERM, _terminate_children)
    signal.signal(signal.SIGINT, _terminate_children)

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.hamiltonian_cache_dir is not None:
        args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)
    if args.pool_id is None:
        args.pool_id = f"local-{int(time.time())}-{os.getpid()}"

    rows = _select_rows(_load_manifest(args.manifest), args)
    pending = [row for row in rows if args.force or not _completed(_summary_path(row, args.outdir))]
    skipped = len(rows) - len(pending)
    if not rows:
        raise SystemExit("No manifest rows selected")

    requested_slots = int(args.concurrency) * int(args.max_workers)
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or "0")
    if slurm_cpus and requested_slots > slurm_cpus:
        print(
            f"WARNING requested {requested_slots} worker slots but SLURM_CPUS_PER_TASK={slurm_cpus}",
            flush=True,
        )

    progress_path = args.outdir / f"pool_progress_{args.pool_id}.json"
    progress: dict[str, Any] = {
        "completed": 0,
        "concurrency": int(args.concurrency),
        "events": [],
        "failed": 0,
        "force": bool(args.force),
        "excluded_case_ids": [] if args.exclude_case_id is None else list(args.exclude_case_id),
        "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
        "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
        "hamiltonian_source": args.hamiltonian_source,
        "max_workers": int(args.max_workers),
        "method": args.method,
        "n_selected": len(rows),
        "n_pending_at_start": len(pending),
        "order": args.order,
        "pool_id": args.pool_id,
        "skipped_completed_at_start": skipped,
        "started_at_epoch": time.time(),
    }
    _write_progress(progress_path, progress)

    print(
        f"POOL {args.method} selected={len(rows)} pending={len(pending)} "
        f"skipped={skipped} concurrency={args.concurrency} max_workers={args.max_workers}",
        flush=True,
    )

    running_cases: list[RunningCase] = []
    while (pending or running_cases) and not _STOP_REQUESTED:
        while pending and len(running_cases) < int(args.concurrency) and not _STOP_REQUESTED:
            row = pending.pop(0)
            if not args.force and _completed(_summary_path(row, args.outdir)):
                progress["skipped_completed_at_launch"] = progress.get("skipped_completed_at_launch", 0) + 1
                continue
            running_cases.append(_start_case(row, args))

        for running in list(running_cases):
            if running.process.poll() is None:
                continue
            outcome, event = _finish_case(running, args)
            running_cases.remove(running)
            progress["events"].append(event)
            if outcome == "completed":
                progress["completed"] += 1
            elif outcome == "failed":
                progress["failed"] += 1
            else:
                progress[outcome] = progress.get(outcome, 0) + 1
            _write_progress(progress_path, progress)

        if pending or running_cases:
            time.sleep(float(args.poll_interval))

    if _STOP_REQUESTED:
        for running in list(running_cases):
            if running.process.poll() is None:
                running.process.terminate()
        time.sleep(5)
        for running in list(running_cases):
            if running.process.poll() is None:
                running.process.kill()
            running.stdout_handle.close()
            running.stderr_handle.close()
        progress["stopped"] = True
        _write_progress(progress_path, progress)
        raise SystemExit(143)

    progress["finished_at_epoch"] = time.time()
    _write_progress(progress_path, progress)
    print(json.dumps({key: progress[key] for key in ("completed", "failed", "skipped_completed_at_start")}, sort_keys=True))
    if progress["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
