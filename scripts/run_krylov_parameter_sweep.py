#!/usr/bin/env python
"""Run restartable QRTE/QKUD parameter sweeps for non-converged cases."""

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
PARAMETER_BY_METHOD = {"qrte": "delta_t", "qkud": "epsilon"}
DIR_PREFIX_BY_METHOD = {"qrte": "dt", "qkud": "eps"}
BASELINE_BY_METHOD = {
    "qrte": BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.1",
    "qkud": BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.1",
}

_STOP_REQUESTED = False
_RUNNING: list["RunningTask"] = []


@dataclass(frozen=True)
class SweepTask:
    row_index: int
    case_id: str
    parameter_name: str
    parameter_value: float
    value_dir: Path
    case_dir: Path
    summary_path: Path


@dataclass
class RunningTask:
    task: SweepTask
    process: subprocess.Popen[str]
    stdout_handle: Any
    stderr_handle: Any
    started_at: float


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for idx, row in enumerate(rows):
        row["_manifest_index"] = str(idx)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _completed(summary_path: Path) -> bool:
    payload = _load_json(summary_path)
    return bool(payload and payload.get("status") == "completed")


def _baseline_hit(case_id: str, baseline_dir: Path) -> bool:
    payload = _load_json(baseline_dir / case_id / "summary.json")
    return bool(
        payload
        and payload.get("status") == "completed"
        and payload.get("hit_chemical_accuracy") is True
        and payload.get("hit_iteration") is not None
    )


def _tag(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


CANONICAL_SWEEP_VALUES = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
)


def _parse_values(raw: str | None) -> list[float]:
    if raw:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = list(CANONICAL_SWEEP_VALUES)
    return [float(f"{value:.10g}") for value in values]


def _select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = rows
    if args.case_id:
        allowed = set(args.case_id)
        selected = [row for row in selected if row["case_id"] in allowed]
    if args.row_index:
        allowed_indexes = set(args.row_index)
        selected = [row for row in selected if int(row["_manifest_index"]) in allowed_indexes]
    return selected


def _build_tasks(rows: list[dict[str, str]], args: argparse.Namespace) -> list[SweepTask]:
    parameter_name = PARAMETER_BY_METHOD[args.method]
    dir_prefix = DIR_PREFIX_BY_METHOD[args.method]
    values = _parse_values(args.values)
    tasks: list[SweepTask] = []
    for value in values:
        value_dir = args.outdir / f"{dir_prefix}_{_tag(value)}"
        for row in rows:
            case_id = row["case_id"]
            if not args.include_baseline_hits and _baseline_hit(case_id, args.baseline_dir):
                continue
            case_dir = value_dir / case_id
            tasks.append(
                SweepTask(
                    row_index=int(row["_manifest_index"]),
                    case_id=case_id,
                    parameter_name=parameter_name,
                    parameter_value=value,
                    value_dir=value_dir,
                    case_dir=case_dir,
                    summary_path=case_dir / "summary.json",
                )
            )
    if args.shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % args.shard_count == args.shard_index]
    return tasks


def _write_task_manifest(path: Path, tasks: list[SweepTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "case_id",
                "parameter_name",
                "parameter_value",
                "case_dir",
                "summary_path",
            ],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "row_index": task.row_index,
                    "case_id": task.case_id,
                    "parameter_name": task.parameter_name,
                    "parameter_value": task.parameter_value,
                    "case_dir": _display_path(task.case_dir),
                    "summary_path": _display_path(task.summary_path),
                }
            )


def _build_child_command(task: SweepTask, args: argparse.Namespace) -> list[str]:
    delta_t = task.parameter_value if args.method == "qrte" else args.delta_t
    epsilon = task.parameter_value if args.method == "qkud" else args.epsilon
    cmd = [
        sys.executable,
        str(DRIVER),
        "--single-case",
        "--method",
        args.method,
        "--manifest",
        str(args.manifest),
        "--outdir",
        str(task.value_dir),
        "--qcant-root",
        str(args.qcant_root),
        "--case-id",
        task.case_id,
        "--n-steps",
        str(args.n_steps),
        "--delta-t",
        str(delta_t),
        "--epsilon",
        str(epsilon),
        "--evolution-mode",
        args.evolution_mode,
        "--trotter-steps",
        str(args.trotter_steps),
        "--max-workers",
        str(args.max_workers),
        "--hamiltonian-source",
        args.hamiltonian_source,
        "--early-stop-chemical-accuracy",
    ]
    if args.hamiltonian_cache_dir is not None:
        cmd.extend(["--hamiltonian-cache-dir", str(args.hamiltonian_cache_dir)])
    return cmd


def _start_task(task: SweepTask, args: argparse.Namespace) -> RunningTask:
    task.case_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_child_command(task, args)
    launch_record = {
        "case_id": task.case_id,
        "command": cmd,
        "launcher": "run_krylov_parameter_sweep.py",
        "manifest_index": task.row_index,
        "method": args.method,
        "parameter_name": task.parameter_name,
        "parameter_value": task.parameter_value,
        "pool_concurrency": int(args.concurrency),
        "pool_id": args.pool_id,
        "case_max_workers": int(args.max_workers),
        "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
        "hamiltonian_source": args.hamiltonian_source,
        "n_steps": int(args.n_steps),
        "started_at_epoch": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    _write_json(task.case_dir / "sweep_launch.json", launch_record)
    stdout_handle = (task.case_dir / "sweep_stdout.log").open("w", encoding="utf-8")
    stderr_handle = (task.case_dir / "sweep_stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        cwd=str(BENCHMARK_ROOT),
    )
    running = RunningTask(task=task, process=process, stdout_handle=stdout_handle, stderr_handle=stderr_handle, started_at=time.perf_counter())
    _RUNNING.append(running)
    print(
        f"START {args.method} case={task.case_id} {task.parameter_name}={task.parameter_value:g} pid={process.pid}",
        flush=True,
    )
    return running


def _finish_task(running: RunningTask) -> tuple[str, dict[str, Any]]:
    rc = running.process.returncode
    running.stdout_handle.close()
    running.stderr_handle.close()
    if running in _RUNNING:
        _RUNNING.remove(running)
    summary = _load_json(running.task.summary_path)
    status = summary.get("status") if summary else None
    if rc == 0 and status == "completed":
        outcome = "completed"
    elif status == "completed":
        outcome = "completed"
    elif status == "failed":
        outcome = "failed"
    else:
        outcome = "failed"
    event = {
        "case_id": running.task.case_id,
        "elapsed_seconds": float(time.perf_counter() - running.started_at),
        "manifest_index": running.task.row_index,
        "outcome": outcome,
        "parameter_name": running.task.parameter_name,
        "parameter_value": running.task.parameter_value,
        "returncode": rc,
        "summary_status": status,
        "hit_iteration": summary.get("hit_iteration") if summary else None,
        "hit_chemical_accuracy": bool(summary.get("hit_chemical_accuracy", False)) if summary else False,
    }
    print(
        f"END case={running.task.case_id} {running.task.parameter_name}={running.task.parameter_value:g} "
        f"outcome={outcome} hit={event['hit_iteration']} elapsed={event['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return outcome, event


def _terminate_children(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Signal {signum} received; terminating {len(_RUNNING)} active sweep tasks.", flush=True)
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
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--row-index", action="append", type=int, default=None)
    parser.add_argument(
        "--values",
        default=None,
        help="Comma-separated sweep values; default is 0.1..1.0 by 0.1 plus 2..10 by 1.",
    )
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--delta-t", type=float, default=0.1, help="Fixed QRTE delta_t fallback for non-QRTE methods.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Fixed QKUD epsilon fallback for non-QKUD methods.")
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=36)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument(
        "--hamiltonian-source",
        choices=("casci", "casscf", "casscf_natorb", "casscf-natural", "molecular"),
        default="casscf_natorb",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--include-baseline-hits", action="store_true")
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0") or "0"))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--pool-id", default=os.environ.get("SLURM_JOB_ID"))
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate_children)
    signal.signal(signal.SIGINT, _terminate_children)
    args = parse_args()
    if args.baseline_dir is None:
        args.baseline_dir = BASELINE_BY_METHOD[args.method]
    args.manifest = args.manifest.resolve()
    args.outdir = args.outdir.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    args.qcant_root = args.qcant_root.resolve()
    if args.hamiltonian_cache_dir is not None:
        args.hamiltonian_cache_dir = args.hamiltonian_cache_dir.resolve()
        args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)
    if args.pool_id is None:
        args.pool_id = f"local-{int(time.time())}-{os.getpid()}"
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit(f"--shard-index must be in [0, {args.shard_count - 1}]")
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = _select_rows(_load_manifest(args.manifest), args)
    if not rows:
        raise SystemExit("No manifest rows selected")
    tasks = _build_tasks(rows, args)
    pending = [task for task in tasks if not _completed(task.summary_path)]
    progress_path = args.outdir / f"sweep_progress_{args.pool_id}_shard{args.shard_index}_of_{args.shard_count}.json"
    task_manifest_path = args.outdir / f"sweep_task_manifest_{args.pool_id}_shard{args.shard_index}_of_{args.shard_count}.csv"
    _write_task_manifest(task_manifest_path, tasks)

    progress: dict[str, Any] = {
        "completed": 0,
        "concurrency": int(args.concurrency),
        "events": [],
        "failed": 0,
        "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
        "hamiltonian_source": args.hamiltonian_source,
        "max_workers": int(args.max_workers),
        "method": args.method,
        "n_steps": int(args.n_steps),
        "n_selected_tasks": len(tasks),
        "n_pending_at_start": len(pending),
        "parameter_name": PARAMETER_BY_METHOD[args.method],
        "parameter_values": _parse_values(args.values),
        "pool_id": args.pool_id,
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "skipped_completed_at_start": len(tasks) - len(pending),
        "started_at_epoch": time.time(),
        "task_manifest": _display_path(task_manifest_path),
    }
    _write_progress(progress_path, progress)
    print(
        f"SWEEP {args.method} shard={args.shard_index}/{args.shard_count} "
        f"tasks={len(tasks)} pending={len(pending)} skipped={progress['skipped_completed_at_start']} "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    running_tasks: list[RunningTask] = []
    while (pending or running_tasks) and not _STOP_REQUESTED:
        while pending and len(running_tasks) < int(args.concurrency) and not _STOP_REQUESTED:
            task = pending.pop(0)
            if _completed(task.summary_path):
                progress["skipped_completed_at_launch"] = progress.get("skipped_completed_at_launch", 0) + 1
                continue
            running_tasks.append(_start_task(task, args))
        for running in list(running_tasks):
            if running.process.poll() is None:
                continue
            outcome, event = _finish_task(running)
            running_tasks.remove(running)
            progress["events"].append(event)
            if outcome == "completed":
                progress["completed"] += 1
            else:
                progress["failed"] += 1
            _write_progress(progress_path, progress)
        if pending or running_tasks:
            time.sleep(float(args.poll_interval))

    if _STOP_REQUESTED:
        for running in list(running_tasks):
            if running.process.poll() is None:
                running.process.terminate()
        time.sleep(5)
        for running in list(running_tasks):
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
