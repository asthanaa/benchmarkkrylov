#!/usr/bin/env python
"""Run a sorted CSV of QCANT parameter-probe tasks inside one allocation."""

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
DRIVER = BENCHMARK_ROOT / "scripts" / "run_krylov_benchmark.py"
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_TASK_CSV = BENCHMARK_ROOT / "manifests" / "unconverged_param_probe_0p5_1p0_tasks.csv"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb"

_STOP_REQUESTED = False
_RUNNING: list["RunningTask"] = []


@dataclass(frozen=True)
class ProbeTask:
    index: int
    method: str
    case_id: str
    parameter_name: str
    parameter_value: float
    outdir: Path
    size_key: str

    @property
    def case_dir(self) -> Path:
        return self.outdir / self.case_id

    @property
    def summary_path(self) -> Path:
        return self.case_dir / "summary.json"

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.method, self.case_id)


@dataclass
class RunningTask:
    task: ProbeTask
    process: subprocess.Popen[str]
    stdout_handle: Any
    stderr_handle: Any
    started_at: float


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _completed(path: Path) -> bool:
    payload = _load_json(path)
    return bool(payload and payload.get("status") == "completed")


def _hit(path: Path) -> bool:
    payload = _load_json(path)
    return bool(payload and payload.get("status") == "completed" and payload.get("hit_chemical_accuracy"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _load_tasks(path: Path) -> list[ProbeTask]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tasks: list[ProbeTask] = []
    for index, row in enumerate(rows):
        tasks.append(
            ProbeTask(
                index=index,
                method=row["method"],
                case_id=row["case_id"],
                parameter_name=row["parameter_name"],
                parameter_value=float(row["parameter_value"]),
                outdir=(BENCHMARK_ROOT / row["outdir"]).resolve(),
                size_key=row.get("size_key", ""),
            )
        )
    return tasks


def _build_command(task: ProbeTask, args: argparse.Namespace) -> list[str]:
    delta_t = task.parameter_value if task.parameter_name == "delta_t" else float(args.delta_t)
    epsilon = task.parameter_value if task.parameter_name == "epsilon" else float(args.epsilon)
    return [
        sys.executable,
        str(DRIVER),
        "--single-case",
        "--method",
        task.method,
        "--manifest",
        str(args.manifest),
        "--outdir",
        str(task.outdir),
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
        "--hamiltonian-cache-dir",
        str(args.hamiltonian_cache_dir),
        "--early-stop-chemical-accuracy",
    ]


def _start_task(task: ProbeTask, args: argparse.Namespace) -> RunningTask:
    task.case_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(task, args)
    _write_json(
        task.case_dir / "param_probe_launch.json",
        {
            "case_id": task.case_id,
            "command": command,
            "launcher": "run_param_task_csv_pool.py",
            "method": task.method,
            "parameter_name": task.parameter_name,
            "parameter_value": task.parameter_value,
            "pool_id": args.pool_id,
            "size_key": task.size_key,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "task_index": task.index,
            "started_at_epoch": time.time(),
        },
    )
    stdout_handle = (task.case_dir / "param_probe_stdout.log").open("w", encoding="utf-8")
    stderr_handle = (task.case_dir / "param_probe_stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, text=True, cwd=str(BENCHMARK_ROOT))
    running = RunningTask(task, process, stdout_handle, stderr_handle, time.perf_counter())
    _RUNNING.append(running)
    print(
        f"START index={task.index} {task.method} {task.case_id} "
        f"{task.parameter_name}={task.parameter_value:g} pid={process.pid}",
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
    outcome = "completed" if status == "completed" else "failed"
    event = {
        "case_id": running.task.case_id,
        "elapsed_seconds": float(time.perf_counter() - running.started_at),
        "hit_chemical_accuracy": bool(summary.get("hit_chemical_accuracy", False)) if summary else False,
        "hit_iteration": summary.get("hit_iteration") if summary else None,
        "method": running.task.method,
        "outcome": outcome,
        "parameter_name": running.task.parameter_name,
        "parameter_value": running.task.parameter_value,
        "returncode": rc,
        "summary_status": status,
        "task_index": running.task.index,
    }
    print(
        f"END index={running.task.index} {running.task.method} {running.task.case_id} "
        f"{running.task.parameter_name}={running.task.parameter_value:g} "
        f"outcome={outcome} hit={event['hit_iteration']} elapsed={event['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return outcome, event


def _pop_next_task(
    pending: list[ProbeTask],
    running_tasks: list[RunningTask],
    hit_groups: set[tuple[str, str]],
    progress: dict[str, Any],
    *,
    stop_after_case_hit: bool,
) -> ProbeTask | None:
    running_groups = {running.task.group_key for running in running_tasks}
    index = 0
    while index < len(pending):
        task = pending[index]
        if _completed(task.summary_path):
            pending.pop(index)
            progress["skipped_completed_at_launch"] = progress.get("skipped_completed_at_launch", 0) + 1
            continue
        if stop_after_case_hit and task.group_key in hit_groups:
            pending.pop(index)
            progress["skipped_after_case_hit"] = progress.get("skipped_after_case_hit", 0) + 1
            continue
        if stop_after_case_hit and task.group_key in running_groups:
            index += 1
            continue
        return pending.pop(index)
    return None


def _terminate_children(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Signal {signum} received; terminating {len(_RUNNING)} active tasks.", flush=True)
    for running in list(_RUNNING):
        if running.process.poll() is None:
            running.process.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-csv", type=Path, default=DEFAULT_TASK_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--pool-id", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument(
        "--stop-after-case-hit",
        action="store_true",
        help="Skip remaining parameter values for a method/case after any completed value reaches chemical accuracy.",
    )
    parser.add_argument(
        "--hamiltonian-source",
        choices=("casci", "casscf", "casscf_natorb", "casscf-natural", "molecular"),
        default="casscf_natorb",
    )
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate_children)
    signal.signal(signal.SIGINT, _terminate_children)
    args = parse_args()
    args.task_csv = args.task_csv.resolve()
    args.manifest = args.manifest.resolve()
    args.qcant_root = args.qcant_root.resolve()
    args.hamiltonian_cache_dir = args.hamiltonian_cache_dir.resolve()
    if args.pool_id is None:
        args.pool_id = f"local-{int(time.time())}-{os.getpid()}"

    tasks = _load_tasks(args.task_csv)
    hit_groups = {task.group_key for task in tasks if _hit(task.summary_path)}
    pending: list[ProbeTask] = []
    skipped_after_case_hit_at_start = 0
    for task in tasks:
        if _completed(task.summary_path):
            continue
        if args.stop_after_case_hit and task.group_key in hit_groups:
            skipped_after_case_hit_at_start += 1
            continue
        pending.append(task)
    progress_path = BENCHMARK_ROOT / "runs" / f"param_probe_progress_{args.pool_id}.json"
    progress: dict[str, Any] = {
        "completed": 0,
        "concurrency": int(args.concurrency),
        "events": [],
        "failed": 0,
        "initial_hit_groups": [f"{method}:{case_id}" for method, case_id in sorted(hit_groups)],
        "max_workers": int(args.max_workers),
        "n_pending_at_start": len(pending),
        "n_tasks": len(tasks),
        "pool_id": args.pool_id,
        "skipped_after_case_hit_at_start": skipped_after_case_hit_at_start,
        "skipped_completed_at_start": len(tasks) - len(pending) - skipped_after_case_hit_at_start,
        "started_at_epoch": time.time(),
        "stop_after_case_hit": bool(args.stop_after_case_hit),
        "task_csv": str(args.task_csv),
    }
    _write_json(progress_path, progress)
    print(
        f"PARAM-PROBE tasks={len(tasks)} pending={len(pending)} "
        f"skipped={progress['skipped_completed_at_start']} concurrency={args.concurrency} max_workers={args.max_workers}",
        flush=True,
    )

    running_tasks: list[RunningTask] = []
    while (pending or running_tasks) and not _STOP_REQUESTED:
        while pending and len(running_tasks) < int(args.concurrency) and not _STOP_REQUESTED:
            task = _pop_next_task(
                pending,
                running_tasks,
                hit_groups,
                progress,
                stop_after_case_hit=bool(args.stop_after_case_hit),
            )
            if task is None:
                break
            running_tasks.append(_start_task(task, args))
        for running in list(running_tasks):
            if running.process.poll() is None:
                continue
            outcome, event = _finish_task(running)
            running_tasks.remove(running)
            progress["events"].append(event)
            progress[outcome] += 1
            if args.stop_after_case_hit and event.get("hit_chemical_accuracy"):
                hit_groups.add(running.task.group_key)
                progress["hit_groups"] = [f"{method}:{case_id}" for method, case_id in sorted(hit_groups)]
            _write_json(progress_path, progress)
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
        _write_json(progress_path, progress)
        raise SystemExit(143)

    progress["finished_at_epoch"] = time.time()
    _write_json(progress_path, progress)
    print(json.dumps({key: progress[key] for key in ("completed", "failed", "skipped_completed_at_start")}, sort_keys=True))
    if progress["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
