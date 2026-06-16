#!/usr/bin/env python
"""Refresh the 0.2 QRTE/QKUD plots as Talon runs finish cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_PLOT_DIR = BENCHMARK_ROOT / "plots" / "mushir_dt0p2_eps0p2"
STATUS_PATH = DEFAULT_PLOT_DIR / "live_plot_watch_status.json"
RUN_DIRS = {
    "qrte": BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.2",
    "qkud": BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.2",
}
COLLECTOR = BENCHMARK_ROOT / "scripts" / "Mushir_script" / "collect_and_plot_dt0p2_eps0p2.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _summary_files(method: str) -> list[Path]:
    root = RUN_DIRS[method]
    if not root.exists():
        return []
    return sorted(root.glob("*/summary.json"))


def _fingerprint(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    items = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        items.append((str(path.relative_to(BENCHMARK_ROOT)), int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(items)


def _counts(method: str) -> dict[str, int]:
    counts = {"completed": 0, "failed": 0, "other": 0, "hit_chemical_accuracy": 0}
    for path in _summary_files(method):
        payload = _read_json(path)
        if payload is None:
            continue
        status = str(payload.get("status", "other"))
        if status in {"completed", "failed"}:
            counts[status] += 1
        else:
            counts["other"] += 1
        if bool(payload.get("hit_chemical_accuracy", False)):
            counts["hit_chemical_accuracy"] += 1
    return counts


def _jobs_in_queue(job_ids: list[str]) -> list[str]:
    if not job_ids:
        return []
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return job_ids
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _refresh(methods: list[str], args: argparse.Namespace, reason: str) -> int:
    returncode = 0
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    started = time.perf_counter()
    for method in methods:
        paths = _summary_files(method)
        if not paths and not args.allow_empty_refresh:
            continue
        cmd = [
            sys.executable,
            str(COLLECTOR),
            "--method",
            method,
            "--manifest",
            str(args.manifest),
            "--plot-dir",
            str(args.plot_dir),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=str(BENCHMARK_ROOT))
        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        returncode = max(returncode, int(result.returncode))

    status = {
        "status": "refreshed" if returncode == 0 else "refresh_failed",
        "reason": reason,
        "updated_at_utc": _utc_now(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "returncode": returncode,
        "refreshed_methods": methods,
        "summary_counts": {method: _counts(method) for method in ("qrte", "qkud")},
        "watched_job_ids": args.watch_job_id,
        "running_watched_job_ids": _jobs_in_queue(args.watch_job_id),
        "stdout": "".join(stdout_parts)[-4000:],
        "stderr": "".join(stderr_parts)[-4000:],
    }
    _write_status(args.status_path, status)
    print(
        f"[{status['updated_at_utc']}] refresh reason={reason} "
        f"methods={','.join(methods)} rc={returncode}",
        flush=True,
    )
    if status["stdout"]:
        print(status["stdout"].rstrip(), flush=True)
    if status["stderr"]:
        print(status["stderr"].rstrip(), file=sys.stderr, flush=True)
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qrte", "qkud", "both"), default="both")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--max-hours", type=float, default=28.0 * 24.0)
    parser.add_argument("--watch-job-id", action="append", default=[])
    parser.add_argument("--status-path", type=Path, default=STATUS_PATH)
    parser.add_argument("--allow-empty-refresh", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ["qrte", "qkud"] if args.method == "both" else [args.method]
    start = time.monotonic()
    last_fingerprint: dict[str, tuple[tuple[str, int, int], ...] | None] = {method: None for method in methods}
    final_refresh_done = False
    initialized = False

    while True:
        running_jobs = _jobs_in_queue(args.watch_job_id)
        changed_methods: list[str] = []
        for method in methods:
            current = _fingerprint(_summary_files(method))
            previous = last_fingerprint[method]
            if previous is None:
                changed_methods.append(method)
            elif current != previous:
                changed_methods.append(method)
            last_fingerprint[method] = current

        reason = None
        if changed_methods:
            reason = "initial" if not initialized else "summary_changed"
            initialized = True
        elif args.watch_job_id and not running_jobs and not final_refresh_done:
            reason = "watched_jobs_finished"
            changed_methods = methods
            final_refresh_done = True

        if reason is not None:
            _refresh(changed_methods, args, reason)

        status = {
            "status": "watching",
            "updated_at_utc": _utc_now(),
            "summary_counts": {method: _counts(method) for method in ("qrte", "qkud")},
            "watched_job_ids": args.watch_job_id,
            "running_watched_job_ids": running_jobs,
            "next_check_seconds": float(args.interval_seconds),
        }
        _write_status(args.status_path, status)

        if args.once:
            status["status"] = "finished_once"
            status["updated_at_utc"] = _utc_now()
            _write_status(args.status_path, status)
            break
        if args.watch_job_id and not running_jobs and final_refresh_done:
            status["status"] = "finished"
            status["updated_at_utc"] = _utc_now()
            _write_status(args.status_path, status)
            print("All watched jobs finished; 0.2 plot watcher exiting.", flush=True)
            break
        if time.monotonic() - start >= float(args.max_hours) * 3600.0:
            status["status"] = "timed_out"
            status["updated_at_utc"] = _utc_now()
            _write_status(args.status_path, status)
            print("0.2 plot watcher reached max-hours; exiting.", flush=True)
            break
        time.sleep(max(1.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
