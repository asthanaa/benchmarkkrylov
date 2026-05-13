#!/usr/bin/env python
"""Refresh benchmark plots whenever new result summaries are written."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_PLOT_DIR = BENCHMARK_ROOT / "plots"
WATCHED_SUMMARY_ROOTS = [
    BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.1",
    BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.1",
    BENCHMARK_ROOT / "runs" / "qrte_natorb_delta_sweep",
    BENCHMARK_ROOT / "runs" / "qkud_natorb_epsilon_sweep",
    BENCHMARK_ROOT / "runs" / "adapt_h10_qcant",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summary_files() -> list[Path]:
    paths: list[Path] = []
    for root in WATCHED_SUMMARY_ROOTS:
        if not root.exists():
            continue
        if root.is_dir():
            paths.extend(root.glob("*/summary.json"))
            paths.extend(root.glob("*/*/summary.json"))
            direct_summary = root / "summary.json"
            if direct_summary.exists():
                paths.append(direct_summary)
    return sorted(set(paths))


def _fingerprint(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    items = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        items.append((str(path.relative_to(BENCHMARK_ROOT)), int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(items)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _count_summaries(paths: list[Path]) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "qrte": {"completed": 0, "failed": 0, "other": 0},
        "qkud": {"completed": 0, "failed": 0, "other": 0},
        "adapt_h10": {"completed": 0, "failed": 0, "other": 0},
    }
    for path in paths:
        payload = _read_json(path)
        if payload is None:
            continue
        method = str(payload.get("method", ""))
        if method in {"qrte", "qkud"}:
            bucket = counts[method]
        elif payload.get("case_id") == "H10_near_equi_sto3g_0" and method == "adapt_vqe_qulacs":
            bucket = counts["adapt_h10"]
        else:
            continue
        status = str(payload.get("status", "other"))
        if status in {"completed", "failed"}:
            bucket[status] += 1
        else:
            bucket["other"] += 1
    return counts


def _method_has_any_summary(paths: list[Path], method: str) -> bool:
    for path in paths:
        payload = _read_json(path)
        if payload is None:
            continue
        if payload.get("method") == method:
            return True
    return False


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


def _refresh(args: argparse.Namespace, *, reason: str, status_path: Path) -> tuple[int, dict[str, Any]]:
    paths_before = _summary_files()
    requested_methods = ["qrte", "qkud"] if args.method == "both" else [args.method]
    methods_to_refresh = [
        method for method in requested_methods if args.allow_empty_refresh or _method_has_any_summary(paths_before, method)
    ]
    skipped_methods = [method for method in requested_methods if method not in methods_to_refresh]
    if not methods_to_refresh:
        status = {
            "status": "skipped_no_summaries",
            "reason": reason,
            "updated_at_utc": _utc_now(),
            "summary_counts": _count_summaries(paths_before),
            "summary_file_count": len(paths_before),
            "watched_job_ids": args.watch_job_id,
            "running_watched_job_ids": _jobs_in_queue(args.watch_job_id),
            "refreshed_methods": [],
            "skipped_methods": skipped_methods,
        }
        _write_status(status_path, status)
        print(f"[{status['updated_at_utc']}] skipped refresh; no summaries yet", flush=True)
        return 0, status

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    returncode = 0
    started = time.perf_counter()
    for method in methods_to_refresh:
        cmd = [
            sys.executable,
            str(BENCHMARK_ROOT / "scripts" / "collect_and_plot.py"),
            "--method",
            method,
            "--manifest",
            str(args.manifest),
            "--plot-dir",
            str(args.plot_dir),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        returncode = max(returncode, int(result.returncode))
    paths = _summary_files()
    status = {
        "status": "refreshed" if returncode == 0 else "refresh_failed",
        "reason": reason,
        "updated_at_utc": _utc_now(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "returncode": returncode,
        "stdout": "".join(stdout_parts)[-4000:],
        "stderr": "".join(stderr_parts)[-4000:],
        "summary_counts": _count_summaries(paths),
        "summary_file_count": len(paths),
        "watched_job_ids": args.watch_job_id,
        "running_watched_job_ids": _jobs_in_queue(args.watch_job_id),
        "refreshed_methods": methods_to_refresh,
        "skipped_methods": skipped_methods,
    }
    _write_status(status_path, status)
    print(
        f"[{status['updated_at_utc']}] refresh reason={reason} "
        f"methods={','.join(methods_to_refresh)} rc={returncode}",
        flush=True,
    )
    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if stdout:
        print(stdout.rstrip(), flush=True)
    if stderr:
        print(stderr.rstrip(), file=sys.stderr, flush=True)
    return returncode, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qrte", "qkud", "both"), default="both")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--max-hours", type=float, default=28.0 * 24.0)
    parser.add_argument("--watch-job-id", action="append", default=[])
    parser.add_argument("--status-path", type=Path, default=DEFAULT_PLOT_DIR / "live_plot_watch_status.json")
    parser.add_argument("--allow-empty-refresh", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status_path
    start = time.monotonic()
    last_fingerprint: tuple[tuple[str, int, int], ...] | None = None
    final_refresh_done = False

    while True:
        paths = _summary_files()
        current_fingerprint = _fingerprint(paths)
        running_jobs = _jobs_in_queue(args.watch_job_id)
        reason = None
        if last_fingerprint is None:
            reason = "initial"
        elif current_fingerprint != last_fingerprint:
            reason = "summary_changed"
        elif args.watch_job_id and not running_jobs and not final_refresh_done:
            reason = "watched_jobs_finished"
            final_refresh_done = True

        if reason is not None:
            _refresh(args, reason=reason, status_path=status_path)
            last_fingerprint = current_fingerprint

        status = {
            "status": "watching",
            "updated_at_utc": _utc_now(),
            "summary_counts": _count_summaries(paths),
            "summary_file_count": len(paths),
            "watched_job_ids": args.watch_job_id,
            "running_watched_job_ids": running_jobs,
            "next_check_seconds": float(args.interval_seconds),
        }
        _write_status(status_path, status)

        if args.once:
            break
        if args.watch_job_id and not running_jobs and final_refresh_done:
            status["status"] = "finished"
            status["updated_at_utc"] = _utc_now()
            _write_status(status_path, status)
            print("All watched jobs finished; plot watcher exiting.", flush=True)
            break
        if time.monotonic() - start >= float(args.max_hours) * 3600.0:
            status["status"] = "timed_out"
            status["updated_at_utc"] = _utc_now()
            _write_status(status_path, status)
            print("Plot watcher reached max-hours; exiting.", flush=True)
            break
        time.sleep(max(1.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
