#!/usr/bin/env python
"""Run a strided subset of a QRTE scan within one Slurm worker job."""

import argparse

import run_qrte_mean_field_active_dt_scan_upto12_exact as scan


def select_tasks(tasks, worker_index, worker_count):
    if not 1 <= worker_count <= len(tasks):
        raise ValueError(f"Worker count must be between 1 and {len(tasks)}")
    if not 0 <= worker_index < worker_count:
        raise ValueError("Worker index must be in [0, worker_count)")
    return [task for task in tasks if task["task_index"] % worker_count == worker_index]


def main(argv=None, *, scan_module=None):
    driver = scan if scan_module is None else scan_module
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    worker, remaining = parser.parse_known_args(argv)
    args = driver.parse_args(remaining)
    if args.case_id is not None or args.case_index is not None:
        raise ValueError("Workers partition the full scan; do not combine with case selection")
    tasks = select_tasks(driver.load_tasks(args), worker.worker_index, worker.worker_count)
    if args.count_cases:
        print(len(tasks))
        return
    print(f"Worker {worker.worker_index}/{worker.worker_count}: "
          f"{len(tasks)} conditions, task indices {[t['task_index'] for t in tasks]}", flush=True)
    if args.dry_run:
        for task in tasks:
            print(f"{task['task_index']:2d} {task['case']['case_id']} factor={task['factor']:g} "
                  f"dt={task['delta_t']:.12g} a.u.")
        return
    failures = 0
    for task in tasks:
        try:
            driver.run_task(task, args)
        except Exception as exc:
            failures += 1
            print(f"FAIL task={task['task_index']} case={task['case']['case_id']} "
                  f"factor={task['factor']}: {exc!r}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
