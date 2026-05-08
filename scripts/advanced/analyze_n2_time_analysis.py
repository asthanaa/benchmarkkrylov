#!/usr/bin/env python3
"""Summarize N2 CAS(12,12) second-orthogonalization timing runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any


HQ_DONE_RE = re.compile(
    r"ccik\(hq\) done\s+call=\s*(?P<call>\d+).*?"
    r"select_s=\s*(?P<select>[0-9.Ee+-]+).*?"
    r"sort_s=\s*(?P<sort>[0-9.Ee+-]+).*?"
    r"int_s=\s*(?P<internal>[0-9.Ee+-]+).*?"
    r"ext_s=\s*(?P<external>[0-9.Ee+-]+).*?"
    r"assign_s=\s*(?P<assign>[0-9.Ee+-]+)"
)

PROJECTED_HQ_RE = re.compile(r"ccik\(thick\) projected_hq_cpu_s=\s*(?P<cpu>[0-9.Ee+-]+)")

RUN_NAMES = ("baseline_kprop500_kstore10000_m031", "secondortho_kprop500_kstore10000_m031")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _trace_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"trace_complete": False}
    rows: list[dict[str, str]] = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        return {"trace_complete": False}
    abs_errors = [_parse_float(row.get("abs_delta_mHa")) for row in rows]
    signed_errors = [_parse_float(row.get("delta_mHa")) for row in rows]
    elapsed = [_parse_float(row.get("elapsed_s")) for row in rows]
    cycles: dict[int, list[float]] = {}
    for row in rows:
        cycle = int(float(row["cycle"]))
        value = _parse_float(row.get("elapsed_s"))
        if value is not None:
            cycles.setdefault(cycle, []).append(value)
    cycle_wall = {}
    prev_end = 0.0
    for cycle in sorted(cycles):
        end = max(cycles[cycle])
        cycle_wall[str(cycle)] = max(0.0, end - prev_end)
        prev_end = end
    return {
        "trace_complete": True,
        "trace_points": len(rows),
        "last_elapsed_s": elapsed[-1],
        "min_abs_delta_mHa": min(x for x in abs_errors if x is not None),
        "min_signed_delta_mHa": min(x for x in signed_errors if x is not None),
        "final_trace_abs_delta_mHa": abs_errors[-1],
        "final_trace_delta_mHa": signed_errors[-1],
        "cycle_wall_s": cycle_wall,
    }


def _parse_hq_log(path: Path) -> dict[str, Any]:
    totals = {
        "hq_call_count": 0,
        "hq_select_cpu_s": 0.0,
        "hq_sort_cpu_s": 0.0,
        "hq_internal_cpu_s": 0.0,
        "hq_external_cpu_s": 0.0,
        "hq_assign_cpu_s": 0.0,
        "projected_hq_cycle_cpu_s": 0.0,
        "projected_hq_cycle_count": 0,
    }
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = HQ_DONE_RE.search(line)
        if match is not None:
            totals["hq_call_count"] += 1
            totals["hq_select_cpu_s"] += float(match.group("select"))
            totals["hq_sort_cpu_s"] += float(match.group("sort"))
            totals["hq_internal_cpu_s"] += float(match.group("internal"))
            totals["hq_external_cpu_s"] += float(match.group("external"))
            totals["hq_assign_cpu_s"] += float(match.group("assign"))
            continue
        cycle_match = PROJECTED_HQ_RE.search(line)
        if cycle_match is not None:
            totals["projected_hq_cycle_count"] += 1
            totals["projected_hq_cycle_cpu_s"] += float(cycle_match.group("cpu"))
    totals["hq_accounted_cpu_s"] = (
        totals["hq_select_cpu_s"]
        + totals["hq_sort_cpu_s"]
        + totals["hq_internal_cpu_s"]
        + totals["hq_external_cpu_s"]
        + totals["hq_assign_cpu_s"]
    )
    return totals


def _run_row(root: Path, core_dir: Path, run_name: str) -> dict[str, Any]:
    cores = int(core_dir.name.removeprefix("cores"))
    run_dir = core_dir / run_name
    summary = _load_json(run_dir / "summary.json")
    trace = _trace_stats(run_dir / "iteration_trace.csv")
    hq = _parse_hq_log(run_dir / "ccik_stdout.log")
    row: dict[str, Any] = {
        "cores": cores,
        "core_dir": core_dir.name,
        "method": "secondortho" if run_name.startswith("secondortho") else "baseline",
        "run_name": run_name,
        "complete": summary is not None and trace.get("trace_complete", False),
        "wall_s": None if summary is None else summary.get("wall_s"),
        "final_abs_delta_mHa": None if summary is None else summary.get("final_abs_delta_mHa"),
        "plotted_final_abs_delta_mHa": None if summary is None else summary.get("plotted_final_abs_delta_mHa"),
        "returncode": None if summary is None else summary.get("returncode"),
    }
    row.update(trace)
    row.update(hq)
    return row


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for core_dir in sorted(root.glob("cores??")):
        if not core_dir.is_dir():
            continue
        for run_name in RUN_NAMES:
            rows.append(_run_row(root, core_dir, run_name))
    return rows


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def _build_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("complete") and row.get("wall_s") is not None]
    by_method = {(row["method"], int(row["cores"])): row for row in complete}
    overhead = {}
    for cores in sorted({int(row["cores"]) for row in rows}):
        base = by_method.get(("baseline", cores))
        second = by_method.get(("secondortho", cores))
        overhead[str(cores)] = None if base is None or second is None else _safe_ratio(second["wall_s"], base["wall_s"])
    speedup = {}
    for method in ("baseline", "secondortho"):
        row2 = by_method.get((method, 2))
        speedup[method] = {}
        for cores in (4, 8):
            row = by_method.get((method, cores))
            speedup[method][str(cores)] = None if row2 is None or row is None else _safe_ratio(row2["wall_s"], row["wall_s"])
    hq_scaling = {}
    for method in ("baseline", "secondortho"):
        hq_scaling[method] = {}
        base = by_method.get((method, 2))
        for component in ("hq_select_cpu_s", "hq_sort_cpu_s", "hq_internal_cpu_s", "hq_external_cpu_s", "hq_assign_cpu_s"):
            base_value = None if base is None else base.get(component)
            hq_scaling[method][component] = {}
            for cores in (4, 8):
                row = by_method.get((method, cores))
                hq_scaling[method][component][str(cores)] = (
                    None if row is None else _safe_ratio(base_value, row.get(component))
                )
    return {
        "complete_runs": len(complete),
        "expected_runs": len(rows),
        "secondortho_wall_over_baseline": overhead,
        "wall_speedup_vs_2core": speedup,
        "hq_cpu_component_speedup_vs_2core": hq_scaling,
    }


def _format_seconds(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def write_outputs(root: Path, rows: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    fields = [
        "cores",
        "method",
        "complete",
        "wall_s",
        "last_elapsed_s",
        "final_abs_delta_mHa",
        "plotted_final_abs_delta_mHa",
        "min_abs_delta_mHa",
        "min_signed_delta_mHa",
        "hq_call_count",
        "hq_accounted_cpu_s",
        "hq_select_cpu_s",
        "hq_sort_cpu_s",
        "hq_internal_cpu_s",
        "hq_external_cpu_s",
        "hq_assign_cpu_s",
        "projected_hq_cycle_cpu_s",
        "returncode",
    ]
    csv_path = root / "time_analysis_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path = root / "time_analysis_summary.json"
    json_path.write_text(json.dumps({"runs": rows, "analysis": analysis}, indent=2), encoding="utf-8")

    lines = [
        "# N2 CAS(12,12) Timing Analysis Summary",
        "",
        f"Complete runs: {analysis['complete_runs']} / {analysis['expected_runs']}",
        "",
        "| cores | method | wall s | final abs mHa | min abs mHa | HQ CPU s | select | sort | internal | external | assign |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (int(item["cores"]), str(item["method"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cores"]),
                    str(row["method"]),
                    _format_seconds(row.get("wall_s")),
                    _format_seconds(row.get("plotted_final_abs_delta_mHa")),
                    _format_seconds(row.get("min_abs_delta_mHa")),
                    _format_seconds(row.get("hq_accounted_cpu_s")),
                    _format_seconds(row.get("hq_select_cpu_s")),
                    _format_seconds(row.get("hq_sort_cpu_s")),
                    _format_seconds(row.get("hq_internal_cpu_s")),
                    _format_seconds(row.get("hq_external_cpu_s")),
                    _format_seconds(row.get("hq_assign_cpu_s")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Wall-Time Scaling",
            "",
            "Speedup values are relative to the 2-core run of the same method.",
            "",
        ]
    )
    for method, values in analysis["wall_speedup_vs_2core"].items():
        lines.append(
            f"- {method}: 4 cores = {values.get('4')}, 8 cores = {values.get('8')}"
        )
    lines.extend(
        [
            "",
            "## Second-Orthogonalization Overhead",
            "",
            "Values are `secondortho wall_s / baseline wall_s` at the same core count.",
            "",
        ]
    )
    for cores, value in analysis["secondortho_wall_over_baseline"].items():
        lines.append(f"- {cores} cores: {value}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `wall_s` comes from the Python runner around the QP CCIK executable.",
            "- `HQ CPU s` components come from QP verbose timing lines and are CPU seconds, not wall seconds.",
            "- The HQ components show where the CPU work is concentrated; wall-speedup is the primary scaling metric.",
        ]
    )
    (root / "time_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("inputs/n2/2.0/12,12/time_analysis"),
        help="Timing campaign root containing cores02/cores04/cores08.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = collect_rows(root)
    analysis = _build_analysis(rows)
    write_outputs(root, rows, analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
