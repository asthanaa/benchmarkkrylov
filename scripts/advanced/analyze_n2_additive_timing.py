#!/usr/bin/env python3
"""Summarize additive wall-clock timing blocks from N2 CCIK runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


RUN_NAMES = ("baseline_kprop500_kstore10000_m031", "secondortho_kprop500_kstore10000_m031")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _kv_pairs(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in KV_RE.findall(line)}


def _parse_timing_block(path: Path) -> dict[str, Any]:
    sections: dict[str, dict[str, float]] = {
        "overall": {},
        "top": {},
        "reconcile": {},
        "next": {},
        "restart": {},
    }
    if not path.exists():
        return {"timing_complete": False}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("CCIK_TIMING total_wall_s="):
            sections["overall"].update(_kv_pairs(line))
        elif line.startswith("CCIK_TIMING_TOP"):
            sections["top"].update(_kv_pairs(line))
        elif line.startswith("CCIK_TIMING_RECONCILE"):
            sections["reconcile"].update(_kv_pairs(line))
        elif line.startswith("CCIK_TIMING_NEXT"):
            sections["next"].update(_kv_pairs(line))
        elif line.startswith("CCIK_TIMING_RESTART"):
            sections["restart"].update(_kv_pairs(line))

    total = sections["overall"].get("total_wall_s")
    top = sections["top"]
    rec = sections["reconcile"]
    next_section = sections["next"]
    restart = sections["restart"]

    if total is None:
        return {"timing_complete": False}

    next_total = next_section.get("total_wall_s")
    next_accounted = sum(
        float(next_section.get(key, 0.0))
        for key in ("first_orth_wall_s", "store_wall_s", "second_orth_wall_s", "normalize_wall_s", "overlap_wall_s")
    )
    next_misc = next_section.get("misc_wall_s")
    if next_misc is None and next_total is not None:
        next_misc = next_total - next_accounted

    restart_total = restart.get("total_wall_s")
    restart_accounted = sum(
        float(restart.get(key, 0.0))
        for key in ("solve_ritz_wall_s", "build_wall_s", "store_wall_s", "orth_wall_s", "normalize_wall_s")
    )
    restart_misc = restart.get("misc_wall_s")
    if restart_misc is None and restart_total is not None:
        restart_misc = restart_total - restart_accounted

    top_accounted = rec.get(
        "accounted_wall_s",
        sum(
            float(top.get(key, 0.0))
            for key in (
                "setup_wall_s",
                "load_wf_wall_s",
                "hq_wall_s",
                "ritz_wall_s",
                "checkpoint_wall_s",
                "next_wall_s",
                "restart_wall_s",
                "final_wall_s",
            )
        ),
    )
    other = rec.get("other_wall_s", total - top_accounted)

    return {
        "timing_complete": True,
        "solver_total_wall_s": total,
        "top_accounted_wall_s": top_accounted,
        "top_other_wall_s": other,
        "top_balance_s": total - top_accounted - other,
        "setup_wall_s": top.get("setup_wall_s"),
        "load_wf_wall_s": top.get("load_wf_wall_s"),
        "hq_wall_s": top.get("hq_wall_s"),
        "ritz_wall_s": top.get("ritz_wall_s"),
        "checkpoint_wall_s": top.get("checkpoint_wall_s"),
        "next_wall_s": top.get("next_wall_s"),
        "restart_wall_s": top.get("restart_wall_s"),
        "final_wall_s": top.get("final_wall_s"),
        "next_calls": _int_or_none(next_section.get("calls")),
        "next_total_wall_s": next_total,
        "next_first_orth_wall_s": next_section.get("first_orth_wall_s"),
        "next_store_wall_s": next_section.get("store_wall_s"),
        "next_second_orth_wall_s": next_section.get("second_orth_wall_s"),
        "next_normalize_wall_s": next_section.get("normalize_wall_s"),
        "next_overlap_wall_s": next_section.get("overlap_wall_s"),
        "next_misc_wall_s": next_misc,
        "next_second_orth_calls": _int_or_none(next_section.get("second_orth_calls")),
        "next_balance_s": None if next_total is None else next_total - next_accounted - float(next_misc or 0.0),
        "restart_calls": _int_or_none(restart.get("calls")),
        "restart_total_wall_s": restart_total,
        "restart_solve_ritz_wall_s": restart.get("solve_ritz_wall_s"),
        "restart_build_wall_s": restart.get("build_wall_s"),
        "restart_store_wall_s": restart.get("store_wall_s"),
        "restart_orth_wall_s": restart.get("orth_wall_s"),
        "restart_normalize_wall_s": restart.get("normalize_wall_s"),
        "restart_misc_wall_s": restart_misc,
        "hq_calls": _int_or_none(restart.get("hq_calls")),
        "restart_balance_s": None if restart_total is None else restart_total - restart_accounted - float(restart_misc or 0.0),
    }


def _int_or_none(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _trace_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"trace_complete": False}
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"trace_complete": False}
    return {
        "trace_complete": True,
        "trace_points": len(rows),
        "last_elapsed_s": _float_or_none(rows[-1].get("elapsed_s")),
        "final_trace_delta_mHa": _float_or_none(rows[-1].get("delta_mHa")),
        "final_trace_abs_delta_mHa": _float_or_none(rows[-1].get("abs_delta_mHa")),
        "min_abs_delta_mHa": min(
            value
            for value in (_float_or_none(row.get("abs_delta_mHa")) for row in rows)
            if value is not None
        ),
    }


def _float_or_none(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _run_row(core_dir: Path, run_name: str) -> dict[str, Any]:
    cores = int(core_dir.name.removeprefix("cores"))
    run_dir = core_dir / run_name
    summary = _load_json(run_dir / "summary.json")
    row: dict[str, Any] = {
        "cores": cores,
        "core_dir": core_dir.name,
        "method": "secondortho" if run_name.startswith("secondortho") else "baseline",
        "run_name": run_name,
        "complete": summary is not None,
        "runner_wall_s": None if summary is None else summary.get("wall_s"),
        "final_abs_delta_mHa": None if summary is None else summary.get("final_abs_delta_mHa"),
        "plotted_final_abs_delta_mHa": None if summary is None else summary.get("plotted_final_abs_delta_mHa"),
        "returncode": None if summary is None else summary.get("returncode"),
    }
    row.update(_trace_stats(run_dir / "iteration_trace.csv"))
    row.update(_parse_timing_block(run_dir / "ccik_stdout.log"))
    row["complete"] = bool(row.get("complete") and row.get("trace_complete") and row.get("timing_complete"))
    return row


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for core_dir in sorted(root.glob("cores??")):
        if not core_dir.is_dir():
            continue
        for run_name in RUN_NAMES:
            rows.append(_run_row(core_dir, run_name))
    return rows


def _safe_ratio(num: Any, den: Any) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def build_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("complete")]
    by_method = {(str(row["method"]), int(row["cores"])): row for row in complete}
    overhead: dict[str, Any] = {}
    second_orth_share: dict[str, Any] = {}
    for cores in sorted({int(row["cores"]) for row in rows}):
        base = by_method.get(("baseline", cores))
        second = by_method.get(("secondortho", cores))
        overhead[str(cores)] = None if base is None or second is None else {
            "solver_total_ratio": _safe_ratio(second.get("solver_total_wall_s"), base.get("solver_total_wall_s")),
            "runner_total_ratio": _safe_ratio(second.get("runner_wall_s"), base.get("runner_wall_s")),
            "solver_total_delta_s": float(second["solver_total_wall_s"]) - float(base["solver_total_wall_s"]),
        }
        second_orth_share[str(cores)] = None if second is None else _safe_ratio(
            second.get("next_second_orth_wall_s"),
            second.get("solver_total_wall_s"),
        )

    speedup: dict[str, dict[str, Any]] = {}
    for method in ("baseline", "secondortho"):
        row2 = by_method.get((method, 2))
        speedup[method] = {}
        for cores in (4, 8):
            row = by_method.get((method, cores))
            speedup[method][str(cores)] = None if row2 is None or row is None else _safe_ratio(
                row2.get("solver_total_wall_s"),
                row.get("solver_total_wall_s"),
            )

    return {
        "complete_runs": len(complete),
        "expected_runs": len(rows),
        "secondortho_over_baseline": overhead,
        "second_orthogonalization_share_of_secondortho_total": second_orth_share,
        "solver_wall_speedup_vs_2core": speedup,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{100.0 * float(value):.2f}%"


def write_outputs(root: Path, rows: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    fields = [
        "cores",
        "method",
        "complete",
        "runner_wall_s",
        "solver_total_wall_s",
        "setup_wall_s",
        "load_wf_wall_s",
        "hq_wall_s",
        "ritz_wall_s",
        "checkpoint_wall_s",
        "next_wall_s",
        "restart_wall_s",
        "final_wall_s",
        "top_other_wall_s",
        "top_balance_s",
        "next_calls",
        "next_first_orth_wall_s",
        "next_store_wall_s",
        "next_second_orth_wall_s",
        "next_normalize_wall_s",
        "next_overlap_wall_s",
        "next_misc_wall_s",
        "next_balance_s",
        "restart_calls",
        "restart_solve_ritz_wall_s",
        "restart_build_wall_s",
        "restart_store_wall_s",
        "restart_orth_wall_s",
        "restart_normalize_wall_s",
        "restart_misc_wall_s",
        "restart_balance_s",
        "hq_calls",
        "plotted_final_abs_delta_mHa",
        "returncode",
    ]
    with (root / "additive_timing_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (root / "additive_timing_summary.json").write_text(
        json.dumps({"runs": rows, "analysis": analysis}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# N2 CAS(12,12) Additive Timing Summary",
        "",
        f"Complete runs: {analysis['complete_runs']} / {analysis['expected_runs']}",
        "",
        "Top-level wall time is additive:",
        "",
        "`total = setup + load_wf + Hq + Ritz + checkpoint + next-vector + restart + final + other`",
        "",
        "| cores | method | total s | setup | load wf | Hq | Ritz | checkpoint | next | restart | final | other | balance |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (int(item["cores"]), str(item["method"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cores"]),
                    str(row["method"]),
                    _fmt(row.get("solver_total_wall_s")),
                    _fmt(row.get("setup_wall_s")),
                    _fmt(row.get("load_wf_wall_s")),
                    _fmt(row.get("hq_wall_s")),
                    _fmt(row.get("ritz_wall_s")),
                    _fmt(row.get("checkpoint_wall_s")),
                    _fmt(row.get("next_wall_s")),
                    _fmt(row.get("restart_wall_s")),
                    _fmt(row.get("final_wall_s")),
                    _fmt(row.get("top_other_wall_s")),
                    _fmt(row.get("top_balance_s"), 6),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Next-vector wall time is additive:",
            "",
            "`next-vector = first_orthogonalization + store/compress + second_orthogonalization + normalize + overlap_check + misc`",
            "",
            "| cores | method | calls | total s | first orth | store | second orth | normalize | overlap | misc | second-orth share of total | balance |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (int(item["cores"]), str(item["method"]))):
        second_share = _safe_ratio(row.get("next_second_orth_wall_s"), row.get("solver_total_wall_s"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cores"]),
                    str(row["method"]),
                    "" if row.get("next_calls") is None else str(row["next_calls"]),
                    _fmt(row.get("next_total_wall_s")),
                    _fmt(row.get("next_first_orth_wall_s")),
                    _fmt(row.get("next_store_wall_s")),
                    _fmt(row.get("next_second_orth_wall_s")),
                    _fmt(row.get("next_normalize_wall_s")),
                    _fmt(row.get("next_overlap_wall_s")),
                    _fmt(row.get("next_misc_wall_s")),
                    _pct(second_share),
                    _fmt(row.get("next_balance_s"), 6),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Restart wall time is additive:",
            "",
            "`restart = solve_Ritz + build_Ritz_vectors + store/compress + restart_orthogonalization + normalize + misc`",
            "",
            "| cores | method | calls | total s | solve Ritz | build | store | restart orth | normalize | misc | balance |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (int(item["cores"]), str(item["method"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cores"]),
                    str(row["method"]),
                    "" if row.get("restart_calls") is None else str(row["restart_calls"]),
                    _fmt(row.get("restart_total_wall_s")),
                    _fmt(row.get("restart_solve_ritz_wall_s")),
                    _fmt(row.get("restart_build_wall_s")),
                    _fmt(row.get("restart_store_wall_s")),
                    _fmt(row.get("restart_orth_wall_s")),
                    _fmt(row.get("restart_normalize_wall_s")),
                    _fmt(row.get("restart_misc_wall_s")),
                    _fmt(row.get("restart_balance_s"), 6),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Comparisons", ""])
    for cores, payload in analysis["secondortho_over_baseline"].items():
        if payload is None:
            lines.append(f"- {cores} cores: incomplete")
            continue
        lines.append(
            f"- {cores} cores: secondortho/baseline total = "
            f"{_fmt(payload['solver_total_ratio'], 4)}; delta = {_fmt(payload['solver_total_delta_s'])} s"
        )
    lines.append("")
    for cores, value in analysis["second_orthogonalization_share_of_secondortho_total"].items():
        lines.append(f"- {cores} cores: second orthogonalization share of secondortho total = {_pct(value)}")

    (root / "additive_timing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("inputs/n2/2.0/12,12/time_analysis/additive_timing"),
        help="Timing root containing cores02/cores04/cores08.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = collect_rows(root)
    analysis = build_analysis(rows)
    write_outputs(root, rows, analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
