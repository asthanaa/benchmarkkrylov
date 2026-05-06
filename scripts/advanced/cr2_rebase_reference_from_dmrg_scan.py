#!/usr/bin/env python3
"""Rebase Cr2 CK traces and plots onto the best DMRG scan reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import advanced.n2_thick_iteration_study as n2_study  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_cell(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    json_path = run_dir / "iteration_trace.json"
    if json_path.exists():
        return list(json.loads(json_path.read_text(encoding="utf-8")))
    csv_path = run_dir / "iteration_trace.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: _normalize_cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _best_scan_row(scan: dict[str, Any]) -> dict[str, Any]:
    if scan.get("best_row") is not None:
        return dict(scan["best_row"])
    rows = [
        row
        for row in scan.get("rows", [])
        if row.get("status") == "success" and row.get("total_energy_eh") is not None
    ]
    if not rows:
        raise RuntimeError("DMRG scan has no successful reference rows")
    return min(rows, key=lambda row: float(row["total_energy_eh"]))


def _reference_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.reference_total is not None:
        return {
            "scan_json": None,
            "reference_total_energy_eh": float(args.reference_total),
            "reference_label": str(args.reference_label),
            "reference_source": str(args.reference_source),
            "best_row": None,
        }
    if args.scan_json is None:
        raise RuntimeError("Provide either --reference-total or --scan-json")
    scan = _load_json(args.scan_json.resolve())
    best = _best_scan_row(scan)
    return {
        "scan_json": str(args.scan_json.resolve()),
        "reference_total_energy_eh": float(best["total_energy_eh"]),
        "reference_label": f"DMRG scan best M={int(best['max_bond_dim'])}",
        "reference_source": "dmrg_scan",
        "best_row": best,
    }


def _case_dirs(case_root: Path) -> list[Path]:
    return sorted(path for path in case_root.glob("cr2_cas12e22_R1p50_ccpvdzdk_*") if path.is_dir())


def _run_specs(case_dir: Path) -> tuple[str, dict[str, str], dict[str, dict[str, Any]]]:
    input_path = case_dir / "input.json"
    if input_path.exists():
        spec = _load_json(input_path)
        labels = {str(run["name"]): str(run["label"]) for run in spec.get("runs", [])}
        params = {str(run["name"]): dict(run["solver"]["params"]) for run in spec.get("runs", [])}
        return str(spec.get("label", case_dir.name)), labels, params
    labels: dict[str, str] = {}
    params: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(path for path in case_dir.iterdir() if path.is_dir() and (path / "iteration_trace.csv").exists()):
        labels[run_dir.name] = run_dir.name
        params[run_dir.name] = {"fixed_store_keep": 0, "fixed_propagate_keep": 0}
    return case_dir.name, labels, params


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ecore_from_case(case_dir: Path, *payloads: dict[str, Any]) -> float | None:
    for payload in payloads:
        value = payload.get("ecore_eh")
        if value is not None:
            return float(value)
    setup_path = case_dir / "system_setup" / "system_setup.json"
    if setup_path.exists():
        setup = _load_json(setup_path)
        value = setup.get("ecore_eh")
        if value is not None:
            return float(value)
    return None


def _stamp_reference_metadata(
    *,
    case_dir: Path,
    reference_total_eh: float,
    reference_payload: dict[str, Any],
) -> dict[str, Any]:
    setup_dir = case_dir / "system_setup"
    choice_path = setup_dir / "reference_choice.json"
    dmrg_path = setup_dir / "reference_dmrg_summary.json"
    if not choice_path.exists() and not dmrg_path.exists():
        return {"reference_metadata_updated": False}

    choice = _load_json(choice_path) if choice_path.exists() else {}
    dmrg = _load_json(dmrg_path) if dmrg_path.exists() else {}
    raw_reference = choice.get("reference_total_energy_eh", dmrg.get("total_energy_eh"))
    ecore = _ecore_from_case(case_dir, choice, dmrg)
    electronic = None if ecore is None else float(reference_total_eh) - float(ecore)
    label = str(reference_payload["reference_label"])
    override = {
        "new_total_energy_eh": float(reference_total_eh),
        "old_total_energy_eh": None if raw_reference is None else float(raw_reference),
        "delta_mHa_vs_old": None
        if raw_reference is None
        else float(1000.0 * (float(reference_total_eh) - float(raw_reference))),
        "source": str(reference_payload.get("reference_source", label)),
    }

    def _update_reference_row(row: dict[str, Any]) -> dict[str, Any]:
        updated = dict(row)
        previous = updated.get("total_energy_eh")
        updated["reference_kind"] = "dmrg"
        updated["reference_label"] = label
        updated["total_energy_eh"] = float(reference_total_eh)
        if electronic is not None:
            updated["electronic_energy_eh"] = float(electronic)
        if "bond_dims" not in updated:
            updated["bond_dims"] = [6000]
        updated["external_reference_override"] = {
            **override,
            "old_total_energy_eh": None if previous is None else float(previous),
            "delta_mHa_vs_old": None
            if previous is None
            else float(1000.0 * (float(reference_total_eh) - float(previous))),
        }
        updated["success"] = bool(updated.get("success", True))
        return updated

    if choice_path.exists():
        attempts = [dict(row) for row in choice.get("attempts", [])]
        if attempts:
            attempts[0] = _update_reference_row(attempts[0])
        choice.update(
            {
                "attempts": attempts,
                "chosen_reference_kind": "dmrg",
                "chosen_reference_label": label,
                "reference_total_energy_eh": float(reference_total_eh),
                "effective_reference_mode": choice.get("effective_reference_mode", "dmrg_only"),
                "external_reference_override": override,
            }
        )
        _write_json(choice_path, choice)

    if dmrg_path.exists():
        _write_json(dmrg_path, _update_reference_row(dmrg))

    return {
        "reference_metadata_updated": True,
        "reference_choice_path": str(choice_path) if choice_path.exists() else None,
        "reference_dmrg_summary_path": str(dmrg_path) if dmrg_path.exists() else None,
    }


def _refresh_summary_json(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    reference_total_eh: float,
    reference_payload: dict[str, Any],
) -> None:
    if not path.exists() or not rows:
        return
    summary = _load_json(path)
    limited_rows = rows
    latest = limited_rows[-1]
    below = [row for row in limited_rows if row.get("delta_mHa") is not None and float(row["delta_mHa"]) < -1.0e-9]
    summary.update(
        {
            "reference_total_energy_eh": float(reference_total_eh),
            "reference_kind": "dmrg",
            "reference_label": str(reference_payload["reference_label"]),
            "reference_rebase_source": str(reference_payload.get("reference_source", reference_payload["reference_label"])),
            "final_delta_mHa": float(latest["delta_mHa"]),
            "final_abs_delta_mHa": float(latest["abs_delta_mHa"]),
            "plotted_final_abs_delta_mHa": float(latest["abs_delta_mHa"]),
            "plotted_iterations": int(latest["cum_new_krylov"]),
            "trace_points": int(len(limited_rows)),
            "below_fci_detected": bool(below),
            "below_reference_points": [
                {
                    "cycle": int(row["cycle"]),
                    "cum_new_krylov": int(row["cum_new_krylov"]),
                    "total_energy_eh": float(row["total_energy_eh"]),
                    "delta_mHa": float(row["delta_mHa"]),
                    "abs_delta_mHa": float(row["abs_delta_mHa"]),
                }
                for row in below
            ],
            "best_below_reference_total_energy_eh": (
                None if not below else float(min(float(row["total_energy_eh"]) for row in below))
            ),
            "min_signed_delta_mHa": float(min(float(row["delta_mHa"]) for row in limited_rows)),
        }
    )
    _write_json(path, summary)


def _refresh_case(
    *,
    case_dir: Path,
    reference_total_eh: float,
    reference_payload: dict[str, Any],
    archive_name: str,
    update_native: bool,
) -> dict[str, Any] | None:
    case_label, run_labels, run_params = _run_specs(case_dir)
    if not run_labels:
        return None

    metadata_update = _stamp_reference_metadata(
        case_dir=case_dir,
        reference_total_eh=reference_total_eh,
        reference_payload=reference_payload,
    )
    archive_dir = case_dir / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    traces: list[tuple[str, list[dict[str, Any]]]] = []
    refreshed_runs: list[dict[str, Any]] = []

    for run_name, run_label in run_labels.items():
        run_dir = case_dir / run_name
        raw_rows = _load_rows(run_dir)
        if not raw_rows:
            continue
        rows = n2_study.rebuild_trace_error_columns(raw_rows, reference_total_eh=reference_total_eh)
        traces.append((run_label, rows))

        archive_run_dir = archive_dir / run_name
        n2_study.write_trace_outputs_to_paths(
            csv_path=archive_run_dir / "iteration_trace.csv",
            json_path=archive_run_dir / "iteration_trace.json",
            rows=rows,
        )
        params = run_params.get(run_name, {})
        title = n2_study._case_plot_title(
            case_label,
            run_label,
            int(params.get("fixed_store_keep", 0)),
            int(params.get("fixed_propagate_keep", 0)),
        )
        n2_study.plot_run_trace_to_paths(
            rows=rows,
            title=title,
            max_cumulative_iterations=max(int(row["cum_new_krylov"]) for row in rows),
            png_path=archive_run_dir / "energy_error_vs_iteration.png",
            pdf_path=archive_run_dir / "energy_error_vs_iteration.pdf",
        )
        n2_study.write_live_status_to_path(
            archive_run_dir / "live_status.json",
            n2_study.build_live_status_payload(label=run_label, latest=rows[-1]),
        )
        if update_native:
            n2_study.write_trace_outputs_to_paths(
                csv_path=run_dir / "iteration_trace.csv",
                json_path=run_dir / "iteration_trace.json",
                rows=rows,
            )
            n2_study.plot_run_trace_to_paths(
                rows=rows,
                title=title,
                max_cumulative_iterations=max(int(row["cum_new_krylov"]) for row in rows),
                png_path=run_dir / "energy_error_vs_iteration.png",
                pdf_path=run_dir / "energy_error_vs_iteration.pdf",
            )
            n2_study.write_live_status_to_path(
                run_dir / "live_status.json",
                n2_study.build_live_status_payload(label=run_label, latest=rows[-1]),
            )
            _refresh_summary_json(
                path=run_dir / "summary.json",
                rows=rows,
                reference_total_eh=reference_total_eh,
                reference_payload=reference_payload,
            )
        refreshed_runs.append(
            {
                "run_name": run_name,
                "latest_k": int(rows[-1]["cum_new_krylov"]),
                "latest_abs_delta_mHa": float(rows[-1]["abs_delta_mHa"]),
            }
        )

    if not traces:
        return None

    max_k = max(int(row["cum_new_krylov"]) for _, rows in traces for row in rows)
    comparison = n2_study.plot_case_comparison_from_traces(
        case_label=case_label,
        traces=traces,
        max_cumulative_iterations=max_k,
        png_path=archive_dir / "energy_error_vs_iteration_comparison.png",
        pdf_path=archive_dir / "energy_error_vs_iteration_comparison.pdf",
    )
    if update_native:
        n2_study.plot_case_comparison_from_traces(
            case_label=case_label,
            traces=traces,
            max_cumulative_iterations=max_k,
            png_path=case_dir / "energy_error_vs_iteration_comparison.png",
            pdf_path=case_dir / "energy_error_vs_iteration_comparison.pdf",
        )
        study_summary_path = case_dir / "study_summary.json"
        if study_summary_path.exists():
            study_summary = _load_json(study_summary_path)
            study_summary.update(
                {
                    "reference_total_energy_eh": float(reference_total_eh),
                    "reference_kind": "dmrg",
                    "reference_label": str(reference_payload["reference_label"]),
                    "reference_rebase_source": str(
                        reference_payload.get("reference_source", reference_payload["reference_label"])
                    ),
                    "comparison_plot_png": str(case_dir / "energy_error_vs_iteration_comparison.png"),
                    "comparison_plot_pdf": str(case_dir / "energy_error_vs_iteration_comparison.pdf"),
                }
            )
            _write_json(study_summary_path, study_summary)

    override = {
        **reference_payload,
        "case_dir": str(case_dir),
        "archive_dir": str(archive_dir),
        "update_native": bool(update_native),
    }
    (archive_dir / "reference_override.json").write_text(
        json.dumps(override, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (case_dir / "reference_override_best_dmrg.json").write_text(
        json.dumps(override, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "case_dir": str(case_dir),
        "archive_dir": str(archive_dir),
        "comparison_png": None if comparison is None else str(comparison[0]),
        "comparison_pdf": None if comparison is None else str(comparison[1]),
        "runs": refreshed_runs,
        **metadata_update,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference_payload = _reference_payload_from_args(args)
    reference_total = float(reference_payload["reference_total_energy_eh"])
    excluded = {str(item) for item in getattr(args, "exclude_case", [])}
    refreshed = []
    for case_dir in _case_dirs(args.case_root.resolve()):
        if case_dir.name in excluded or str(case_dir) in excluded or str(case_dir.resolve()) in excluded:
            continue
        payload = _refresh_case(
            case_dir=case_dir,
            reference_total_eh=reference_total,
            reference_payload=reference_payload,
            archive_name=str(args.archive_name),
            update_native=bool(args.update_native),
        )
        if payload is not None:
            refreshed.append(payload)
    result = {
        **reference_payload,
        "case_root": str(args.case_root.resolve()),
        "archive_name": str(args.archive_name),
        "update_native": bool(args.update_native),
        "refreshed_cases": refreshed,
    }
    out_path = args.case_root.resolve() / "cr2_reference_rebase_best_dmrg.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-json", type=Path)
    parser.add_argument("--reference-total", type=float)
    parser.add_argument("--reference-label", default="DMRG M=6000 override")
    parser.add_argument("--reference-source", default="user supplied Cr2 CAS(12e,22o) DMRG M=6000 override")
    parser.add_argument("--case-root", type=Path, default=ROOT / "inputs" / "cr2" / "1.5")
    parser.add_argument("--archive-name", default="corrected_reference_best_dmrg")
    parser.add_argument("--update-native", action="store_true")
    parser.add_argument("--exclude-case", action="append", default=[])
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
