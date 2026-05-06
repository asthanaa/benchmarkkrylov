"""Rebase Mo2 reference energies onto QP-exported FCIDUMP metadata.

This utility recomputes a corrected Mo2 reference without restarting any live
CCIK jobs. It writes corrected trace/status/plot sidecars under
`corrected_reference/` and mirrors the correction metadata into
`system_setup/reference_correction.json` for future reruns.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import debug_selector_parity as dsp  # noqa: E402
import mo2_thick_iteration_study as mo2_study  # noqa: E402
import n2_thick_iteration_study as n2_study  # noqa: E402
from qp_ccik_common import discover_qp_paths_from_root  # noqa: E402
from reference_backends import (  # noqa: E402
    DEFAULT_MAX_FILE_MB,
    DEFAULT_TOTAL_SCRATCH_GB,
    closed_shell_hf_total_energy_from_fcidump,
    load_or_compute_dmrg_reference_from_fcidump,
    load_or_compute_pyscf_selected_ci_reference_from_fcidump,
)


MO2_INPUT_ROOT = ROOT / "inputs" / "mo2" / "1.94"
HF_ALIGNMENT_TOL_MHA = 5.0


def _discover_mo2_inputs() -> list[Path]:
    return sorted(MO2_INPUT_ROOT.rglob("input.json"))


def _corrected_dir(case_dir: Path) -> Path:
    return case_dir / "corrected_reference"


def _system_reference_correction_path(case_dir: Path) -> Path:
    return case_dir / "system_setup" / "reference_correction.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress_log_path(corrected_dir: Path) -> Path:
    return corrected_dir / "rebase_progress.log"


def _log_progress(corrected_dir: Path, message: str) -> None:
    corrected_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{stamp}] {message}"
    with _progress_log_path(corrected_dir).open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _primary_run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    runs = list(spec.get("runs", []))
    if not runs:
        raise RuntimeError(f"No runs found in {spec!r}")
    return dict(runs[0])


def _load_run_rows(case_dir: Path, run_name: str) -> list[dict[str, Any]]:
    run_dir = case_dir / run_name
    _, json_path = n2_study._trace_paths(run_dir)
    rows = n2_study.load_trace_rows_from_json(json_path)
    if not rows:
        raise FileNotFoundError(f"Missing iteration trace for {run_name} in {case_dir}")
    return rows


def _reference_backend_payload(
    *,
    spec: dict[str, Any],
    qp_export_fcidump: Path,
    outdir: Path,
    force_reference: bool,
) -> tuple[dict[str, Any], str]:
    study_cfg = spec.get("study", {})
    _log_progress(outdir, f"starting DMRG reference on {qp_export_fcidump}")
    dmrg_summary = load_or_compute_dmrg_reference_from_fcidump(
        fcidump_path=qp_export_fcidump,
        outdir=outdir / "reference_dmrg",
        force=force_reference,
        memory_gb=int(study_cfg.get("dmrg_memory_gb", 256)),
        threads=int(study_cfg.get("dmrg_threads", study_cfg.get("omp_threads", 12))),
        conv_tol=float(study_cfg.get("dmrg_conv_tol", 1.0e-8)),
        dmrg_python=Path(str(study_cfg.get("dmrg_python", ROOT / ".conda" / "cka-dmrg-py312" / "bin" / "python"))),
        max_file_mb=int(study_cfg.get("dmrg_max_file_mb", DEFAULT_MAX_FILE_MB)),
        max_total_scratch_gb=float(study_cfg.get("dmrg_total_scratch_gb", DEFAULT_TOTAL_SCRATCH_GB)),
        scratch_root=Path(str(study_cfg.get("dmrg_scratch_root", "/tmp/cka_dmrg"))),
        bond_dims=None
        if study_cfg.get("dmrg_bond_dims") is None
        else tuple(int(x) for x in study_cfg.get("dmrg_bond_dims", [])),
        noises=None
        if study_cfg.get("dmrg_noises") is None
        else tuple(float(x) for x in study_cfg.get("dmrg_noises", [])),
        dav_thrds=None
        if study_cfg.get("dmrg_dav_thrds") is None
        else tuple(float(x) for x in study_cfg.get("dmrg_dav_thrds", [])),
        n_steps=int(study_cfg.get("dmrg_n_steps", 10)),
    )
    if bool(dmrg_summary.get("success")):
        _log_progress(
            outdir,
            "DMRG reference complete "
            f"(total={float(dmrg_summary['total_energy_eh']):+.12f} Eh)",
        )
        return dmrg_summary, "dmrg"

    _log_progress(
        outdir,
        "DMRG reference did not succeed; switching to PySCF selected-CI fallback",
    )
    cipsi_summary = load_or_compute_pyscf_selected_ci_reference_from_fcidump(
        fcidump_path=qp_export_fcidump,
        outdir=outdir / "reference_cipsi_fallback",
        force=force_reference,
        memory_gb=int(study_cfg.get("cipsi_memory_gb", study_cfg.get("dmrg_memory_gb", 256))),
        conv_tol=float(study_cfg.get("cipsi_conv_tol", 1.0e-10)),
        cutoff_ladder=tuple(float(x) for x in study_cfg.get("cipsi_cutoff_ladder", [5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5])),
    )
    if not bool(cipsi_summary.get("success")):
        raise RuntimeError(
            "QP-exported FCIDUMP reference failed for both DMRG and PySCF selected-CI fallback. "
            f"See {outdir / 'reference_dmrg'} and {outdir / 'reference_cipsi_fallback'}"
        )
    _log_progress(
        outdir,
        "PySCF selected-CI fallback complete "
        f"(total={float(cipsi_summary['total_energy_eh']):+.12f} Eh)",
    )
    return cipsi_summary, "pyscf_selected_ci"


def _ensure_qp_export_fcidump(
    *,
    corrected_dir: Path,
    qp: Any,
    ezfio_path: Path,
    force_export: bool,
) -> tuple[Path, dict[str, Any]]:
    qp_export_dir = corrected_dir / "qp_export"
    qp_export_dir.mkdir(parents=True, exist_ok=True)
    qp_export_fcidump = qp_export_dir / "qp_active.FCIDUMP"
    qp_summary_path = qp_export_dir / "qp_fcidump_summary.json"
    if not force_export and qp_export_fcidump.exists() and qp_summary_path.exists():
        _log_progress(
            corrected_dir,
            f"reusing existing QP export at {qp_export_fcidump}",
        )
        return qp_export_fcidump, _load_json(qp_summary_path)

    _log_progress(
        corrected_dir,
        f"starting QP FCIDUMP export from {ezfio_path}",
    )
    qp_export_fcidump = dsp._generate_qp_fcidump(qp, ezfio_path, qp_export_dir, case_spec=None)
    qp_summary = _load_json(qp_summary_path)
    _log_progress(
        corrected_dir,
        "QP FCIDUMP export complete "
        f"(norb={int(qp_summary['norb'])}, nelec={qp_summary['nelec']}, ecore={float(qp_summary['ecore']):+.12f} Eh)",
    )
    return qp_export_fcidump, qp_summary


def _compute_reference_correction(
    *,
    input_path: Path,
    spec: dict[str, Any],
    case_dir: Path,
    corrected_dir: Path,
    force_reference: bool,
) -> dict[str, Any]:
    system_setup = _load_json(case_dir / "system_setup" / "system_setup.json")
    raw_choice = _load_json(case_dir / "system_setup" / "reference_choice.json")
    primary_run = _primary_run_spec(spec)
    run_name = str(primary_run["name"])
    rows = _load_run_rows(case_dir, run_name)
    first_logged_total = float(rows[0]["total_energy_eh"])

    qp_root = mo2_study._require_qp_root(spec)
    qp = discover_qp_paths_from_root(qp_root)
    ezfio_path = Path(str(system_setup["ezfio_path"]))
    qp_export_fcidump, qp_summary = _ensure_qp_export_fcidump(
        corrected_dir=corrected_dir,
        qp=qp,
        ezfio_path=ezfio_path,
        force_export=force_reference,
    )

    expected_norb = int(system_setup["ncas"])
    expected_nelec = int(system_setup["nelecas"])
    if int(qp_summary["norb"]) != expected_norb:
        raise RuntimeError(
            f"QP-exported FCIDUMP NORB mismatch for {input_path}: expected {expected_norb}, got {qp_summary['norb']}"
        )
    qp_nelec_total = int(qp_summary["nelec"][0]) + int(qp_summary["nelec"][1])
    if qp_nelec_total != expected_nelec:
        raise RuntimeError(
            f"QP-exported FCIDUMP NELEC mismatch for {input_path}: expected {expected_nelec}, got {qp_nelec_total}"
        )

    _log_progress(
        corrected_dir,
        "validated QP-exported FCIDUMP metadata "
        f"(expected_norb={expected_norb}, expected_nelec={expected_nelec})",
    )
    reference_backend, validator_kind = _reference_backend_payload(
        spec=spec,
        qp_export_fcidump=qp_export_fcidump,
        outdir=corrected_dir,
        force_reference=force_reference,
    )
    hf_anchor = closed_shell_hf_total_energy_from_fcidump(fcidump_path=qp_export_fcidump)
    hf_anchor_total = float(hf_anchor["total_energy_eh"])
    raw_validator_total = float(reference_backend["total_energy_eh"])
    _log_progress(
        corrected_dir,
        "computed HF anchor and validator totals "
        f"(hf={hf_anchor_total:+.12f} Eh, validator={raw_validator_total:+.12f} Eh)",
    )

    anchor_mismatch_eh = float(first_logged_total - hf_anchor_total)
    anchor_mismatch_mha = 1000.0 * anchor_mismatch_eh
    alignment_shift_eh = 0.0 if abs(anchor_mismatch_mha) <= HF_ALIGNMENT_TOL_MHA else anchor_mismatch_eh
    corrected_total = float(raw_validator_total + alignment_shift_eh)
    corrected_kind = (
        f"{validator_kind}_qp_export_raw"
        if alignment_shift_eh == 0.0
        else f"{validator_kind}_qp_export_aligned_to_first_qp_point"
    )

    payload = {
        "input_path": str(input_path),
        "case_dir": str(case_dir),
        "raw_reference_total_energy_eh": float(raw_choice["reference_total_energy_eh"]),
        "qp_export_fcidump_path": str(qp_export_fcidump),
        "qp_export_norb": int(qp_summary["norb"]),
        "qp_export_nelec": [int(qp_summary["nelec"][0]), int(qp_summary["nelec"][1])],
        "qp_export_ecore_eh": float(qp_summary["ecore"]),
        "qp_export_dmrg_total_energy_eh": (
            None if validator_kind != "dmrg" else float(reference_backend["total_energy_eh"])
        ),
        "hf_anchor_total_energy_eh": float(hf_anchor_total),
        "first_logged_qp_total_energy_eh": float(first_logged_total),
        "hf_anchor_mismatch_mHa": float(anchor_mismatch_mha),
        "alignment_shift_eh": float(alignment_shift_eh),
        "corrected_reference_total_energy_eh": float(corrected_total),
        "corrected_reference_kind": corrected_kind,
        "validator_kind": validator_kind,
        "validator_total_energy_eh": float(raw_validator_total),
        "validator_payload_path": (
            str((corrected_dir / "reference_dmrg" / "reference_dmrg_summary.json"))
            if validator_kind == "dmrg"
            else str((corrected_dir / "reference_cipsi_fallback" / "reference_cipsi_summary.json"))
        ),
    }
    _log_progress(
        corrected_dir,
        "reference correction assembled "
        f"(kind={corrected_kind}, corrected_total={corrected_total:+.12f} Eh)",
    )
    return payload


def _write_reference_correction(case_dir: Path, corrected_dir: Path, payload: dict[str, Any]) -> None:
    corrected_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = corrected_dir / "reference_correction.json"
    corrected_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    system_path = _system_reference_correction_path(case_dir)
    system_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log_progress(
        corrected_dir,
        f"wrote reference correction metadata to {corrected_path} and {system_path}",
    )


def _load_reference_correction(case_dir: Path) -> dict[str, Any]:
    corrected_path = _corrected_dir(case_dir) / "reference_correction.json"
    if corrected_path.exists():
        return _load_json(corrected_path)
    system_path = _system_reference_correction_path(case_dir)
    if system_path.exists():
        return _load_json(system_path)
    raise FileNotFoundError(f"Missing reference correction metadata for {case_dir}")


def _refresh_corrected_outputs(
    *,
    spec: dict[str, Any],
    case_dir: Path,
    corrected_dir: Path,
    correction_payload: dict[str, Any],
) -> dict[str, Any]:
    case_label = str(spec["label"])
    study_cfg = spec.get("study", {})
    max_cumulative_iterations = int(study_cfg["max_cumulative_iterations"])
    plot_max_cumulative_iterations = int(study_cfg.get("plot_max_cumulative_iterations", max_cumulative_iterations))
    run_labels = {str(run["name"]): str(run["label"]) for run in spec["runs"]}
    traces: list[tuple[str, list[dict[str, Any]]]] = []
    primary_run = _primary_run_spec(spec)
    primary_rows: list[dict[str, Any]] | None = None
    _log_progress(corrected_dir, f"rebuilding corrected traces and plots for {case_label}")

    for run_spec in spec["runs"]:
        run_name = str(run_spec["name"])
        run_label = str(run_spec["label"])
        raw_rows = _load_run_rows(case_dir, run_name)
        corrected_rows = n2_study.rebuild_trace_error_columns(
            raw_rows,
            reference_total_eh=float(correction_payload["corrected_reference_total_energy_eh"]),
        )
        corrected_rows = n2_study.limit_iteration_rows(corrected_rows, max_cumulative_iterations)
        traces.append((run_label, corrected_rows))
        if run_name == str(primary_run["name"]):
            primary_rows = corrected_rows

    if not primary_rows:
        raise RuntimeError(f"Primary corrected trace missing for {case_dir}")

    primary_solver = dict(primary_run["solver"]["params"])
    title = n2_study._case_plot_title(
        case_label,
        str(primary_run["label"]),
        int(primary_solver["fixed_store_keep"]),
        int(primary_solver["fixed_propagate_keep"]),
    )
    n2_study.write_trace_outputs_to_paths(
        csv_path=corrected_dir / "iteration_trace_corrected.csv",
        json_path=corrected_dir / "iteration_trace_corrected.json",
        rows=primary_rows,
    )
    n2_study.write_live_status_to_path(
        corrected_dir / "live_status_corrected.json",
        n2_study.build_live_status_payload(label=str(primary_run["label"]), latest=primary_rows[-1]),
    )
    n2_study.plot_run_trace_to_paths(
        rows=primary_rows,
        title=title,
        max_cumulative_iterations=plot_max_cumulative_iterations,
        png_path=corrected_dir / "energy_error_vs_iteration_corrected.png",
        pdf_path=corrected_dir / "energy_error_vs_iteration_corrected.pdf",
    )
    comparison = n2_study.plot_case_comparison_from_traces(
        case_label=case_label,
        traces=traces,
        max_cumulative_iterations=plot_max_cumulative_iterations,
        png_path=corrected_dir / "energy_error_vs_iteration_comparison_corrected.png",
        pdf_path=corrected_dir / "energy_error_vs_iteration_comparison_corrected.pdf",
    )
    _log_progress(
        corrected_dir,
        "corrected trace/status/plot artifacts refreshed",
    )
    return {
        "corrected_trace_json": str(corrected_dir / "iteration_trace_corrected.json"),
        "corrected_live_status": str(corrected_dir / "live_status_corrected.json"),
        "corrected_plot_png": str(corrected_dir / "energy_error_vs_iteration_corrected.png"),
        "corrected_plot_pdf": str(corrected_dir / "energy_error_vs_iteration_corrected.pdf"),
        "corrected_comparison_png": None if comparison is None else str(comparison[0]),
        "corrected_comparison_pdf": None if comparison is None else str(comparison[1]),
    }


def probe_case_export(input_path: Path, *, force_reference: bool) -> dict[str, Any]:
    input_path = input_path.resolve()
    case_dir = input_path.parent
    corrected_dir = _corrected_dir(case_dir)
    corrected_dir.mkdir(parents=True, exist_ok=True)
    spec = n2_study.load_study_input(input_path)
    system_setup = _load_json(case_dir / "system_setup" / "system_setup.json")
    qp_root = mo2_study._require_qp_root(spec)
    qp = discover_qp_paths_from_root(qp_root)
    ezfio_path = Path(str(system_setup["ezfio_path"]))
    _log_progress(corrected_dir, f"starting export-only probe for {input_path}")
    qp_export_fcidump, qp_summary = _ensure_qp_export_fcidump(
        corrected_dir=corrected_dir,
        qp=qp,
        ezfio_path=ezfio_path,
        force_export=force_reference,
    )
    payload = {
        "input_path": str(input_path),
        "case_dir": str(case_dir),
        "export_only": True,
        "force_reference": bool(force_reference),
        "qp_export_fcidump_path": str(qp_export_fcidump),
        "qp_export_norb": int(qp_summary["norb"]),
        "qp_export_nelec": [int(qp_summary["nelec"][0]), int(qp_summary["nelec"][1])],
        "qp_export_ecore_eh": float(qp_summary["ecore"]),
    }
    probe_path = corrected_dir / "qp_export_probe.json"
    probe_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log_progress(corrected_dir, f"export-only probe complete; wrote {probe_path}")
    return payload


def rebase_case(input_path: Path, *, plots_only: bool, force_reference: bool) -> dict[str, Any]:
    input_path = input_path.resolve()
    case_dir = input_path.parent
    corrected_dir = _corrected_dir(case_dir)
    corrected_dir.mkdir(parents=True, exist_ok=True)
    spec = n2_study.load_study_input(input_path)
    _log_progress(
        corrected_dir,
        "starting full Mo2 reference rebase "
        f"(plots_only={bool(plots_only)}, force_reference={bool(force_reference)}) for {input_path}",
    )

    correction_path = corrected_dir / "reference_correction.json"
    if plots_only:
        correction_payload = _load_reference_correction(case_dir)
    elif correction_path.exists() and not force_reference:
        correction_payload = _load_reference_correction(case_dir)
    else:
        correction_payload = _compute_reference_correction(
            input_path=input_path,
            spec=spec,
            case_dir=case_dir,
            corrected_dir=corrected_dir,
            force_reference=force_reference,
        )
        _write_reference_correction(case_dir, corrected_dir, correction_payload)

    outputs = _refresh_corrected_outputs(
        spec=spec,
        case_dir=case_dir,
        corrected_dir=corrected_dir,
        correction_payload=correction_payload,
    )
    payload = {
        "input_path": str(input_path),
        "case_dir": str(case_dir),
        "plots_only": bool(plots_only),
        "force_reference": bool(force_reference),
        "corrected_reference_total_energy_eh": float(correction_payload["corrected_reference_total_energy_eh"]),
        "corrected_reference_kind": correction_payload["corrected_reference_kind"],
        "validator_kind": correction_payload["validator_kind"],
        **outputs,
    }
    (corrected_dir / "rebase_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _log_progress(corrected_dir, "full Mo2 reference rebase complete")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebase Mo2 reference traces onto QP-exported FCIDUMP references")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Case input.json to rebase")
    group.add_argument("--all-mo2-cases", action="store_true", help="Rebase every Mo2 input under inputs/mo2/1.94")
    parser.add_argument("--plots-only", action="store_true", help="Reuse existing correction metadata and rerender plots")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only probe/regenerate the QP-exported FCIDUMP for one case and stop before reference/plot rebuilding",
    )
    parser.add_argument("--force-reference", action="store_true", help="Force a fresh QP-export/DMRG recomputation")
    args = parser.parse_args(argv)

    if args.export_only and args.all_mo2_cases:
        raise ValueError("--export-only requires --input and cannot be combined with --all-mo2-cases")
    if args.export_only and args.plots_only:
        raise ValueError("--export-only cannot be combined with --plots-only")

    input_paths = _discover_mo2_inputs() if args.all_mo2_cases else [args.input.resolve()]
    if not input_paths:
        raise FileNotFoundError(f"No Mo2 inputs found under {MO2_INPUT_ROOT}")

    if args.export_only:
        results = [probe_case_export(input_paths[0], force_reference=bool(args.force_reference))]
    else:
        results = [
            rebase_case(path, plots_only=bool(args.plots_only), force_reference=bool(args.force_reference))
            for path in input_paths
        ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
