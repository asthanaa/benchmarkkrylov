from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import debug_selector_parity as dsp  # noqa: E402
import n2_qp_threshold_sweep_plots as threshold  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    CCIKParams as QPCCIKParams,
    discover_qp_paths_from_root,
    rebuild_ccik_plugin,
    resolve_omp_threads,
    set_ccik_params,
    validate_fresh_qp_root,
    write_ezfio_value,
)


FIXED_BUDGET_TEST_ROOT = threshold.THRESHOLD_TEST_ROOT
DEFAULT_STORE_KEEP_GRID = (16_384, 32_768, 65_536, 131_072, 262_144)
DEFAULT_PROPAGATE_KEEP_GRID = (128, 256, 512, 1_024, 2_048)
PLOT_FLOOR_MEH = threshold.PLOT_FLOOR_MEH

CSV_FIELDS = [
    "case",
    "reference_kind",
    "reference_label",
    "sweep_kind",
    "budget_name",
    "budget_value",
    "method",
    "config_tag",
    "fixed_store_keep",
    "fixed_propagate_keep",
    "e_total_eh",
    "delta_vs_ref_mEh",
    "abs_delta_vs_ref_mEh",
    "wall_s",
    "basis_size",
    "max_cycle",
    "restart_m_cycle",
    "restart_nroot",
    "failure",
]


def _parse_int_grid(text: str | None, default: tuple[int, ...]) -> list[int]:
    if text is None:
        return list(default)
    values: list[int] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(float(stripped))
        if value < 1:
            raise ValueError(f"Budget values must be positive integers, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Budget grid cannot be empty")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="QP fixed-budget sweeps for large donor-backed N2 active spaces")
    p.add_argument("--case", choices=sorted(threshold.CASES), default="n2_cas12_12_R2p50")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--qp-root", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--omp-threads", default="auto")
    p.add_argument("--reference-kind", choices=["dmrg", "auto"], default="dmrg")
    p.add_argument(
        "--methods",
        choices=["dense", "thick", "both"],
        default="thick",
        help='Which solver branch to run. Use "both" for the paired dense/thick plots.',
    )
    p.add_argument("--row-timeout-s", type=int, default=None)
    p.add_argument("--dmrg-memory-gb", type=int, default=64)
    p.add_argument("--dmrg-threads", type=int, default=threshold.detect_physical_cpu_count())
    p.add_argument("--dmrg-conv-tol", type=float, default=1e-8)
    p.add_argument("--dmrg-python", type=Path, default=threshold.DEFAULT_DMRG_PYTHON)
    p.add_argument("--fixed-store-keep", type=int, default=65_536)
    p.add_argument("--fixed-propagate-keep", type=int, default=512)
    p.add_argument(
        "--store-grid",
        default=None,
        help="Comma-separated fixed_store_keep sweep values. Default: 16384,32768,65536,131072,262144",
    )
    p.add_argument(
        "--propagate-grid",
        default=None,
        help="Comma-separated fixed_propagate_keep sweep values. Default: 128,256,512,1024,2048",
    )
    args = p.parse_args(argv)
    if args.outdir is None:
        args.outdir = FIXED_BUDGET_TEST_ROOT / f"n2_qp_{args.case}_fixed_budget_sweep_{stamp}"
    if int(args.fixed_store_keep) < 1 or int(args.fixed_propagate_keep) < 1:
        raise ValueError("fixed-budget anchor values must be positive")
    args.fixed_store_keep = int(args.fixed_store_keep)
    args.fixed_propagate_keep = int(args.fixed_propagate_keep)
    args.store_grid = _parse_int_grid(args.store_grid, DEFAULT_STORE_KEEP_GRID)
    args.propagate_grid = _parse_int_grid(args.propagate_grid, DEFAULT_PROPAGATE_KEEP_GRID)
    if args.row_timeout_s is not None and int(args.row_timeout_s) <= 0:
        args.row_timeout_s = None
    return args


def _row_key(row: dict[str, object]) -> tuple[str, int, str]:
    return (str(row["sweep_kind"]), int(row["budget_value"]), str(row["method"]))


def _parse_optional_float(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped)


def _parse_optional_int(text: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return None
    return int(float(stripped))


def _load_cached_rows(csv_path: Path) -> dict[tuple[str, int, str], dict[str, object]]:
    if not csv_path.exists():
        return {}
    rows: dict[tuple[str, int, str], dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = {
                "case": raw["case"],
                "reference_kind": raw["reference_kind"],
                "reference_label": raw["reference_label"],
                "sweep_kind": raw["sweep_kind"],
                "budget_name": raw["budget_name"],
                "budget_value": int(raw["budget_value"]),
                "method": raw["method"],
                "config_tag": raw["config_tag"],
                "fixed_store_keep": int(raw["fixed_store_keep"]),
                "fixed_propagate_keep": int(raw["fixed_propagate_keep"]),
                "e_total_eh": _parse_optional_float(raw["e_total_eh"]),
                "delta_vs_ref_mEh": _parse_optional_float(raw["delta_vs_ref_mEh"]),
                "abs_delta_vs_ref_mEh": _parse_optional_float(raw["abs_delta_vs_ref_mEh"]),
                "wall_s": _parse_optional_float(raw["wall_s"]),
                "basis_size": _parse_optional_int(raw["basis_size"]),
                "max_cycle": _parse_optional_int(raw["max_cycle"]),
                "restart_m_cycle": _parse_optional_int(raw["restart_m_cycle"]),
                "restart_nroot": _parse_optional_int(raw["restart_nroot"]),
                "failure": raw["failure"] or None,
            }
            rows[_row_key(row)] = row
    return rows


def _sorted_rows(rows: dict[tuple[str, int, str], dict[str, object]]) -> list[dict[str, object]]:
    return [
        rows[key]
        for key in sorted(
            rows,
            key=lambda item: (
                0 if item[0] == "fixed_store_keep" else 1,
                -int(item[1]),
                0 if item[2] == "dense" else 1,
            ),
        )
    ]


def _write_rows_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fixed_budget_text(sweep_kind: str, fixed_store_keep: int, fixed_propagate_keep: int) -> str:
    if sweep_kind == "fixed_store_keep":
        return f"fixed fixed_propagate_keep = {int(fixed_propagate_keep)}"
    return f"fixed fixed_store_keep = {int(fixed_store_keep)}"


def _build_case_matrix(
    case: threshold.SweepCaseSpec,
    dense_config: threshold.StructuralConfig,
    thick_config: threshold.StructuralConfig,
    *,
    methods: list[str],
    store_grid: list[int],
    propagate_grid: list[int],
    fixed_store_keep: int,
    fixed_propagate_keep: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config_map = {"dense": dense_config, "thick": thick_config}
    active_methods = [method for method in ("dense", "thick") if method in methods]
    for budget in store_grid:
        for method in active_methods:
            rows.append(
                {
                    "sweep_kind": "fixed_store_keep",
                    "budget_name": "fixed_store_keep",
                    "budget_value": int(budget),
                    "method": method,
                    "config": config_map[method],
                    "fixed_store_keep": int(budget),
                    "fixed_propagate_keep": int(fixed_propagate_keep),
                }
            )
    for budget in propagate_grid:
        for method in active_methods:
            rows.append(
                {
                    "sweep_kind": "fixed_propagate_keep",
                    "budget_name": "fixed_propagate_keep",
                    "budget_value": int(budget),
                    "method": method,
                    "config": config_map[method],
                    "fixed_store_keep": int(fixed_store_keep),
                    "fixed_propagate_keep": int(budget),
                }
            )
    return rows


def _execution_order(
    job_specs: list[dict[str, object]],
    *,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
) -> list[dict[str, object]]:
    anchor_map = {
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
    }
    return sorted(
        job_specs,
        key=lambda job: (
            0 if str(job["sweep_kind"]) == "fixed_store_keep" else 1,
            abs(int(job["budget_value"]) - anchor_map[str(job["sweep_kind"])]),
            -int(job["budget_value"]),
            0 if str(job["method"]) == "dense" else 1,
        ),
    )


def _run_dir_for(*, sweep_kind: str, method: str, budget: int) -> Path:
    return Path("runs") / sweep_kind / method / str(int(budget))


def _make_fixed_budget_params(
    case: threshold.SweepCaseSpec,
    config: threshold.StructuralConfig,
    *,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
) -> QPCCIKParams:
    return QPCCIKParams(
        ccik_basis_size=int(config.basis_size),
        ccik_generator_keep=1,
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_max_parents_for_expansion=0,
        ccik_include_current_support=True,
        ccik_selector_keep=1,
        ccik_external_keep=int(case.internal_external_keep),
        ccik_external_candidate_factor=1,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_compression_policy="fixed_budget",
        ccik_fixed_store_keep=int(fixed_store_keep),
        ccik_fixed_propagate_keep=int(fixed_propagate_keep),
        ccik_store_abs_tol=0.0,
        ccik_propagate_abs_tol=0.0,
        ccik_nkeep=1,
        ccik_tmp_keep_factor=1,
        ccik_preserve_prev_support=False,
        ccik_preserve_prev_support_cap=0,
        ccik_total_det_budget=0,
        ccik_conv_tol=1e-12,
        ccik_ritz_overlap_tol=1e-8,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=False,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=False,
        ccik_max_cycle=int(config.max_cycle),
        ccik_restart_m_cycle=int(config.restart_m_cycle),
        ccik_restart_nroot=int(config.restart_nroot),
        ccik_verbose=False,
    )


def _run_single_qp_job(
    *,
    case: threshold.SweepCaseSpec,
    reference: dict[str, object],
    qp_root: Path,
    config: threshold.StructuralConfig,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    run_dir: Path,
    omp_threads: int,
    timeout_s: int | None,
    force: bool,
) -> dict[str, object]:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    qp = discover_qp_paths_from_root(qp_root)
    params = _make_fixed_budget_params(
        case,
        config,
        fixed_store_keep=int(fixed_store_keep),
        fixed_propagate_keep=int(fixed_propagate_keep),
    )
    run_args = dsp.parse_args(
        [
            "--case",
            case.donor_case_name,
            "--steps",
            str(max(1, config.max_cycle if config.method == "thick" else config.basis_size - 1)),
            "--selector-mode",
            "abs_residual",
            "--matvec-backend",
            "connected_sd_fullq",
            "--compression-policy",
            "fixed_budget",
            "--fixed-store-keep",
            str(int(fixed_store_keep)),
            "--fixed-propagate-keep",
            str(int(fixed_propagate_keep)),
            "--qp-root",
            str(qp_root),
            "--omp-threads",
            str(int(omp_threads)),
        ]
    )
    runtime_dir, ezfio, _ = dsp._prepare_qp_workdir(
        run_args,
        qp,
        case_spec=dsp.DONOR_N2_CASES[case.donor_case_name],
        inject_pyscf_basis=False,
    )
    set_ccik_params(qp, ezfio, params)
    write_ezfio_value(ezfio, "ccik/ccik_m", int(params.ccik_basis_size))
    write_ezfio_value(ezfio, "ccik/ccik_debug_trace", True)
    write_ezfio_value(ezfio, "ccik/ccik_debug_trace_topn", 32)
    t0 = time.perf_counter()
    failure = None
    energy = None
    qp_summary_path = run_dir / "qp_summary.json"
    try:
        qp_summary = dsp.run_qp_trace(run_args, run_dir, prepared=(runtime_dir, ezfio), timeout_s=timeout_s)
        energy = qp_summary.get("final_total_energy")
    except Exception as exc:
        qp_summary = None
        failure = str(exc)
    wall_s = float(time.perf_counter() - t0)
    shutil.rmtree(runtime_dir, ignore_errors=True)

    if qp_summary is None and qp_summary_path.exists():
        qp_summary = json.loads(qp_summary_path.read_text(encoding="utf-8"))

    first_hq, first_next = (None, None) if qp_summary is None else threshold._first_step_meta(qp_summary)
    delta = None if energy is None else float((float(energy) - float(reference["total_energy_eh"])) * 1000.0)
    payload = {
        "case": case.name,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "method": config.method,
        "config_tag": config.tag,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "e_total_eh": None if energy is None else float(energy),
        "delta_vs_ref_mEh": delta,
        "abs_delta_vs_ref_mEh": None if delta is None else abs(float(delta)),
        "wall_s": wall_s,
        "omp_threads": int(omp_threads),
        "timeout_s": None if timeout_s is None else int(timeout_s),
        "compression_policy": params.ccik_compression_policy,
        "basis_size": int(config.basis_size),
        "max_cycle": int(config.max_cycle),
        "restart_m_cycle": int(config.restart_m_cycle),
        "restart_nroot": int(config.restart_nroot),
        "failure": failure,
        "artifacts": {
            "qp_summary": str(qp_summary_path) if qp_summary_path.exists() else None,
            "stdout": str(run_dir / "ccik_stdout.log"),
            "stderr": str(run_dir / "ccik_stderr.log"),
            "trace": str(run_dir / "ccik_debug_trace.log"),
        },
        "first_step_hq": first_hq,
        "first_step_next": first_next,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _run_preflight_gate(
    case: threshold.SweepCaseSpec,
    *,
    qp_root: Path,
    outdir: Path,
    omp_threads: int,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    force: bool,
) -> dict[str, object]:
    path = outdir / "preflight_gate.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    run_dir = outdir / "preflight" / "fixed_budget_anchor"
    cached = run_dir / "run_summary.json"
    if cached.exists() and not force:
        summary = json.loads(cached.read_text(encoding="utf-8"))
    else:
        args = argparse.Namespace(
            case=case.donor_case_name,
            steps=1,
            selector_mode="abs_residual",
            matvec_backend="connected_sd_fullq",
            compression_policy="fixed_budget",
            fixed_store_keep=int(fixed_store_keep),
            fixed_propagate_keep=int(fixed_propagate_keep),
            store_abs_tol=0.0,
            propagate_abs_tol=0.0,
            trace_topn=16,
            parity_basis_mode="qp_fcidump",
            with_qp=True,
            qp_root=qp_root,
            omp_threads=int(omp_threads),
            keep_ezfio=False,
            R=None,
            sweep=None,
        )
        donor = dsp.DONOR_N2_CASES[case.donor_case_name]
        summary = dsp._run_single_case_backend(args, donor, "connected_sd_fullq", base_outdir=run_dir)
    qp_summary = json.loads((run_dir / "qp_summary.json").read_text(encoding="utf-8"))
    payload = {
        "case": case.name,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "energy_gap_mEh": summary.get("energy_gap_mEh"),
        "final_step_projected_l2": summary.get("final_step_projected_l2"),
        "final_step_qnext_l2": summary.get("final_step_qnext_l2"),
        "observed_markers": qp_summary.get("observed_markers"),
        "trace_layout": qp_summary.get("trace_layout"),
        "stale_binary": qp_summary.get("likely_stale_generated_binary"),
        "artifacts": {
            "run_summary": str(run_dir / "run_summary.json"),
            "qp_summary": str(run_dir / "qp_summary.json"),
            "compare": str(run_dir / "python_qp_compare.json"),
        },
    }
    marker_gate = bool(all(bool(v) for v in payload["observed_markers"].values())) and bool(
        all(bool(v) for v in payload["trace_layout"].values())
    ) and not bool(payload["stale_binary"])
    numerical_gate = True
    if case.name == "n2_cas12_12_R2p50":
        numerical_gate = (
            float(payload["final_step_projected_l2"]) <= 1.0e-5
            and float(payload["final_step_qnext_l2"]) <= 1.0e-5
        )
    gate = {
        "case": case.name,
        "runs": [payload],
        "gates": {
            "marker_gate": marker_gate,
            "numerical_gate": numerical_gate,
            "overall_passed": bool(marker_gate and numerical_gate),
        },
    }
    path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def _to_log_abs(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        fval = float(value)
        if not math.isfinite(fval):
            out.append(float("nan"))
        else:
            out.append(max(abs(fval), PLOT_FLOOR_MEH))
    return out


def _write_plot(
    outdir: Path,
    *,
    case: threshold.SweepCaseSpec,
    reference_label: str,
    sweep_kind: str,
    x_values: list[int],
    dense_vals: list[float],
    thick_vals: list[float],
    dense_failed: list[bool],
    thick_failed: list[bool],
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    include_dense: bool = True,
    include_thick: bool = True,
) -> None:
    dense_plot = _to_log_abs(dense_vals)
    thick_plot = _to_log_abs(thick_vals)
    finite_vals = [value for value in dense_plot + thick_plot if math.isfinite(value)]
    if finite_vals:
        ymin = min(finite_vals)
        ymax = max(finite_vals)
        failure_y = 10.0 ** (math.ceil(math.log10(max(ymax, PLOT_FLOOR_MEH))) + 0.5)
        axis_ymin = 10.0 ** (math.floor(math.log10(max(ymin, PLOT_FLOOR_MEH))) - 0.5)
        axis_ymax = failure_y * 1.8
    else:
        failure_y = 1.0
        axis_ymin = PLOT_FLOOR_MEH
        axis_ymax = 10.0

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    if include_dense:
        ax.plot(x_values, dense_plot, marker="o", label="dense CCIK")
    if include_thick:
        ax.plot(x_values, thick_plot, marker="s", label="thick CCIK")
    dense_fail_x = [x for x, failed in zip(x_values, dense_failed, strict=True) if failed]
    thick_fail_x = [x for x, failed in zip(x_values, thick_failed, strict=True) if failed]
    if include_dense and dense_fail_x:
        ax.scatter(dense_fail_x, [failure_y] * len(dense_fail_x), marker="x", s=48, linewidths=1.5, label="dense fail")
    if include_thick and thick_fail_x:
        ax.scatter(thick_fail_x, [failure_y] * len(thick_fail_x), marker="x", s=48, linewidths=1.5, label="thick fail")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(sweep_kind)
    ax.set_ylabel(rf"$|\Delta E|$ vs {reference_label} (mEh)")
    ax.set_title(case.title)
    ax.text(
        0.03,
        0.97,
        _fixed_budget_text(sweep_kind, fixed_store_keep, fixed_propagate_keep),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    if dense_fail_x or thick_fail_x:
        ax.text(
            0.03,
            0.83,
            "x = run failed",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(axis_ymin, axis_ymax)
    fig.tight_layout()
    stem = outdir / f"{sweep_kind}_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def _write_progress(
    progress_path: Path,
    *,
    case: threshold.SweepCaseSpec,
    reference: dict[str, object],
    rows: list[dict[str, object]],
    total_rows: int,
) -> None:
    progress = {
        "case": case.name,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "completed_rows": len(rows),
        "total_rows": total_rows,
        "best_dense_abs_delta_mEh": min(
            (float(r["abs_delta_vs_ref_mEh"]) for r in rows if r["method"] == "dense" and r["abs_delta_vs_ref_mEh"] is not None),
            default=None,
        ),
        "best_thick_abs_delta_mEh": min(
            (float(r["abs_delta_vs_ref_mEh"]) for r in rows if r["method"] == "thick" and r["abs_delta_vs_ref_mEh"] is not None),
            default=None,
        ),
    }
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary(
    outdir: Path,
    *,
    case: threshold.SweepCaseSpec,
    reference: dict[str, object],
    calibration: dict[str, object],
    rows: list[dict[str, object]],
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    store_grid: list[int],
    propagate_grid: list[int],
    methods: list[str],
) -> None:
    payload = {
        "case": case.name,
        "title": case.title,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "methods": list(methods),
        "reference_total_energy_eh": float(reference["total_energy_eh"]),
        "fixed_store_keep_anchor": int(fixed_store_keep),
        "fixed_propagate_keep_anchor": int(fixed_propagate_keep),
        "store_grid": [int(v) for v in store_grid],
        "propagate_grid": [int(v) for v in propagate_grid],
        "calibration": calibration,
        "completed_rows": len(rows),
        "best_dense_abs_delta_mEh": min(
            (float(r["abs_delta_vs_ref_mEh"]) for r in rows if r["method"] == "dense" and r["abs_delta_vs_ref_mEh"] is not None),
            default=None,
        ),
        "best_thick_abs_delta_mEh": min(
            (float(r["abs_delta_vs_ref_mEh"]) for r in rows if r["method"] == "thick" and r["abs_delta_vs_ref_mEh"] is not None),
            default=None,
        ),
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_outputs(
    *,
    outdir: Path,
    case: threshold.SweepCaseSpec,
    reference: dict[str, object],
    calibration: dict[str, object],
    rows_by_key: dict[tuple[str, int, str], dict[str, object]],
    total_rows: int,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    store_grid: list[int],
    propagate_grid: list[int],
    methods: list[str],
    csv_path: Path,
    progress_path: Path,
) -> None:
    rows = _sorted_rows(rows_by_key)
    _write_rows_csv(csv_path, rows)
    _write_progress(progress_path, case=case, reference=reference, rows=rows, total_rows=total_rows)

    active_methods = set(methods)
    for sweep_kind, x_values in (
        ("fixed_store_keep", store_grid),
        ("fixed_propagate_keep", propagate_grid),
    ):
        dense_vals: list[float] = []
        thick_vals: list[float] = []
        dense_failed: list[bool] = []
        thick_failed: list[bool] = []
        for budget in x_values:
            dense_row = rows_by_key.get((sweep_kind, int(budget), "dense"))
            thick_row = rows_by_key.get((sweep_kind, int(budget), "thick"))
            dense_vals.append(
                float("nan")
                if dense_row is None or dense_row["abs_delta_vs_ref_mEh"] is None
                else float(dense_row["abs_delta_vs_ref_mEh"])
            )
            thick_vals.append(
                float("nan")
                if thick_row is None or thick_row["abs_delta_vs_ref_mEh"] is None
                else float(thick_row["abs_delta_vs_ref_mEh"])
            )
            dense_failed.append(bool(dense_row and dense_row["failure"]))
            thick_failed.append(bool(thick_row and thick_row["failure"]))
        if "dense" not in active_methods:
            dense_vals = [float("nan")] * len(x_values)
            dense_failed = [False] * len(x_values)
        if "thick" not in active_methods:
            thick_vals = [float("nan")] * len(x_values)
            thick_failed = [False] * len(x_values)
        _write_plot(
            outdir,
            case=case,
            reference_label=str(reference["reference_label"]),
            sweep_kind=sweep_kind,
            x_values=[int(x) for x in x_values],
            dense_vals=dense_vals,
            thick_vals=thick_vals,
            dense_failed=dense_failed,
            thick_failed=thick_failed,
            fixed_store_keep=fixed_store_keep,
            fixed_propagate_keep=fixed_propagate_keep,
            include_dense="dense" in active_methods,
            include_thick="thick" in active_methods,
        )

    _write_summary(
        outdir,
        case=case,
        reference=reference,
        calibration=calibration,
        rows=rows,
        fixed_store_keep=fixed_store_keep,
        fixed_propagate_keep=fixed_propagate_keep,
        store_grid=store_grid,
        propagate_grid=propagate_grid,
        methods=methods,
    )


def _load_or_run_calibration_for_methods(
    *,
    case: threshold.SweepCaseSpec,
    reference: dict[str, object],
    qp_root: Path,
    outdir: Path,
    omp_threads: int,
    force: bool,
    methods: list[str],
) -> dict[str, object]:
    path = outdir / "calibration.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    payload: dict[str, object] = {"calibration_mode": "surrogate_topk"}

    if "dense" in methods:
        dense_choice, dense_results = threshold._select_config(
            case=case,
            method="dense",
            ladder=threshold._dense_ladder(case),
            reference=reference,
            qp_root=qp_root,
            outdir=outdir,
            omp_threads=omp_threads,
            force=force,
        )
        payload["dense"] = {
            "chosen": {
                "tag": dense_choice.tag,
                "basis_size": dense_choice.basis_size,
                "max_cycle": dense_choice.max_cycle,
                "restart_m_cycle": dense_choice.restart_m_cycle,
                "restart_nroot": dense_choice.restart_nroot,
            },
            "runs": dense_results,
        }
    else:
        payload["dense"] = {"chosen": None, "runs": []}

    if "thick" in methods:
        thick_choice, thick_results = threshold._select_config(
            case=case,
            method="thick",
            ladder=threshold._thick_ladder(case),
            reference=reference,
            qp_root=qp_root,
            outdir=outdir,
            omp_threads=omp_threads,
            force=force,
        )
        payload["thick"] = {
            "chosen": {
                "tag": thick_choice.tag,
                "basis_size": thick_choice.basis_size,
                "max_cycle": thick_choice.max_cycle,
                "restart_m_cycle": thick_choice.restart_m_cycle,
                "restart_nroot": thick_choice.restart_nroot,
            },
            "runs": thick_results,
        }
    else:
        payload["thick"] = {"chosen": None, "runs": []}

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = threshold.CASES[args.case]
    threshold._validate_case_files(case)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    qp_root = threshold._require_qp_root(args.qp_root)
    omp_threads = resolve_omp_threads(args.omp_threads)
    active_methods = ["dense", "thick"] if args.methods == "both" else [args.methods]
    qp = discover_qp_paths_from_root(qp_root)
    rebuild_summary = rebuild_ccik_plugin(qp, clean=False, sync_workspace=True)
    (outdir / "qp_plugin_rebuild.json").write_text(
        json.dumps(
            {
                "plugin_dir": rebuild_summary["plugin_dir"],
                "binary_path": rebuild_summary["binary_path"],
                "binary_exists": rebuild_summary["binary_exists"],
                "command": rebuild_summary["command"],
                "sync_summary": rebuild_summary["sync_summary"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    qp_root_validation = validate_fresh_qp_root(qp_root, outdir=outdir / "qp_root_validation")
    (outdir / "qp_root_validation.json").write_text(
        json.dumps(qp_root_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not qp_root_validation["passed"]:
        raise RuntimeError(
            f"Fresh-root validation failed for {qp_root}. See {outdir / 'qp_root_validation.json'} for details."
        )

    if not args.skip_preflight:
        preflight = _run_preflight_gate(
            case,
            qp_root=qp_root,
            outdir=outdir,
            omp_threads=omp_threads,
            fixed_store_keep=args.fixed_store_keep,
            fixed_propagate_keep=args.fixed_propagate_keep,
            force=args.force,
        )
        if not preflight["gates"]["overall_passed"]:
            raise RuntimeError(f"Preflight gate failed for {case.name}. See {outdir / 'preflight_gate.json'} for details.")

    reference = threshold._load_or_compute_reference(
        case=case,
        outdir=outdir,
        force=args.force,
        dmrg_memory_gb=args.dmrg_memory_gb,
        dmrg_threads=args.dmrg_threads,
        dmrg_conv_tol=args.dmrg_conv_tol,
        dmrg_python=args.dmrg_python,
        reference_kind=args.reference_kind,
    )
    calibration = _load_or_run_calibration_for_methods(
        case=case,
        reference=reference,
        qp_root=qp_root,
        outdir=outdir,
        omp_threads=omp_threads,
        force=args.force,
        methods=active_methods,
    )
    dense_selected = calibration.get("dense", {}).get("chosen")
    thick_selected = calibration.get("thick", {}).get("chosen")
    dense_choice = (
        threshold.StructuralConfig(method="dense", **dense_selected)
        if dense_selected is not None
        else threshold.StructuralConfig(method="dense", tag="unused_dense", basis_size=1, max_cycle=1, restart_m_cycle=0, restart_nroot=2)
    )
    thick_choice = (
        threshold.StructuralConfig(method="thick", **thick_selected)
        if thick_selected is not None
        else threshold.StructuralConfig(method="thick", tag="unused_thick", basis_size=1, max_cycle=1, restart_m_cycle=1, restart_nroot=1)
    )

    csv_path = outdir / "fixed_budget_sweep.csv"
    progress_path = outdir / "progress.json"
    rows_by_key = {} if args.force else _load_cached_rows(csv_path)
    job_specs = _build_case_matrix(
        case,
        dense_choice,
        thick_choice,
        methods=active_methods,
        store_grid=args.store_grid,
        propagate_grid=args.propagate_grid,
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
    )
    ordered_jobs = _execution_order(
        job_specs,
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
    )

    _refresh_outputs(
        outdir=outdir,
        case=case,
        reference=reference,
        calibration=calibration,
        rows_by_key=rows_by_key,
        total_rows=len(job_specs),
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
        store_grid=args.store_grid,
        propagate_grid=args.propagate_grid,
        methods=active_methods,
        csv_path=csv_path,
        progress_path=progress_path,
    )

    for job in ordered_jobs:
        key = (str(job["sweep_kind"]), int(job["budget_value"]), str(job["method"]))
        if key in rows_by_key and not args.force:
            continue
        config = job["config"]
        run_dir = outdir / _run_dir_for(
            sweep_kind=str(job["sweep_kind"]),
            method=str(job["method"]),
            budget=int(job["budget_value"]),
        )
        result = _run_single_qp_job(
            case=case,
            reference=reference,
            qp_root=qp_root,
            config=config,
            fixed_store_keep=int(job["fixed_store_keep"]),
            fixed_propagate_keep=int(job["fixed_propagate_keep"]),
            run_dir=run_dir,
            omp_threads=omp_threads,
            timeout_s=args.row_timeout_s,
            force=args.force,
        )
        rows_by_key[key] = {
            "case": case.name,
            "reference_kind": reference["reference_kind"],
            "reference_label": reference["reference_label"],
            "sweep_kind": str(job["sweep_kind"]),
            "budget_name": str(job["budget_name"]),
            "budget_value": int(job["budget_value"]),
            "method": str(job["method"]),
            "config_tag": str(config.tag),
            "fixed_store_keep": int(job["fixed_store_keep"]),
            "fixed_propagate_keep": int(job["fixed_propagate_keep"]),
            "e_total_eh": result["e_total_eh"],
            "delta_vs_ref_mEh": result["delta_vs_ref_mEh"],
            "abs_delta_vs_ref_mEh": result["abs_delta_vs_ref_mEh"],
            "wall_s": result["wall_s"],
            "basis_size": int(config.basis_size),
            "max_cycle": int(config.max_cycle),
            "restart_m_cycle": int(config.restart_m_cycle),
            "restart_nroot": int(config.restart_nroot),
            "failure": result["failure"],
        }
        _refresh_outputs(
            outdir=outdir,
            case=case,
            reference=reference,
            calibration=calibration,
            rows_by_key=rows_by_key,
            total_rows=len(job_specs),
            fixed_store_keep=args.fixed_store_keep,
            fixed_propagate_keep=args.fixed_propagate_keep,
            store_grid=args.store_grid,
            propagate_grid=args.propagate_grid,
            methods=active_methods,
            csv_path=csv_path,
            progress_path=progress_path,
        )

    _refresh_outputs(
        outdir=outdir,
        case=case,
        reference=reference,
        calibration=calibration,
        rows_by_key=rows_by_key,
        total_rows=len(job_specs),
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
        store_grid=args.store_grid,
        propagate_grid=args.propagate_grid,
        csv_path=csv_path,
        progress_path=progress_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
