"""Run one live-updating N2 thick-restart study from an `inputs/*/input.json`.

The study runner is intentionally source-driven:

- `input.json` describes the molecule, active space, budgets, and reference
  policy
- generated logs, plots, traces, and QP runtime folders are local-only output
- cumulative Krylov counts are written explicitly so plots stay readable across
  thick restarts
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import debug_selector_parity as dsp  # noqa: E402
from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense_thick_restart,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKThickRestartParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from ccik.thick_restart import ThickRestartIterationRecord  # noqa: E402
from n2_utils import make_n2_atom  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    CCIKParams as QPCCIKParams,
    _with_qp_build_env,
    discover_qp_paths_from_root,
    ensure_ezfio_defaults,
    ensure_ezfio_workdir,
    fixed_budget_bridge_kwargs,
    qp_parallel_env,
    run_qp_setup,
    set_ccik_params,
    validate_rebuildable_qp_root,
    write_ezfio_value,
)
from reference_backends import (  # noqa: E402
    DEFAULT_CIPSI_RUNGS,
    DEFAULT_PYSCF_CIPSI_CUTOFFS,
    DEFAULT_CIPSI_SCRATCH_ROOT,
    DEFAULT_DMRG_PYTHON,
    DEFAULT_DMRG_SCRATCH_ROOT,
    DEFAULT_MAX_FILE_MB,
    DEFAULT_TOTAL_SCRATCH_GB,
    load_or_compute_dmrg_reference_from_fcidump,
    load_or_compute_pyscf_selected_ci_reference_from_fcidump,
)


QP_ROOT_CANDIDATES = (
    Path("/home/ayush.asthana/quantum_package/qp2"),
    Path("/Users/ayush.asthana/quantum_package/qp2_ccik_pristine_20260325"),
    Path("/Users/ayush.asthana/quantum_package/qp2_ccik_20260325"),
    Path("/private/tmp/qp2_fresh_20260320"),
)

QP_N2_SETUP_DONOR_CANDIDATES = (
    ROOT
    / "docs"
    / "assets"
    / "pub"
    / "n2_abs_residual_qp_thick_cas12_qp8_recheck_20260402"
    / "systems"
    / "n2_cas12_12_R2p50"
    / "system_setup"
    / "n2.ezfio",
    ROOT / "inputs" / "n2" / "2.0" / "n2_cas8_8_R2p00_m10_c11_r3" / "system_setup" / "n2.ezfio",
)

QP_THICK_LINE_RE = re.compile(
    r"ccik\(thick\)\s+cycle=\s*(?P<cycle>\d+)\s+iter=\s*(?P<iter>\d+)\s+"
    r"total=\s*(?P<total>[+-]?\d+\.\d+)\s+ref=\s*(?P<ref>[+-]?\d+\.\d+)\s+"
    r"delta_mEh=\s*(?P<delta>[+-]?\d+\.\d+)"
)

QP_BASELINE_LINE_RE = re.compile(
    r"ccik\(baseline\)\s+iter=\s*(?P<iter>\d+)\s+"
    r"total=\s*(?P<total>[+-]?\d+\.\d+)\s+ref=\s*(?P<ref>[+-]?\d+\.\d+)\s+"
    r"delta_mEh=\s*(?P<delta>[+-]?\d+\.\d+)"
)

QP_EXACTIZED_TRACE_RE = re.compile(
    r"CCIK_TRACE_EXACTIZED\s+"
    r"cycle=(?P<cycle>\d+)\s+"
    r"iter=(?P<iter>\d+)\s+"
    r"k=(?P<k>\d+)\s+"
    r"support=(?P<support>\d+)\s+"
    r"electronic=(?P<electronic>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"total=(?P<total>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"ref=(?P<ref>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"delta_mEh=(?P<delta>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"solver=(?P<solver>\S+)\s+"
    r"iters=(?P<iters>\d+)\s+"
    r"matvecs=(?P<matvecs>\d+)\s+"
    r"converged=(?P<converged>[TF])\s+"
    r"resid=(?P<resid>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"t_union_s=(?P<t_union>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"t_prep_s=(?P<t_prep>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"t_solve_s=(?P<t_solve>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"t_wall_s=(?P<t_wall>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)\s+"
    r"t_cpu_s=(?P<t_cpu>[+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)"
)

FINAL_VALUE_RE = re.compile(r"(?P<key>CCIK_FINAL_[A-Z_]+)\s*=\s*(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


class _StudyEarlyStop(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedQpCase:
    qp_root: Path
    setup_dir: Path
    ezfio_path: Path
    fcidump_path: Path
    ecore: float
    norb: int
    nelec: tuple[int, int]
    reference_total_eh: float
    reference_summary: dict[str, Any]
    reference_mode: str
    omp_threads: int


def load_study_input(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_input_files(inputs_dir: Path) -> list[Path]:
    return sorted(inputs_dir.rglob("input.json"))


def limit_iteration_rows(rows: list[dict[str, Any]], max_cumulative_iterations: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["cum_new_krylov"]) <= int(max_cumulative_iterations)]


def _abs_error_mha(total_energy_eh: float, reference_total_eh: float) -> float:
    return 1000.0 * abs(float(total_energy_eh) - float(reference_total_eh))


def _signed_error_mha(total_energy_eh: float, reference_total_eh: float) -> float:
    return 1000.0 * (float(total_energy_eh) - float(reference_total_eh))


def _case_dir_from_input(path: Path) -> Path:
    return path.parent


def _run_dir(case_dir: Path, run_name: str) -> Path:
    return case_dir / run_name


def _create_xyz(path: Path, atom: str, *, comment: str) -> None:
    lines = [line.strip() for line in atom.strip().splitlines() if line.strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{len(lines)}\n{comment}\n" + "\n".join(lines) + "\n", encoding="ascii")


def _trace_paths(run_dir: Path) -> tuple[Path, Path]:
    return run_dir / "iteration_trace.csv", run_dir / "iteration_trace.json"


def _runtime_ezfio_path(run_dir: Path, ezfio_name: str) -> Path:
    return run_dir / "runtime" / ezfio_name


def write_trace_outputs_to_paths(
    *,
    csv_path: Path,
    json_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "cycle",
                "basis_size",
                "restart_seed_count",
                "cum_new_krylov",
                "support_union_size",
                "electronic_energy_eh",
                "total_energy_eh",
                "delta_mHa",
                "abs_delta_mHa",
                "exactized_electronic_energy_eh",
                "exactized_total_energy_eh",
                "exactized_delta_mHa",
                "exactized_abs_delta_mHa",
                "exactized_solver_kind",
                "exactized_subspace_dim",
                "exactized_hamiltonian_nnz",
                "exactizeplus_electronic_energy_eh",
                "exactizeplus_total_energy_eh",
                "exactizeplus_delta_mHa",
                "exactizeplus_abs_delta_mHa",
                "exactizeplus_solver_kind",
                "exactizeplus_subspace_dim",
                "exactizeplus_parent_count",
                "exactizeplus_shell_added",
                "exactized_residual_norm",
                "exactized_davidson_iterations",
                "exactized_matvec_count",
                "exactized_converged",
                "exactized_support_build_s",
                "exactized_prep_s",
                "exactized_solve_s",
                "exactized_wall_s",
                "exactized_cpu_s",
                "elapsed_s",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_trace_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = _trace_paths(run_dir)
    write_trace_outputs_to_paths(csv_path=csv_path, json_path=json_path, rows=rows)


def load_trace_rows_from_json(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text(encoding="utf-8"))


def rebuild_trace_error_columns(rows: list[dict[str, Any]], *, reference_total_eh: float) -> list[dict[str, Any]]:
    corrected_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        total = updated.get("total_energy_eh")
        updated["delta_mHa"] = None if total is None else _signed_error_mha(float(total), reference_total_eh)
        updated["abs_delta_mHa"] = None if total is None else _abs_error_mha(float(total), reference_total_eh)
        for prefix in ("exactized", "exactizeplus"):
            exact_total = updated.get(f"{prefix}_total_energy_eh")
            updated[f"{prefix}_delta_mHa"] = (
                None if exact_total is None else _signed_error_mha(float(exact_total), reference_total_eh)
            )
            updated[f"{prefix}_abs_delta_mHa"] = (
                None if exact_total is None else _abs_error_mha(float(exact_total), reference_total_eh)
            )
        corrected_rows.append(updated)
    return corrected_rows


def build_live_status_payload(*, label: str, latest: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "cum_new_krylov": int(latest["cum_new_krylov"]),
        "cycle": int(latest["cycle"]),
        "basis_size": int(latest["basis_size"]),
        "support_union_size": None if latest.get("support_union_size") is None else int(latest["support_union_size"]),
        "total_energy_eh": float(latest["total_energy_eh"]),
        "delta_mHa": float(latest["delta_mHa"]),
        "abs_delta_mHa": float(latest["abs_delta_mHa"]),
        "exactized_total_energy_eh": (
            None if latest.get("exactized_total_energy_eh") is None else float(latest["exactized_total_energy_eh"])
        ),
        "exactized_abs_delta_mHa": (
            None if latest.get("exactized_abs_delta_mHa") is None else float(latest["exactized_abs_delta_mHa"])
        ),
        "exactized_solver_kind": (
            None if latest.get("exactized_solver_kind") is None else str(latest["exactized_solver_kind"])
        ),
        "exactizeplus_total_energy_eh": (
            None if latest.get("exactizeplus_total_energy_eh") is None else float(latest["exactizeplus_total_energy_eh"])
        ),
        "exactizeplus_abs_delta_mHa": (
            None if latest.get("exactizeplus_abs_delta_mHa") is None else float(latest["exactizeplus_abs_delta_mHa"])
        ),
        "exactizeplus_solver_kind": (
            None if latest.get("exactizeplus_solver_kind") is None else str(latest["exactizeplus_solver_kind"])
        ),
    }


def write_live_status_to_path(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_live_status(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_live_status_to_path(run_dir / "live_status.json", payload)


def _case_plot_title(case_label: str, run_label: str, fixed_store_keep: int, fixed_propagate_keep: int) -> str:
    return f"{case_label} | {run_label} | Kprop={fixed_propagate_keep} Kstore={fixed_store_keep}"


def plot_run_trace_to_paths(
    *,
    rows: list[dict[str, Any]],
    title: str,
    max_cumulative_iterations: int,
    png_path: Path,
    pdf_path: Path,
) -> tuple[Path, Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    xvals = [int(row["cum_new_krylov"]) for row in rows]
    yvals = [max(1.0e-12, float(row["abs_delta_mHa"])) for row in rows]
    latest = rows[-1]
    exactized_points = [
        (int(row["cum_new_krylov"]), max(1.0e-12, float(row["exactized_abs_delta_mHa"])))
        for row in rows
        if row.get("exactized_abs_delta_mHa") is not None
    ]
    exactizeplus_points = [
        (int(row["cum_new_krylov"]), max(1.0e-12, float(row["exactizeplus_abs_delta_mHa"])))
        for row in rows
        if row.get("exactizeplus_abs_delta_mHa") is not None
    ]
    below_reference_points = [
        (int(row["cum_new_krylov"]), max(1.0e-12, float(row["abs_delta_mHa"])))
        for row in rows
        if row.get("delta_mHa") is not None and float(row["delta_mHa"]) < -1.0e-9
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(
        xvals,
        yvals,
        color="#0B5FFF",
        linewidth=1.6,
        marker="o",
        markersize=2.8,
        label="Standard CCIK",
    )
    if exactized_points:
        ax.plot(
            [point[0] for point in exactized_points],
            [point[1] for point in exactized_points],
            color="#E67E22",
            linewidth=1.6,
            marker="s",
            markersize=2.6,
            label="CKCI-0",
        )
    if exactizeplus_points:
        ax.plot(
            [point[0] for point in exactizeplus_points],
            [point[1] for point in exactizeplus_points],
            color="#1B9E77",
            linewidth=1.6,
            marker="^",
            markersize=2.8,
            label="Exactizeplus checkpoint",
        )
    if below_reference_points:
        ax.scatter(
            [point[0] for point in below_reference_points],
            [point[1] for point in below_reference_points],
            color="#C00000",
            linewidths=1.2,
            s=42,
            marker="x",
            zorder=6,
            label="Below reference",
        )
    ax.set_yscale("log")
    ax.set_xlim(0, int(max_cumulative_iterations))
    ax.set_xlabel("Cumulative new Krylov vectors")
    ax.set_ylabel("|Energy error| (mHa)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.axhline(1.0, color="#C0392B", linestyle="--", linewidth=1.1, alpha=0.8, label="Chemical accuracy (1 mHa)")
    ax.annotate(
        f"latest: k={latest['cum_new_krylov']}\n|ΔE|={float(latest['abs_delta_mHa']):.6f} mHa",
        xy=(int(latest["cum_new_krylov"]), max(1.0e-12, float(latest["abs_delta_mHa"]))),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
    )
    ax.legend()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_run_trace(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    title: str,
    max_cumulative_iterations: int,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    return plot_run_trace_to_paths(
        rows=rows,
        title=title,
        max_cumulative_iterations=max_cumulative_iterations,
        png_path=run_dir / "energy_error_vs_iteration.png",
        pdf_path=run_dir / "energy_error_vs_iteration.pdf",
    )


def _load_existing_trace_rows(run_dir: Path) -> list[dict[str, Any]]:
    _, json_path = _trace_paths(run_dir)
    return load_trace_rows_from_json(json_path)


def plot_case_comparison_from_traces(
    *,
    case_label: str,
    traces: list[tuple[str, list[dict[str, Any]]]],
    max_cumulative_iterations: int,
    png_path: Path,
    pdf_path: Path,
) -> tuple[Path, Path] | None:
    if not traces:
        return None

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    palette = ["#0B5FFF", "#E67E22", "#1B9E77", "#8E44AD"]
    for idx, (run_label, rows) in enumerate(traces):
        xvals = [int(row["cum_new_krylov"]) for row in rows]
        yvals = [max(1.0e-12, float(row["abs_delta_mHa"])) for row in rows]
        ax.plot(
            xvals,
            yvals,
            label=run_label,
            color=palette[idx % len(palette)],
            linewidth=1.6,
            marker="o",
            markersize=2.6,
        )
        below_reference_points = [
            (int(row["cum_new_krylov"]), max(1.0e-12, float(row["abs_delta_mHa"])))
            for row in rows
            if row.get("delta_mHa") is not None and float(row["delta_mHa"]) < -1.0e-9
        ]
        if below_reference_points:
            ax.scatter(
                [point[0] for point in below_reference_points],
                [point[1] for point in below_reference_points],
                color="#C00000",
                linewidths=1.2,
                s=42,
                marker="x",
                zorder=6,
                label=f"{run_label} | below reference",
            )
        if len(traces) == 1:
            exactized_points = [
                (int(row["cum_new_krylov"]), max(1.0e-12, float(row["exactized_abs_delta_mHa"])))
                for row in rows
                if row.get("exactized_abs_delta_mHa") is not None
            ]
            if exactized_points:
                ax.plot(
                    [point[0] for point in exactized_points],
                    [point[1] for point in exactized_points],
                    label=f"{run_label} | CKCI-0",
                    color="#E67E22",
                    linewidth=1.5,
                    marker="s",
                    markersize=2.4,
                )
            exactizeplus_points = [
                (int(row["cum_new_krylov"]), max(1.0e-12, float(row["exactizeplus_abs_delta_mHa"])))
                for row in rows
                if row.get("exactizeplus_abs_delta_mHa") is not None
            ]
            if exactizeplus_points:
                ax.plot(
                    [point[0] for point in exactizeplus_points],
                    [point[1] for point in exactizeplus_points],
                    label=f"{run_label} | exactizeplus checkpoint",
                    color="#1B9E77",
                    linewidth=1.5,
                    marker="^",
                    markersize=2.5,
                )
    ax.set_yscale("log")
    ax.set_xlim(0, int(max_cumulative_iterations))
    ax.set_xlabel("Cumulative new Krylov vectors")
    ax.set_ylabel("|Energy error| (mHa)")
    if len(traces) == 1:
        ax.set_title(f"{case_label} | {traces[0][0]}")
    else:
        ax.set_title(f"{case_label} | parameter sweep")
    ax.grid(True, which="both", alpha=0.25)
    ax.axhline(1.0, color="#C0392B", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.legend()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_case_comparison(
    *,
    case_dir: Path,
    case_label: str,
    run_labels: dict[str, str],
    max_cumulative_iterations: int,
) -> tuple[Path, Path] | None:
    traces: list[tuple[str, list[dict[str, Any]]]] = []
    for run_name, run_label in run_labels.items():
        rows = _load_existing_trace_rows(_run_dir(case_dir, run_name))
        if rows:
            traces.append((run_label, rows))
    return plot_case_comparison_from_traces(
        case_label=case_label,
        traces=traces,
        max_cumulative_iterations=max_cumulative_iterations,
        png_path=case_dir / "energy_error_vs_iteration_comparison.png",
        pdf_path=case_dir / "energy_error_vs_iteration_comparison.pdf",
    )


def _prepare_case(spec: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, float, int, tuple[int, int], float]:
    molecule = spec["molecule"]
    cas_cfg = spec["cas"]
    mol = make_mol_pyscf(
        atom=make_n2_atom(float(molecule["distance_ang"])),
        basis=str(molecule["basis"]),
        unit=str(molecule.get("unit", "Angstrom")),
        charge=int(molecule.get("charge", 0)),
        spin=int(molecule.get("spin", 0)),
        verbose=int(molecule.get("verbose", 0)),
    )
    cas = CASSpec(
        ncas=int(cas_cfg["ncas"]),
        nelecas=int(cas_cfg["nelecas"]),
        ncore=int(cas_cfg.get("ncore", 0)),
    )
    h1, eri8, ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, norb, nelec))
    return mol, h1, eri8, float(ecore), int(norb), nelec, e_fci_total


def _requested_reference_mode(spec: dict[str, Any]) -> str:
    return str(spec.get("study", {}).get("reference_mode", "auto")).strip().lower()


def _effective_reference_mode(spec: dict[str, Any]) -> str:
    requested = _requested_reference_mode(spec)
    if requested != "auto":
        return requested
    cas_cfg = spec["cas"]
    # Small active spaces can still use exact CAS-FCI cheaply. Larger cases use
    # DMRG first and fall back to bounded selected-CI if DMRG is unavailable.
    active_scale = max(int(cas_cfg["ncas"]), int(cas_cfg["nelecas"]))
    return "dmrg_then_cipsi" if active_scale >= 14 else "exact_fci"


def _load_or_compute_exact_reference_from_hamiltonian(
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float,
    outdir: Path,
    force: bool,
) -> dict[str, Any]:
    summary_path = outdir / "reference_exact_fci_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    total_energy = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, int(norb), tuple(nelec)))
    payload = {
        "success": True,
        "reference_kind": "exact_fci",
        "reference_label": "Exact FCI",
        "total_energy_eh": total_energy,
        "electronic_energy_eh": float(total_energy - float(ecore)),
        "ecore_eh": float(ecore),
        "norb": int(norb),
        "nelec": [int(nelec[0]), int(nelec[1])],
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _resolve_qp_reference(
    *,
    spec: dict[str, Any],
    qp_root: Path,
    ezfio_path: Path,
    fcidump_path: Path,
    setup_dir: Path,
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float,
    omp_threads: int,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    study_cfg = spec.get("study", {})
    requested_mode = _requested_reference_mode(spec)
    effective_mode = _effective_reference_mode(spec)
    choice_path = setup_dir / "reference_choice.json"

    # Large-case inputs can omit `dmrg_python`; the runner will use the local
    # reproducible bootstrap path under `.conda/`.
    dmrg_python = Path(str(study_cfg.get("dmrg_python", DEFAULT_DMRG_PYTHON)))
    dmrg_memory_gb = int(study_cfg.get("dmrg_memory_gb", 64))
    dmrg_threads = int(study_cfg.get("dmrg_threads", omp_threads))
    dmrg_conv_tol = float(study_cfg.get("dmrg_conv_tol", 1.0e-8))
    dmrg_bond_dims = study_cfg.get("dmrg_bond_dims")
    dmrg_noises = study_cfg.get("dmrg_noises")
    dmrg_dav_thrds = study_cfg.get("dmrg_dav_thrds")
    dmrg_n_steps = int(study_cfg.get("dmrg_n_steps", 40))
    dmrg_total_scratch_gb = float(study_cfg.get("dmrg_total_scratch_gb", DEFAULT_TOTAL_SCRATCH_GB))
    dmrg_max_file_mb = int(study_cfg.get("dmrg_max_file_mb", DEFAULT_MAX_FILE_MB))
    dmrg_scratch_root = Path(str(study_cfg.get("dmrg_scratch_root", DEFAULT_DMRG_SCRATCH_ROOT)))
    cipsi_total_scratch_gb = float(study_cfg.get("cipsi_total_scratch_gb", DEFAULT_TOTAL_SCRATCH_GB))
    cipsi_max_file_mb = int(study_cfg.get("cipsi_max_file_mb", DEFAULT_MAX_FILE_MB))
    cipsi_scratch_root = Path(str(study_cfg.get("cipsi_scratch_root", DEFAULT_CIPSI_SCRATCH_ROOT)))
    cipsi_rungs = tuple(int(x) for x in study_cfg.get("cipsi_rungs", list(DEFAULT_CIPSI_RUNGS)))
    cipsi_memory_gb = int(study_cfg.get("cipsi_memory_gb", dmrg_memory_gb))
    cipsi_conv_tol = float(study_cfg.get("cipsi_conv_tol", 1.0e-10))
    cipsi_cutoff_ladder = tuple(float(x) for x in study_cfg.get("cipsi_cutoff_ladder", list(DEFAULT_PYSCF_CIPSI_CUTOFFS)))

    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any]
    if effective_mode == "exact_fci":
        chosen = _load_or_compute_exact_reference_from_hamiltonian(
            h1=h1,
            eri8=eri8,
            norb=norb,
            nelec=tuple(nelec),
            ecore=ecore,
            outdir=setup_dir,
            force=force,
        )
    elif effective_mode == "cipsi_only":
        chosen = load_or_compute_pyscf_selected_ci_reference_from_fcidump(
            fcidump_path=fcidump_path,
            outdir=setup_dir,
            force=force,
            memory_gb=cipsi_memory_gb,
            conv_tol=cipsi_conv_tol,
            cutoff_ladder=cipsi_cutoff_ladder,
        )
        attempts.append(chosen)
        if not bool(chosen.get("success")):
            raise RuntimeError(f"Bounded CIPSI reference failed: {chosen.get('failure', 'unknown failure')}")
    elif effective_mode == "dmrg_only":
        dmrg_summary = load_or_compute_dmrg_reference_from_fcidump(
            fcidump_path=fcidump_path,
            outdir=setup_dir,
            force=force,
            memory_gb=dmrg_memory_gb,
            threads=dmrg_threads,
            conv_tol=dmrg_conv_tol,
            dmrg_python=dmrg_python,
            max_file_mb=dmrg_max_file_mb,
            max_total_scratch_gb=dmrg_total_scratch_gb,
            scratch_root=dmrg_scratch_root,
            bond_dims=None if dmrg_bond_dims is None else tuple(int(x) for x in dmrg_bond_dims),
            noises=None if dmrg_noises is None else tuple(float(x) for x in dmrg_noises),
            dav_thrds=None if dmrg_dav_thrds is None else tuple(float(x) for x in dmrg_dav_thrds),
            n_steps=dmrg_n_steps,
        )
        attempts.append(dmrg_summary)
        if not bool(dmrg_summary.get("success")):
            failure_payload = {
                "requested_reference_mode": requested_mode,
                "effective_reference_mode": effective_mode,
                "chosen_reference_kind": None,
                "reference_total_energy_eh": None,
                "attempts": attempts,
                "failure": "DMRG reference failed in dmrg_only mode",
            }
            choice_path.write_text(json.dumps(failure_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise RuntimeError(
                "DMRG reference failed in dmrg_only mode; see "
                f"{setup_dir / 'reference_dmrg_attempt.json'}"
            )
        chosen = dmrg_summary
    elif effective_mode == "dmrg_then_cipsi":
        dmrg_summary = load_or_compute_dmrg_reference_from_fcidump(
            fcidump_path=fcidump_path,
            outdir=setup_dir,
            force=force,
            memory_gb=dmrg_memory_gb,
            threads=dmrg_threads,
            conv_tol=dmrg_conv_tol,
            dmrg_python=dmrg_python,
            max_file_mb=dmrg_max_file_mb,
            max_total_scratch_gb=dmrg_total_scratch_gb,
            scratch_root=dmrg_scratch_root,
            bond_dims=None if dmrg_bond_dims is None else tuple(int(x) for x in dmrg_bond_dims),
            noises=None if dmrg_noises is None else tuple(float(x) for x in dmrg_noises),
            dav_thrds=None if dmrg_dav_thrds is None else tuple(float(x) for x in dmrg_dav_thrds),
            n_steps=dmrg_n_steps,
        )
        attempts.append(dmrg_summary)
        if bool(dmrg_summary.get("success")):
            chosen = dmrg_summary
        else:
            cipsi_summary = load_or_compute_pyscf_selected_ci_reference_from_fcidump(
                fcidump_path=fcidump_path,
                outdir=setup_dir,
                force=force,
                memory_gb=cipsi_memory_gb,
                conv_tol=cipsi_conv_tol,
                cutoff_ladder=cipsi_cutoff_ladder,
            )
            attempts.append(cipsi_summary)
            if not bool(cipsi_summary.get("success")):
                failure_payload = {
                    "requested_reference_mode": requested_mode,
                    "effective_reference_mode": effective_mode,
                    "chosen_reference_kind": None,
                    "reference_total_energy_eh": None,
                    "attempts": attempts,
                    "failure": "Both DMRG and bounded CIPSI references failed",
                }
                choice_path.write_text(json.dumps(failure_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                raise RuntimeError(
                    "DMRG reference failed and bounded CIPSI fallback failed; see "
                    f"{setup_dir / 'reference_dmrg_attempt.json'} and {setup_dir / 'reference_cipsi_attempt.json'}"
                )
            chosen = cipsi_summary
    else:
        raise ValueError(f"Unsupported reference_mode: {effective_mode!r}")

    choice_payload = {
        "requested_reference_mode": requested_mode,
        "effective_reference_mode": effective_mode,
        "chosen_reference_kind": chosen.get("reference_kind"),
        "chosen_reference_label": chosen.get("reference_label"),
        "reference_total_energy_eh": float(chosen["total_energy_eh"]),
        "dmrg_python": str(dmrg_python),
        "dmrg_memory_gb": dmrg_memory_gb,
        "dmrg_threads": dmrg_threads,
        "dmrg_conv_tol": dmrg_conv_tol,
        "dmrg_total_scratch_gb": dmrg_total_scratch_gb,
        "dmrg_max_file_mb": dmrg_max_file_mb,
        "dmrg_scratch_root": str(dmrg_scratch_root),
        "cipsi_total_scratch_gb": cipsi_total_scratch_gb,
        "cipsi_max_file_mb": cipsi_max_file_mb,
        "cipsi_scratch_root": str(cipsi_scratch_root),
        "cipsi_rungs": list(cipsi_rungs),
        "cipsi_memory_gb": cipsi_memory_gb,
        "cipsi_conv_tol": cipsi_conv_tol,
        "cipsi_cutoff_ladder": list(cipsi_cutoff_ladder),
        "attempts": attempts,
    }
    choice_path.write_text(json.dumps(choice_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return chosen, choice_payload


def _select_qp_root(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_root = os.environ.get("QP_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(QP_ROOT_CANDIDATES)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if validate_rebuildable_qp_root(resolved)["is_rebuildable"]:
            return resolved
    raise RuntimeError("No rebuildable Quantum Package root found for QP-backed study runs")


def _ensure_qp_study_binaries(qp_root: Path) -> None:
    qp = discover_qp_paths_from_root(qp_root)
    fcidump_binary = qp.qp_root / "src" / "tools" / "fcidump"
    if qp.scf_binary.exists() and fcidump_binary.exists():
        return
    cmd = (
        "eval \"$(opam env --switch=default --set-switch)\" && "
        f"source {shlex.quote(str(qp.quantum_package_rc))} >/dev/null 2>&1 || true && "
        f"{shlex.quote(str((qp.qp_root / 'scripts' / 'compilation' / 'qp_create_ninja').resolve()))} update >/dev/null 2>&1 || true && "
        f"ninja -f {shlex.quote(str((qp.qp_root / 'config' / 'build.ninja').resolve()))} "
        f"{shlex.quote(str(qp.scf_binary.resolve()))} "
        f"{shlex.quote(str(fcidump_binary.resolve()))}"
    )
    subprocess.run(
        ["/bin/bash", "-lc", cmd],
        check=True,
        text=True,
        cwd=str(qp.qp_root),
        env={**_with_qp_build_env(), "QP_ROOT": str(qp.qp_root)},
        capture_output=True,
    )


def _rewrite_n2_geometry_direct(qp_root: Path, ezfio: Path, *, distance_ang: float) -> None:
    z_bohr = float(distance_ang) / 2.0 / 0.529177210903
    ezfio_resolved = ezfio.resolve()
    script = f"""
from ezfio import ezfio
ezfio.set_file(r"{ezfio_resolved}")
ezfio.set_nuclei_nucl_num(2)
ezfio.set_nuclei_nucl_label(["N", "N"])
ezfio.set_nuclei_nucl_charge([7.0, 7.0])
ezfio.set_nuclei_nucl_coord([[0.0, 0.0], [0.0, 0.0], [{-z_bohr:.16f}, {z_bohr:.16f}]])
"""
    subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"eval \"$(opam env --switch=default --set-switch)\"; "
            f"source {shlex.quote(str((qp_root / 'quantum_package.rc').resolve()))} >/dev/null 2>&1 || true; "
            f"python3 - <<'PY'\n{script}\nPY",
        ],
        check=True,
        text=True,
        cwd=str(ezfio.parent),
        env={**os.environ, "QP_ROOT": str(qp_root)},
        capture_output=True,
    )


def _prepare_n2_setup_from_donor(
    *,
    qp_root: Path,
    ezfio_path: Path,
    distance_ang: float,
    ncore: int,
    ncas: int,
    nmo_total: int,
) -> None:
    donor = next((path for path in QP_N2_SETUP_DONOR_CANDIDATES if path.exists()), None)
    if donor is None:
        candidates = "\n".join(str(path) for path in QP_N2_SETUP_DONOR_CANDIDATES)
        raise FileNotFoundError(f"Missing N2 donor EZFIO for QP study setup. Checked:\n{candidates}")
    if ezfio_path.exists():
        shutil.rmtree(ezfio_path)
    shutil.copytree(donor, ezfio_path)
    qp = discover_qp_paths_from_root(qp_root)
    ensure_ezfio_defaults(ezfio_path, qp_root=qp_root)
    _rewrite_n2_geometry_direct(qp_root, ezfio_path, distance_ang=distance_ang)
    write_ezfio_value(ezfio_path, "nuclei/io_nuclear_repulsion", "None")
    write_ezfio_value(ezfio_path, "ao_two_e_ints/io_ao_two_e_integrals", "Write")
    write_ezfio_value(ezfio_path, "mo_two_e_ints/io_mo_two_e_integrals", "Write")
    write_ezfio_value(ezfio_path, "mo_one_e_ints/io_mo_one_e_integrals", "None")
    from qp_ccik_common import _set_mo_class_direct as _qp_set_mo_class_direct  # noqa: PLC0415

    _qp_set_mo_class_direct(qp, ezfio_path, mo_num=nmo_total, ncore=ncore, ncas=ncas)


def _prepare_case_qp(spec: dict[str, Any], *, case_dir: Path, force: bool = False) -> PreparedQpCase:
    molecule = spec["molecule"]
    cas_cfg = spec["cas"]
    study_cfg = spec.get("study", {})
    omp_threads = int(study_cfg.get("omp_threads", 8))
    qp_root = _select_qp_root(str(study_cfg.get("qp_root")) if study_cfg.get("qp_root") else None)
    _ensure_qp_study_binaries(qp_root)
    qp = discover_qp_paths_from_root(qp_root)

    setup_dir = case_dir / "system_setup"
    xyz_path = setup_dir / "n2.xyz"
    ezfio_path = setup_dir / "n2.ezfio"
    fcidump_path = setup_dir / "qp_active.FCIDUMP"
    summary_path = setup_dir / "system_setup.json"

    if summary_path.exists() and not force:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if Path(payload["ezfio_path"]).exists() and Path(payload["fcidump_path"]).exists():
            return PreparedQpCase(
                qp_root=Path(payload["qp_root"]),
                setup_dir=setup_dir,
                ezfio_path=Path(payload["ezfio_path"]),
                fcidump_path=Path(payload["fcidump_path"]),
                ecore=float(payload["ecore"]),
                norb=int(payload["norb"]),
                nelec=(int(payload["nelec"][0]), int(payload["nelec"][1])),
                reference_total_eh=float(payload["reference_total_energy_eh"]),
                reference_summary=dict(payload.get("reference_summary", {})),
                reference_mode=str(payload.get("effective_reference_mode", payload.get("reference_mode", "exact_fci"))),
                # Thread count is a live run setting, not part of the cached
                # system setup. Reusing the setup should not silently pin later
                # runs to an old OMP value.
                omp_threads=int(omp_threads),
            )

    if setup_dir.exists():
        shutil.rmtree(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)

    mol = make_mol_pyscf(
        atom=make_n2_atom(float(molecule["distance_ang"])),
        basis=str(molecule["basis"]),
        unit=str(molecule.get("unit", "Angstrom")),
        charge=int(molecule.get("charge", 0)),
        spin=int(molecule.get("spin", 0)),
        verbose=int(molecule.get("verbose", 0)),
    )
    nmo_total = int(mol.nao_nr())
    _create_xyz(xyz_path, make_n2_atom(float(molecule["distance_ang"])), comment=str(spec["label"]))
    setup_backend = "fresh_qp_setup"
    setup_fallback_reason: str | None = None
    try:
        run_qp_setup(
            qp,
            basis=str(molecule["basis"]),
            ezfio=ezfio_path,
            xyz=xyz_path,
            ncore=int(cas_cfg.get("ncore", 0)),
            ncas=int(cas_cfg["ncas"]),
            nmo_total=nmo_total,
        )
    except Exception as exc:
        if str(molecule["basis"]).strip().lower() != "6-31g":
            raise
        _prepare_n2_setup_from_donor(
            qp_root=qp_root,
            ezfio_path=ezfio_path,
            distance_ang=float(molecule["distance_ang"]),
            ncore=int(cas_cfg.get("ncore", 0)),
            ncas=int(cas_cfg["ncas"]),
            nmo_total=nmo_total,
        )
        setup_backend = "donor_ezfio_fallback"
        setup_fallback_reason = f"{type(exc).__name__}: {exc}"
    generated_fcidump = dsp._generate_qp_fcidump(qp, ezfio_path, setup_dir, case_spec=None)
    h1, eri8, norb, nelec, ecore = dsp._load_qp_fcidump_hamiltonian(generated_fcidump)
    reference_summary, reference_choice = _resolve_qp_reference(
        spec=spec,
        qp_root=qp_root,
        ezfio_path=ezfio_path,
        fcidump_path=generated_fcidump,
        setup_dir=setup_dir,
        h1=h1,
        eri8=eri8,
        norb=int(norb),
        nelec=tuple(nelec),
        ecore=float(ecore),
        omp_threads=int(omp_threads),
        force=force,
    )
    payload = {
        "qp_root": str(qp_root),
        "omp_threads": int(omp_threads),
        "label": str(spec["label"]),
        "ezfio_path": str(ezfio_path),
        "fcidump_path": str(generated_fcidump),
        "ecore": float(ecore),
        "norb": int(norb),
        "nelec": [int(nelec[0]), int(nelec[1])],
        "reference_mode": _requested_reference_mode(spec),
        "effective_reference_mode": str(reference_choice["effective_reference_mode"]),
        "reference_total_energy_eh": float(reference_summary["total_energy_eh"]),
        "reference_summary": reference_summary,
        "reference_choice_path": str(setup_dir / "reference_choice.json"),
        "basis": str(molecule["basis"]),
        "setup_backend": setup_backend,
        "setup_fallback_reason": setup_fallback_reason,
        "distance_ang": float(molecule["distance_ang"]),
        "ncas": int(cas_cfg["ncas"]),
        "nelecas": int(cas_cfg["nelecas"]),
        "ncore": int(cas_cfg.get("ncore", 0)),
        "nmo_total": int(nmo_total),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PreparedQpCase(
        qp_root=qp_root,
        setup_dir=setup_dir,
        ezfio_path=ezfio_path,
        fcidump_path=generated_fcidump,
        ecore=float(ecore),
        norb=int(norb),
        nelec=(int(nelec[0]), int(nelec[1])),
        reference_total_eh=float(reference_summary["total_energy_eh"]),
        reference_summary=reference_summary,
        reference_mode=str(reference_choice["effective_reference_mode"]),
        omp_threads=int(omp_threads),
    )


def _make_qp_thick_params(
    *,
    run_params: dict[str, Any],
    reference_total_eh: float,
) -> QPCCIKParams:
    def _get_optional_int(name: str) -> int | None:
        value = run_params.get(name)
        return None if value is None else int(value)

    def _get_optional_float(name: str) -> float | None:
        value = run_params.get(name)
        return None if value is None else float(value)

    def _get_optional_bool(name: str) -> bool | None:
        value = run_params.get(name)
        return None if value is None else bool(value)

    def _get_optional_str(name: str) -> str | None:
        value = run_params.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    fixed_store_keep = int(run_params["fixed_store_keep"])
    fixed_propagate_keep = int(run_params["fixed_propagate_keep"])
    m_cycle = int(run_params["m_cycle"])
    ncycles = int(run_params["ncycles"])
    nroot = int(run_params["nroot"])
    selector_mode = _get_optional_str("selector_mode") or "abs_residual"
    matvec_backend = _get_optional_str("matvec_backend") or "connected_sd_fullq"
    compression_policy = _get_optional_str("compression_policy") or "fixed_budget"
    conv_tol = _get_optional_float("conv_tol")
    ritz_overlap_tol = _get_optional_float("ritz_overlap_tol")
    ritz_max_drop = _get_optional_float("ritz_max_drop")
    save_final_wf = _get_optional_bool("save_final_wf")
    use_existing_wf = _get_optional_bool("use_existing_wf")
    restart_from_run = _get_optional_str("restart_from_run")
    do_post_pt2 = _get_optional_bool("do_post_pt2")
    verbose = _get_optional_bool("verbose")
    debug_trace = _get_optional_bool("debug_trace")
    debug_allow_legacy_fixed_budget = _get_optional_bool("debug_allow_legacy_fixed_budget")
    debug_trace_topn = _get_optional_int("debug_trace_topn")
    if restart_from_run is not None and use_existing_wf is None:
        use_existing_wf = True
    bridge = fixed_budget_bridge_kwargs(
        fixed_store_keep=fixed_store_keep,
        fixed_propagate_keep=fixed_propagate_keep,
        external_keep=max(2048, fixed_store_keep),
    )
    if compression_policy != str(bridge.get("ccik_compression_policy", "fixed_budget")):
        bridge["ccik_compression_policy"] = compression_policy
    int_overrides = {
        "ccik_generator_keep": _get_optional_int("generator_keep"),
        "ccik_max_parents_for_expansion": _get_optional_int("max_parents_for_expansion"),
        "ccik_selector_keep": _get_optional_int("selector_keep"),
        "ccik_external_keep": _get_optional_int("external_keep"),
        "ccik_external_candidate_factor": _get_optional_int("external_candidate_factor"),
        "ccik_fixed_store_keep": _get_optional_int("fixed_store_keep_override"),
        "ccik_fixed_propagate_keep": _get_optional_int("fixed_propagate_keep_override"),
        "ccik_nkeep": _get_optional_int("nkeep"),
        "ccik_tmp_keep_factor": _get_optional_int("tmp_keep_factor"),
        "ccik_preserve_prev_support_cap": _get_optional_int("preserve_prev_support_cap"),
        "ccik_total_det_budget": _get_optional_int("total_det_budget"),
        "ccik_exactized_checkpoint_every": _get_optional_int("exactized_checkpoint_every"),
        "ccik_exactized_max_iter": _get_optional_int("exactized_max_iter"),
        "ccik_exactized_max_subspace": _get_optional_int("exactized_max_subspace"),
        "ccik_exactized_restart_keep": _get_optional_int("exactized_restart_keep"),
        "ccik_debug_trace_topn": _get_optional_int("debug_trace_topn"),
    }
    float_overrides = {
        "ccik_generator_fraction": _get_optional_float("generator_fraction"),
        "ccik_generator_abs_coeff_cutoff": _get_optional_float("generator_abs_coeff_cutoff"),
        "ccik_pt2_rank_weight": _get_optional_float("pt2_rank_weight"),
        "ccik_store_abs_tol": _get_optional_float("store_abs_tol"),
        "ccik_propagate_abs_tol": _get_optional_float("propagate_abs_tol"),
        "ccik_conv_tol": _get_optional_float("conv_tol"),
        "ccik_ritz_overlap_tol": _get_optional_float("ritz_overlap_tol"),
        "ccik_ritz_max_drop": _get_optional_float("ritz_max_drop"),
        "ccik_exactized_resid_tol": _get_optional_float("exactized_resid_tol"),
    }
    bool_overrides = {
        "ccik_include_current_support": _get_optional_bool("include_current_support"),
        "ccik_preserve_prev_support": _get_optional_bool("preserve_prev_support"),
        "ccik_save_final_wf": _get_optional_bool("save_final_wf"),
        "ccik_use_existing_wf": _get_optional_bool("use_existing_wf"),
        "ccik_do_post_pt2": _get_optional_bool("do_post_pt2"),
        "ccik_secondortho": _get_optional_bool("secondortho"),
        "ccik_verbose": _get_optional_bool("verbose"),
        "ccik_debug_trace": _get_optional_bool("debug_trace"),
        "ccik_debug_allow_legacy_fixed_budget": _get_optional_bool("debug_allow_legacy_fixed_budget"),
    }
    for key, value in int_overrides.items():
        if value is not None:
            bridge[key] = value
    for key, value in float_overrides.items():
        if value is not None:
            bridge[key] = value
    for key, value in bool_overrides.items():
        if value is not None:
            bridge[key] = value
    # In the QP plugin, `ccik_basis_size` and `ccik_restart_m_cycle` both set
    # the within-cycle Krylov basis size. For example, `m_cycle = 51` means the
    # first thick restart happens after 50 newly generated vectors because cycle
    # 1 starts from a single seed vector.
    # Avoid passing any override twice when it is already present in `bridge`.
    params_defaults: dict[str, Any] = {
        "ccik_basis_size": m_cycle,
        "ccik_selector_mode": selector_mode,
        "ccik_matvec_backend": matvec_backend,
        "ccik_conv_tol": 1.0e-10 if conv_tol is None else conv_tol,
        "ccik_ritz_overlap_tol": 1.0e-8 if ritz_overlap_tol is None else ritz_overlap_tol,
        "ccik_ritz_max_drop": 0.0 if ritz_max_drop is None else ritz_max_drop,
        "ccik_max_cycle": ncycles,
        "ccik_restart_m_cycle": m_cycle,
        "ccik_restart_nroot": nroot,
        "ccik_reference_total_energy": float(reference_total_eh),
        "ccik_print_iteration_error": True,
        "ccik_verbose": False if verbose is None else verbose,
    }
    params_kwargs: dict[str, Any] = {key: value for key, value in params_defaults.items() if key not in bridge}
    if "ccik_debug_trace" not in bridge:
        params_kwargs["ccik_debug_trace"] = False if debug_trace is None else debug_trace
    if "ccik_debug_trace_topn" not in bridge:
        params_kwargs["ccik_debug_trace_topn"] = 8 if debug_trace_topn is None else debug_trace_topn
    if "ccik_debug_allow_legacy_fixed_budget" not in bridge:
        params_kwargs["ccik_debug_allow_legacy_fixed_budget"] = (
            False if debug_allow_legacy_fixed_budget is None else debug_allow_legacy_fixed_budget
        )
    if "ccik_exactized_checkpoint_every" not in bridge:
        params_kwargs["ccik_exactized_checkpoint_every"] = 0
    if "ccik_exactized_resid_tol" not in bridge:
        params_kwargs["ccik_exactized_resid_tol"] = 1.0e-10
    if "ccik_exactized_max_iter" not in bridge:
        params_kwargs["ccik_exactized_max_iter"] = 200
    if "ccik_exactized_max_subspace" not in bridge:
        params_kwargs["ccik_exactized_max_subspace"] = 24
    if "ccik_exactized_restart_keep" not in bridge:
        params_kwargs["ccik_exactized_restart_keep"] = 4
    if "ccik_save_final_wf" not in bridge:
        params_kwargs["ccik_save_final_wf"] = False if save_final_wf is None else save_final_wf
    if "ccik_use_existing_wf" not in bridge:
        params_kwargs["ccik_use_existing_wf"] = False if use_existing_wf is None else use_existing_wf
    if "ccik_do_post_pt2" not in bridge:
        params_kwargs["ccik_do_post_pt2"] = False if do_post_pt2 is None else do_post_pt2
    return QPCCIKParams(**params_kwargs, **bridge)


def _cumulative_new_krylov(
    cycle: int,
    iter_idx: int,
    *,
    restart_m_cycle: int,
    restart_nroot: int,
    first_cycle_seed_count: int = 1,
) -> int:
    # Thick restart reuses Ritz vectors at the start of each cycle. The study
    # plots count only *newly generated* Krylov vectors, so later cycles start
    # from `restart_nroot` stored vectors instead of zero.
    if cycle == 1:
        return int(iter_idx) - int(first_cycle_seed_count)
    generated_in_cycle1 = max(0, int(restart_m_cycle) - int(first_cycle_seed_count))
    generated_in_later_cycle = max(0, int(restart_m_cycle) - int(restart_nroot))
    return generated_in_cycle1 + (int(cycle) - 2) * generated_in_later_cycle + (int(iter_idx) - int(restart_nroot))


def _extract_final_value(stdout_text: str, key: str) -> float | None:
    for raw in stdout_text.splitlines():
        match = FINAL_VALUE_RE.search(raw)
        if match is None:
            continue
        if match.group("key") == key:
            return float(match.group("value"))
    return None


def _build_qp_trace_row(
    *,
    cycle: int,
    iter_idx: int,
    total_energy_eh: float,
    delta_mha: float,
    ecore: float,
    restart_m_cycle: int,
    restart_nroot: int,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "cycle": int(cycle),
        "basis_size": int(iter_idx),
        "restart_seed_count": 1 if int(cycle) == 1 else int(restart_nroot),
        "cum_new_krylov": int(
            _cumulative_new_krylov(
                int(cycle),
                int(iter_idx),
                restart_m_cycle=int(restart_m_cycle),
                restart_nroot=int(restart_nroot),
                first_cycle_seed_count=1,
            )
        ),
        "support_union_size": None,
        "electronic_energy_eh": float(total_energy_eh - float(ecore)),
        "total_energy_eh": float(total_energy_eh),
        "delta_mHa": float(delta_mha),
        "abs_delta_mHa": abs(float(delta_mha)),
        "exactized_electronic_energy_eh": None,
        "exactized_total_energy_eh": None,
        "exactized_delta_mHa": None,
        "exactized_abs_delta_mHa": None,
        "exactized_solver_kind": None,
        "exactized_subspace_dim": None,
        "exactized_hamiltonian_nnz": None,
        "exactizeplus_electronic_energy_eh": None,
        "exactizeplus_total_energy_eh": None,
        "exactizeplus_delta_mHa": None,
        "exactizeplus_abs_delta_mHa": None,
        "exactizeplus_solver_kind": None,
        "exactizeplus_subspace_dim": None,
        "exactizeplus_parent_count": None,
        "exactizeplus_shell_added": None,
        "exactized_residual_norm": None,
        "exactized_davidson_iterations": None,
        "exactized_matvec_count": None,
        "exactized_converged": None,
        "exactized_support_build_s": None,
        "exactized_prep_s": None,
        "exactized_solve_s": None,
        "exactized_wall_s": None,
        "exactized_cpu_s": None,
        "elapsed_s": float(elapsed_s),
    }


def _update_qp_row_with_exactized_trace(row: dict[str, Any], match: re.Match[str]) -> None:
    exactized_total = float(match.group("total"))
    exactized_delta = float(match.group("delta"))
    row["support_union_size"] = int(match.group("support"))
    row["exactized_electronic_energy_eh"] = float(match.group("electronic"))
    row["exactized_total_energy_eh"] = exactized_total
    row["exactized_delta_mHa"] = exactized_delta
    row["exactized_abs_delta_mHa"] = abs(exactized_delta)
    row["exactized_solver_kind"] = str(match.group("solver"))
    row["exactized_subspace_dim"] = int(match.group("support"))
    row["exactized_hamiltonian_nnz"] = None
    row["exactized_residual_norm"] = float(match.group("resid"))
    row["exactized_davidson_iterations"] = int(match.group("iters"))
    row["exactized_matvec_count"] = int(match.group("matvecs"))
    row["exactized_converged"] = bool(match.group("converged") == "T")
    row["exactized_support_build_s"] = float(match.group("t_union"))
    row["exactized_prep_s"] = float(match.group("t_prep"))
    row["exactized_solve_s"] = float(match.group("t_solve"))
    row["exactized_wall_s"] = float(match.group("t_wall"))
    row["exactized_cpu_s"] = float(match.group("t_cpu"))


def _run_qp_thick_trace(
    *,
    case_dir: Path,
    case_label: str,
    run_name: str,
    run_label: str,
    prepared: PreparedQpCase,
    run_params: dict[str, Any],
    max_cumulative_iterations: int,
    plot_max_cumulative_iterations: int,
    stop_abs_delta_mha: float | None,
    run_labels: dict[str, str],
) -> dict[str, Any]:
    qp = discover_qp_paths_from_root(prepared.qp_root)
    run_dir = _run_dir(case_dir, run_name)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    restart_from_run = None
    restart_source_ezfio = prepared.ezfio_path
    raw_restart_from_run = run_params.get("restart_from_run")
    if raw_restart_from_run is not None:
        restart_from_run = str(raw_restart_from_run).strip()
        if not restart_from_run:
            restart_from_run = None
    if restart_from_run is not None:
        if restart_from_run == run_name:
            raise ValueError(f"Run {run_name!r} cannot restart from itself")
        restart_source_ezfio = _runtime_ezfio_path(_run_dir(case_dir, restart_from_run), prepared.ezfio_path.name)
        if not restart_source_ezfio.exists():
            raise FileNotFoundError(
                f"Restart source EZFIO not found for run {run_name!r}: "
                f"{restart_source_ezfio} (restart_from_run={restart_from_run!r})"
            )
    runtime_ezfio = runtime_dir / restart_source_ezfio.name
    shutil.copytree(restart_source_ezfio, runtime_ezfio)
    ensure_ezfio_defaults(runtime_ezfio, qp_root=prepared.qp_root)
    ensure_ezfio_workdir(runtime_ezfio)
    if not (runtime_ezfio / "work").is_dir():
        raise RuntimeError(f"Runtime EZFIO is missing the required work directory: {runtime_ezfio / 'work'}")

    params = _make_qp_thick_params(run_params=run_params, reference_total_eh=prepared.reference_total_eh)
    set_ccik_params(qp, runtime_ezfio, params)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_reference_total_energy", float(prepared.reference_total_eh))
    write_ezfio_value(runtime_ezfio, "ccik/ccik_print_iteration_error", True)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace", bool(params.ccik_debug_trace))
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace_topn", int(params.ccik_debug_trace_topn))
    write_ezfio_value(
        runtime_ezfio,
        "ccik/ccik_debug_allow_legacy_fixed_budget",
        bool(params.ccik_debug_allow_legacy_fixed_budget),
    )

    stdout_path = run_dir / "ccik_stdout.log"
    stderr_path = run_dir / "ccik_stderr.log"
    env = {**_with_qp_build_env(), "QP_ROOT": str(prepared.qp_root), **qp_parallel_env(prepared.omp_threads)}
    quoted_rc = shlex.quote(str(qp.quantum_package_rc.resolve()))
    quoted_ccik = shlex.quote(str(qp.ccik_binary.resolve()))
    short_ezfio = shlex.quote(runtime_ezfio.name)

    rows: list[dict[str, Any]] = []
    early_stop_reached = False
    early_stop_target = None if stop_abs_delta_mha is None else float(stop_abs_delta_mha)
    t0 = time.perf_counter()
    stdout_chunks: list[str] = []
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        proc = subprocess.Popen(
            [
                "/bin/bash",
                "-lc",
                f"source {quoted_rc} >/dev/null 2>&1 && qp set_file {short_ezfio} >/dev/null && {quoted_ccik} {short_ezfio}",
            ],
            cwd=str(runtime_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            stdout_fh.write(raw)
            stdout_fh.flush()
            stdout_chunks.append(raw)
            exactized_match = QP_EXACTIZED_TRACE_RE.search(raw)
            if exactized_match is not None:
                if rows:
                    latest = rows[-1]
                    if int(latest["cycle"]) == int(exactized_match.group("cycle")) and int(latest["basis_size"]) == int(
                        exactized_match.group("iter")
                    ):
                        _update_qp_row_with_exactized_trace(latest, exactized_match)
                        _refresh_run_outputs(
                            case_dir=case_dir,
                            case_label=case_label,
                            run_name=run_name,
                            run_label=run_label,
                            rows=rows,
                            max_cumulative_iterations=max_cumulative_iterations,
                            plot_max_cumulative_iterations=plot_max_cumulative_iterations,
                            fixed_store_keep=int(run_params["fixed_store_keep"]),
                            fixed_propagate_keep=int(run_params["fixed_propagate_keep"]),
                            run_labels=run_labels,
                        )
                continue
            match = QP_THICK_LINE_RE.search(raw)
            if match is None:
                baseline_match = QP_BASELINE_LINE_RE.search(raw)
                if baseline_match is None:
                    continue
                cycle = 1
                iter_idx = int(baseline_match.group("iter"))
                total_energy_eh = float(baseline_match.group("total"))
                delta_mha = float(baseline_match.group("delta"))
            else:
                cycle = int(match.group("cycle"))
                iter_idx = int(match.group("iter"))
                total_energy_eh = float(match.group("total"))
                delta_mha = float(match.group("delta"))
            row = _build_qp_trace_row(
                cycle=cycle,
                iter_idx=iter_idx,
                total_energy_eh=total_energy_eh,
                delta_mha=delta_mha,
                ecore=prepared.ecore,
                restart_m_cycle=int(run_params["m_cycle"]),
                restart_nroot=int(run_params["nroot"]),
                elapsed_s=float(time.perf_counter() - t0),
            )
            rows.append(row)
            _refresh_run_outputs(
                case_dir=case_dir,
                case_label=case_label,
                run_name=run_name,
                run_label=run_label,
                rows=rows,
                max_cumulative_iterations=max_cumulative_iterations,
                plot_max_cumulative_iterations=plot_max_cumulative_iterations,
                fixed_store_keep=int(run_params["fixed_store_keep"]),
                fixed_propagate_keep=int(run_params["fixed_propagate_keep"]),
                run_labels=run_labels,
            )
            if early_stop_target is not None and float(row["abs_delta_mHa"]) <= early_stop_target and not early_stop_reached:
                early_stop_reached = True
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
        returncode = int(proc.wait())
    wall_s = float(time.perf_counter() - t0)
    stdout_text = "".join(stdout_chunks) if stdout_chunks else stdout_path.read_text(encoding="utf-8", errors="ignore")
    final_total = _extract_final_value(stdout_text, "CCIK_FINAL_TOTAL_ENERGY")
    limited_rows = limit_iteration_rows(rows, max_cumulative_iterations)
    if not limited_rows:
        raise RuntimeError(f"No thick-restart iteration rows were recorded for {run_name}")
    below_fci_points = [row for row in limited_rows if float(row["delta_mHa"]) < -1.0e-9]
    below_reference_report = [
        {
            "cycle": int(row["cycle"]),
            "cum_new_krylov": int(row["cum_new_krylov"]),
            "total_energy_eh": float(row["total_energy_eh"]),
            "delta_mHa": float(row["delta_mHa"]),
            "abs_delta_mHa": float(row["abs_delta_mHa"]),
        }
        for row in below_fci_points
    ]
    chemical_accuracy_mha = 1.0
    standard_hit = next((row for row in limited_rows if float(row["abs_delta_mHa"]) <= chemical_accuracy_mha), None)
    exactized_rows = [row for row in limited_rows if row.get("exactized_abs_delta_mHa") is not None]
    exactized_hit = next(
        (row for row in exactized_rows if float(row["exactized_abs_delta_mHa"]) <= chemical_accuracy_mha),
        None,
    )
    exactized_wall_total = float(
        sum(float(row["exactized_wall_s"]) for row in exactized_rows if row.get("exactized_wall_s") is not None)
    )
    exactized_support_build_total = float(
        sum(float(row["exactized_support_build_s"]) for row in exactized_rows if row.get("exactized_support_build_s") is not None)
    )
    exactized_prep_total = float(
        sum(float(row["exactized_prep_s"]) for row in exactized_rows if row.get("exactized_prep_s") is not None)
    )
    exactized_solve_total = float(
        sum(float(row["exactized_solve_s"]) for row in exactized_rows if row.get("exactized_solve_s") is not None)
    )
    exactized_cpu_total = float(
        sum(float(row["exactized_cpu_s"]) for row in exactized_rows if row.get("exactized_cpu_s") is not None)
    )
    exactized_matvec_total = int(
        sum(int(row["exactized_matvec_count"]) for row in exactized_rows if row.get("exactized_matvec_count") is not None)
    )
    effective_final_total = float(limited_rows[-1]["total_energy_eh"]) if final_total is None else float(final_total)
    threshold_hit = (
        next((row for row in limited_rows if early_stop_target is not None and float(row["abs_delta_mHa"]) <= early_stop_target), None)
        if early_stop_target is not None
        else None
    )
    summary = {
        "backend": "qp",
        "qp_root": str(prepared.qp_root),
        "omp_threads": int(prepared.omp_threads),
        "run_name": run_name,
        "run_label": run_label,
        "reference_mode": prepared.reference_mode,
        "reference_kind": prepared.reference_summary.get("reference_kind"),
        "reference_label": prepared.reference_summary.get("reference_label"),
        "reference_total_energy_eh": float(prepared.reference_total_eh),
        "final_total_energy_eh": effective_final_total,
        "final_delta_mHa": _signed_error_mha(effective_final_total, float(prepared.reference_total_eh)),
        "final_abs_delta_mHa": _abs_error_mha(effective_final_total, float(prepared.reference_total_eh)),
        "first_logged_abs_delta_mHa": float(limited_rows[0]["abs_delta_mHa"]),
        "plotted_final_total_energy_eh": float(limited_rows[-1]["total_energy_eh"]),
        "plotted_final_abs_delta_mHa": float(limited_rows[-1]["abs_delta_mHa"]),
        "plotted_iterations": int(limited_rows[-1]["cum_new_krylov"]),
        "trace_points": int(len(limited_rows)),
        "exactized_trace_points": int(len(exactized_rows)),
        "below_fci_detected": bool(below_fci_points),
        "below_reference_points": below_reference_report,
        "best_below_reference_total_energy_eh": (
            None if not below_fci_points else float(min(float(row["total_energy_eh"]) for row in below_fci_points))
        ),
        "min_signed_delta_mHa": float(min(row["delta_mHa"] for row in limited_rows)),
        "improves_over_first_point": float(limited_rows[-1]["abs_delta_mHa"]) < float(limited_rows[0]["abs_delta_mHa"]),
        "chemical_accuracy_target_mHa": chemical_accuracy_mha,
        "standard_reached_chemical_accuracy": bool(standard_hit),
        "standard_first_chemical_accuracy_iteration": None if standard_hit is None else int(standard_hit["cum_new_krylov"]),
        "exactized_reached_chemical_accuracy": bool(exactized_hit),
        "exactized_first_chemical_accuracy_iteration": None if exactized_hit is None else int(exactized_hit["cum_new_krylov"]),
        "exactized_first_chemical_accuracy_support_union_size": (
            None if exactized_hit is None or exactized_hit.get("support_union_size") is None else int(exactized_hit["support_union_size"])
        ),
        "best_exactized_total_energy_eh": (
            None
            if not exactized_rows
            else float(min(float(row["exactized_total_energy_eh"]) for row in exactized_rows if row["exactized_total_energy_eh"] is not None))
        ),
        "best_exactized_abs_delta_mHa": (
            None
            if not exactized_rows
            else float(min(float(row["exactized_abs_delta_mHa"]) for row in exactized_rows if row["exactized_abs_delta_mHa"] is not None))
        ),
        "early_stop_target_abs_delta_mHa": early_stop_target,
        "early_stop_reached": bool(early_stop_reached),
        "exactized_total_checkpoint_wall_s": exactized_wall_total,
        "exactized_total_support_build_s": exactized_support_build_total,
        "exactized_total_prep_s": exactized_prep_total,
        "exactized_total_solve_s": exactized_solve_total,
        "exactized_total_checkpoint_cpu_s": exactized_cpu_total,
        "exactized_total_matvec_count": exactized_matvec_total,
        "exactized_mean_checkpoint_wall_s": (None if not exactized_rows else exactized_wall_total / float(len(exactized_rows))),
        "exactized_mean_support_build_s": (
            None if not exactized_rows else exactized_support_build_total / float(len(exactized_rows))
        ),
        "exactized_mean_prep_s": (None if not exactized_rows else exactized_prep_total / float(len(exactized_rows))),
        "exactized_mean_solve_s": (None if not exactized_rows else exactized_solve_total / float(len(exactized_rows))),
        "threshold_hit_iteration": None if threshold_hit is None else int(threshold_hit["cum_new_krylov"]),
        "threshold_hit_elapsed_s": None if threshold_hit is None else float(threshold_hit["elapsed_s"]),
        "wall_s": wall_s,
        "returncode": returncode,
        "plot_png": str((run_dir / "energy_error_vs_iteration.png")),
        "plot_pdf": str((run_dir / "energy_error_vs_iteration.pdf")),
        "trace_csv": str((run_dir / "iteration_trace.csv")),
        "trace_json": str((run_dir / "iteration_trace.json")),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "runtime_ezfio": str(runtime_ezfio),
        "runtime_source_ezfio": str(restart_source_ezfio),
        "restart_from_run": restart_from_run,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _clean_case_outputs(case_dir: Path, run_names: list[str], *, remove_system_setup: bool) -> None:
    for run_name in run_names:
        run_dir = _run_dir(case_dir, run_name)
        if run_dir.exists():
            shutil.rmtree(run_dir)
    for rel in ("study_summary.json", "energy_error_vs_iteration_comparison.png", "energy_error_vs_iteration_comparison.pdf"):
        target = case_dir / rel
        if target.exists():
            target.unlink()
    if remove_system_setup:
        setup_dir = case_dir / "system_setup"
        if setup_dir.exists():
            shutil.rmtree(setup_dir)


def _build_trace_row(record: ThickRestartIterationRecord, *, ecore: float, e_fci_total: float) -> dict[str, Any]:
    total_energy = float(ecore + record.electronic_energy_eh)
    exactized_total = (
        None
        if record.exactized_electronic_energy_eh is None
        else float(ecore + float(record.exactized_electronic_energy_eh))
    )
    exactizeplus_total = (
        None
        if record.exactizeplus_electronic_energy_eh is None
        else float(ecore + float(record.exactizeplus_electronic_energy_eh))
    )
    return {
        "cycle": int(record.cycle),
        "basis_size": int(record.basis_size),
        "restart_seed_count": int(record.restart_seed_count),
        "cum_new_krylov": int(record.cum_new_krylov),
        "support_union_size": None if record.support_union_size is None else int(record.support_union_size),
        "electronic_energy_eh": float(record.electronic_energy_eh),
        "total_energy_eh": total_energy,
        "delta_mHa": _signed_error_mha(total_energy, e_fci_total),
        "abs_delta_mHa": _abs_error_mha(total_energy, e_fci_total),
        "exactized_electronic_energy_eh": (
            None if record.exactized_electronic_energy_eh is None else float(record.exactized_electronic_energy_eh)
        ),
        "exactized_total_energy_eh": exactized_total,
        "exactized_delta_mHa": None if exactized_total is None else _signed_error_mha(exactized_total, e_fci_total),
        "exactized_abs_delta_mHa": None if exactized_total is None else _abs_error_mha(exactized_total, e_fci_total),
        "exactized_solver_kind": None if record.exactized_solver_kind is None else str(record.exactized_solver_kind),
        "exactized_subspace_dim": None if record.exactized_subspace_dim is None else int(record.exactized_subspace_dim),
        "exactized_hamiltonian_nnz": (
            None if record.exactized_hamiltonian_nnz is None else int(record.exactized_hamiltonian_nnz)
        ),
        "exactizeplus_electronic_energy_eh": (
            None if record.exactizeplus_electronic_energy_eh is None else float(record.exactizeplus_electronic_energy_eh)
        ),
        "exactizeplus_total_energy_eh": exactizeplus_total,
        "exactizeplus_delta_mHa": (
            None if exactizeplus_total is None else _signed_error_mha(exactizeplus_total, e_fci_total)
        ),
        "exactizeplus_abs_delta_mHa": (
            None if exactizeplus_total is None else _abs_error_mha(exactizeplus_total, e_fci_total)
        ),
        "exactizeplus_solver_kind": (
            None if record.exactizeplus_solver_kind is None else str(record.exactizeplus_solver_kind)
        ),
        "exactizeplus_subspace_dim": (
            None if record.exactizeplus_subspace_dim is None else int(record.exactizeplus_subspace_dim)
        ),
        "exactizeplus_parent_count": (
            None if record.exactizeplus_parent_count is None else int(record.exactizeplus_parent_count)
        ),
        "exactizeplus_shell_added": (
            None if record.exactizeplus_shell_added is None else int(record.exactizeplus_shell_added)
        ),
        "elapsed_s": None,
    }


def _refresh_run_outputs(
    *,
    case_dir: Path,
    case_label: str,
    run_name: str,
    run_label: str,
    rows: list[dict[str, Any]],
    max_cumulative_iterations: int,
    plot_max_cumulative_iterations: int,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    run_labels: dict[str, str],
) -> None:
    limited_rows = limit_iteration_rows(rows, max_cumulative_iterations)
    if not limited_rows:
        return
    run_dir = _run_dir(case_dir, run_name)
    write_trace_outputs(run_dir, limited_rows)
    plot_run_trace(
        run_dir=run_dir,
        rows=limited_rows,
        title=_case_plot_title(case_label, run_label, fixed_store_keep, fixed_propagate_keep),
        max_cumulative_iterations=plot_max_cumulative_iterations,
    )
    latest = limited_rows[-1]
    write_live_status(run_dir, build_live_status_payload(label=run_label, latest=latest))
    plot_case_comparison(
        case_dir=case_dir,
        case_label=case_label,
        run_labels=run_labels,
        max_cumulative_iterations=plot_max_cumulative_iterations,
    )


def _run_thick_trace(
    *,
    case_dir: Path,
    case_label: str,
    run_name: str,
    run_label: str,
    h1: np.ndarray,
    eri8: np.ndarray,
    ecore: float,
    norb: int,
    nelec: tuple[int, int],
    e_fci_total: float,
    params: CCIKThickRestartParams,
    max_cumulative_iterations: int,
    plot_max_cumulative_iterations: int,
    stop_abs_delta_mha: float | None,
    run_labels: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    early_stop_reached = False
    t0 = time.perf_counter()
    chemical_accuracy_mha = 1.0

    def _collect(record: ThickRestartIterationRecord) -> None:
        rows.append(_build_trace_row(record, ecore=ecore, e_fci_total=e_fci_total))
        _refresh_run_outputs(
            case_dir=case_dir,
            case_label=case_label,
            run_name=run_name,
            run_label=run_label,
            rows=rows,
            max_cumulative_iterations=max_cumulative_iterations,
            plot_max_cumulative_iterations=plot_max_cumulative_iterations,
            fixed_store_keep=int(params.fixed_store_keep),
            fixed_propagate_keep=int(params.fixed_propagate_keep),
            run_labels=run_labels,
        )
        if stop_abs_delta_mha is not None and float(rows[-1]["abs_delta_mHa"]) <= float(stop_abs_delta_mha):
            raise _StudyEarlyStop(f"Reached target |ΔE| <= {float(stop_abs_delta_mha):.6g} mHa")

    try:
        e_total = float(
            ecore
            + ccik_ground_energy_dense_thick_restart(
                h1,
                eri8,
                norb,
                nelec,
                params=params,
                stats=stats,
                iteration_callback=_collect,
            )
        )
    except _StudyEarlyStop:
        early_stop_reached = True
        if not rows:
            raise
        e_total = float(rows[-1]["total_energy_eh"])
    wall_s = float(time.perf_counter() - t0)
    limited_rows = limit_iteration_rows(rows, max_cumulative_iterations)
    if not limited_rows:
        raise RuntimeError(f"No thick-restart iteration rows were recorded for {run_name}")

    below_fci_points = [row for row in limited_rows if float(row["delta_mHa"]) < -1.0e-9]
    below_reference_report = [
        {
            "cycle": int(row["cycle"]),
            "cum_new_krylov": int(row["cum_new_krylov"]),
            "total_energy_eh": float(row["total_energy_eh"]),
            "delta_mHa": float(row["delta_mHa"]),
            "abs_delta_mHa": float(row["abs_delta_mHa"]),
        }
        for row in below_fci_points
    ]
    standard_hit = next((row for row in limited_rows if float(row["abs_delta_mHa"]) <= chemical_accuracy_mha), None)
    exactized_rows = [row for row in limited_rows if row.get("exactized_abs_delta_mHa") is not None]
    exactized_hit = next(
        (row for row in exactized_rows if float(row["exactized_abs_delta_mHa"]) <= chemical_accuracy_mha),
        None,
    )
    exactizeplus_rows = [row for row in limited_rows if row.get("exactizeplus_abs_delta_mHa") is not None]
    exactizeplus_hit = next(
        (row for row in exactizeplus_rows if float(row["exactizeplus_abs_delta_mHa"]) <= chemical_accuracy_mha),
        None,
    )
    summary = {
        "run_name": run_name,
        "run_label": run_label,
        "reference_total_energy_eh": e_fci_total,
        "final_total_energy_eh": e_total,
        "final_delta_mHa": _signed_error_mha(e_total, e_fci_total),
        "final_abs_delta_mHa": _abs_error_mha(e_total, e_fci_total),
        "first_logged_abs_delta_mHa": float(limited_rows[0]["abs_delta_mHa"]),
        "plotted_final_total_energy_eh": float(limited_rows[-1]["total_energy_eh"]),
        "plotted_final_abs_delta_mHa": float(limited_rows[-1]["abs_delta_mHa"]),
        "plotted_iterations": int(limited_rows[-1]["cum_new_krylov"]),
        "trace_points": int(len(limited_rows)),
        "exactized_trace_points": int(len(exactized_rows)),
        "exactizeplus_trace_points": int(len(exactizeplus_rows)),
        "below_fci_detected": bool(below_fci_points),
        "below_reference_points": below_reference_report,
        "best_below_reference_total_energy_eh": (
            None if not below_fci_points else float(min(float(row["total_energy_eh"]) for row in below_fci_points))
        ),
        "min_signed_delta_mHa": float(min(row["delta_mHa"] for row in limited_rows)),
        "improves_over_first_point": float(limited_rows[-1]["abs_delta_mHa"]) < float(limited_rows[0]["abs_delta_mHa"]),
        "chemical_accuracy_target_mHa": chemical_accuracy_mha,
        "standard_reached_chemical_accuracy": bool(standard_hit),
        "standard_first_chemical_accuracy_iteration": None if standard_hit is None else int(standard_hit["cum_new_krylov"]),
        "exactized_reached_chemical_accuracy": bool(exactized_hit),
        "exactized_first_chemical_accuracy_iteration": None if exactized_hit is None else int(exactized_hit["cum_new_krylov"]),
        "exactized_first_chemical_accuracy_support_union_size": (
            None if exactized_hit is None or exactized_hit.get("support_union_size") is None else int(exactized_hit["support_union_size"])
        ),
        "exactizeplus_reached_chemical_accuracy": bool(exactizeplus_hit),
        "exactizeplus_first_chemical_accuracy_iteration": (
            None if exactizeplus_hit is None else int(exactizeplus_hit["cum_new_krylov"])
        ),
        "exactizeplus_first_chemical_accuracy_subspace_dim": (
            None
            if exactizeplus_hit is None or exactizeplus_hit.get("exactizeplus_subspace_dim") is None
            else int(exactizeplus_hit["exactizeplus_subspace_dim"])
        ),
        "best_exactized_total_energy_eh": (
            None
            if not exactized_rows
            else float(min(float(row["exactized_total_energy_eh"]) for row in exactized_rows if row["exactized_total_energy_eh"] is not None))
        ),
        "best_exactized_abs_delta_mHa": (
            None
            if not exactized_rows
            else float(min(float(row["exactized_abs_delta_mHa"]) for row in exactized_rows if row["exactized_abs_delta_mHa"] is not None))
        ),
        "best_exactizeplus_total_energy_eh": (
            None
            if not exactizeplus_rows
            else float(
                min(
                    float(row["exactizeplus_total_energy_eh"])
                    for row in exactizeplus_rows
                    if row["exactizeplus_total_energy_eh"] is not None
                )
            )
        ),
        "best_exactizeplus_abs_delta_mHa": (
            None
            if not exactizeplus_rows
            else float(
                min(
                    float(row["exactizeplus_abs_delta_mHa"])
                    for row in exactizeplus_rows
                    if row["exactizeplus_abs_delta_mHa"] is not None
                )
            )
        ),
        "early_stop_target_abs_delta_mHa": None if stop_abs_delta_mha is None else float(stop_abs_delta_mha),
        "early_stop_reached": bool(early_stop_reached),
        "wall_s": wall_s,
        "solver_stats": stats,
        "plot_png": str((_run_dir(case_dir, run_name) / "energy_error_vs_iteration.png")),
        "plot_pdf": str((_run_dir(case_dir, run_name) / "energy_error_vs_iteration.pdf")),
    }
    (_run_dir(case_dir, run_name) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_baseline_no_restart_audit(
    *,
    case_dir: Path,
    h1: np.ndarray,
    eri8: np.ndarray,
    ecore: float,
    norb: int,
    nelec: tuple[int, int],
    e_fci_total: float,
    params: CCIKThickRestartParams,
    max_cumulative_iterations: int,
) -> dict[str, Any]:
    audit_rows: list[dict[str, Any]] = []
    audit_params = CCIKThickRestartParams(
        **{
            **params.__dict__,
            "m_cycle": int(max_cumulative_iterations) + 1,
            "ncycles": 1,
            "nroot": 1,
            "exactized_checkpoint_every": None,
        }
    )

    def _collect(record: ThickRestartIterationRecord) -> None:
        audit_rows.append(_build_trace_row(record, ecore=ecore, e_fci_total=e_fci_total))

    e_total = float(
        ecore
        + ccik_ground_energy_dense_thick_restart(
            h1,
            eri8,
            norb,
            nelec,
            params=audit_params,
            iteration_callback=_collect,
        )
    )
    limited_rows = limit_iteration_rows(audit_rows, max_cumulative_iterations)
    monotone_nonincreasing = all(
        float(limited_rows[idx]["total_energy_eh"]) <= float(limited_rows[idx - 1]["total_energy_eh"]) + 1.0e-12
        for idx in range(1, len(limited_rows))
    )
    below_fci_detected = any(float(row["delta_mHa"]) < -1.0e-9 for row in limited_rows)
    payload = {
        "reference_total_energy_eh": e_fci_total,
        "final_total_energy_eh": e_total,
        "trace_points": int(len(limited_rows)),
        "monotone_nonincreasing": bool(monotone_nonincreasing),
        "below_fci_detected": bool(below_fci_detected),
        "min_signed_delta_mHa": float(min(row["delta_mHa"] for row in limited_rows)) if limited_rows else None,
    }
    (_run_dir(case_dir, "exact_oracle") / "baseline_no_restart_audit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def run_case(input_path: Path, *, force: bool = False) -> dict[str, Any]:
    input_path = input_path.resolve()
    spec = load_study_input(input_path)
    case_dir = _case_dir_from_input(input_path)
    study_summary_path = case_dir / "study_summary.json"
    if study_summary_path.exists() and not force:
        return json.loads(study_summary_path.read_text(encoding="utf-8"))

    case_label = str(spec["label"])
    max_cumulative_iterations = int(spec["study"]["max_cumulative_iterations"])
    plot_max_cumulative_iterations = int(spec.get("study", {}).get("plot_max_cumulative_iterations", max_cumulative_iterations))
    stop_abs_delta_mha = (
        None if spec.get("study", {}).get("stop_abs_delta_mHa") is None else float(spec["study"]["stop_abs_delta_mHa"])
    )
    run_specs = spec["runs"]
    run_labels = {str(run["name"]): str(run["label"]) for run in run_specs}
    backends = {str(run["solver"].get("backend", "qp")).strip() for run in run_specs}
    if len(backends) != 1:
        raise ValueError(f"Mixed study backends are not supported in one case: {sorted(backends)}")
    backend = next(iter(backends))

    if force:
        _clean_case_outputs(
            case_dir,
            [str(run["name"]) for run in run_specs],
            remove_system_setup=(backend == "qp"),
        )

    run_summaries: dict[str, dict[str, Any]] = {}
    exact_params: CCIKThickRestartParams | None = None
    if backend == "python":
        _, h1, eri8, ecore, norb, nelec, e_fci_total = _prepare_case(spec)
        for run_spec in run_specs:
            run_name = str(run_spec["name"])
            run_label = str(run_spec["label"])
            params = CCIKThickRestartParams(**run_spec["solver"]["params"])
            summary = _run_thick_trace(
                case_dir=case_dir,
                case_label=case_label,
                run_name=run_name,
                run_label=run_label,
                h1=h1,
                eri8=eri8,
                ecore=ecore,
                norb=norb,
                nelec=nelec,
                e_fci_total=e_fci_total,
                params=params,
                max_cumulative_iterations=max_cumulative_iterations,
                plot_max_cumulative_iterations=plot_max_cumulative_iterations,
                stop_abs_delta_mha=stop_abs_delta_mha,
                run_labels=run_labels,
            )
            run_summaries[run_name] = summary
            if run_name == "exact_oracle":
                exact_params = params

        if "exact_oracle" in run_summaries and exact_params is not None:
            baseline_audit = _run_baseline_no_restart_audit(
                case_dir=case_dir,
                h1=h1,
                eri8=eri8,
                ecore=ecore,
                norb=norb,
                nelec=nelec,
                e_fci_total=e_fci_total,
                params=exact_params,
                max_cumulative_iterations=max_cumulative_iterations,
            )
            run_summaries["exact_oracle"]["baseline_no_restart_audit"] = baseline_audit
            (_run_dir(case_dir, "exact_oracle") / "summary.json").write_text(
                json.dumps(run_summaries["exact_oracle"], indent=2),
                encoding="utf-8",
            )
    elif backend == "qp":
        prepared = _prepare_case_qp(spec, case_dir=case_dir, force=force)
        e_fci_total = prepared.reference_total_eh
        for run_spec in run_specs:
            run_name = str(run_spec["name"])
            run_label = str(run_spec["label"])
            summary = _run_qp_thick_trace(
                case_dir=case_dir,
                case_label=case_label,
                run_name=run_name,
                run_label=run_label,
                prepared=prepared,
                run_params=dict(run_spec["solver"]["params"]),
                max_cumulative_iterations=max_cumulative_iterations,
                plot_max_cumulative_iterations=plot_max_cumulative_iterations,
                stop_abs_delta_mha=stop_abs_delta_mha,
                run_labels=run_labels,
            )
            run_summaries[run_name] = summary
    else:
        raise ValueError(f"Unsupported solver backend for study case: {backend!r}")

    comparison = plot_case_comparison(
        case_dir=case_dir,
        case_label=case_label,
        run_labels=run_labels,
        max_cumulative_iterations=plot_max_cumulative_iterations,
    )
    payload = {
        "input_path": str(input_path),
        "case_dir": str(case_dir),
        "label": case_label,
        "backend": backend,
        "reference_total_energy_eh": e_fci_total,
        "reference_kind": None if backend != "qp" else prepared.reference_summary.get("reference_kind"),
        "reference_mode": None if backend != "qp" else prepared.reference_mode,
        "max_cumulative_iterations": max_cumulative_iterations,
        "plot_max_cumulative_iterations": plot_max_cumulative_iterations,
        "stop_abs_delta_mHa": stop_abs_delta_mha,
        "runs": run_summaries,
        "comparison_plot_png": None if comparison is None else str(comparison[0]),
        "comparison_plot_pdf": None if comparison is None else str(comparison[1]),
    }
    study_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Run live-updating N2 thick-restart iteration studies from inputs/*/input.json")
    p.add_argument("--inputs-dir", type=Path, default=ROOT / "inputs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--input", dest="input_files", action="append", type=Path, default=None)
    args = p.parse_args()

    input_files = args.input_files if args.input_files else discover_input_files(args.inputs_dir)
    if not input_files:
        raise SystemExit(f"No input.json files found under {args.inputs_dir}")

    payloads = [run_case(path, force=args.force) for path in input_files]
    print(json.dumps({"cases": payloads}, indent=2))


if __name__ == "__main__":
    main()
