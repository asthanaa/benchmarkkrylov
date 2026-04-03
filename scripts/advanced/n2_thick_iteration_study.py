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
    discover_qp_paths_from_root,
    ensure_ezfio_defaults,
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
    Path("/Users/ayush.asthana/quantum_package/qp2_ccik_pristine_20260325"),
    Path("/Users/ayush.asthana/quantum_package/qp2_ccik_20260325"),
    Path("/private/tmp/qp2_fresh_20260320"),
)

QP_N2_SETUP_DONOR = (
    ROOT
    / "docs"
    / "assets"
    / "pub"
    / "n2_abs_residual_qp_thick_cas12_qp8_recheck_20260402"
    / "systems"
    / "n2_cas12_12_R2p50"
    / "system_setup"
    / "n2.ezfio"
)

QP_THICK_LINE_RE = re.compile(
    r"ccik\(thick\)\s+cycle=\s*(?P<cycle>\d+)\s+iter=\s*(?P<iter>\d+)\s+"
    r"total=\s*(?P<total>[+-]?\d+\.\d+)\s+ref=\s*(?P<ref>[+-]?\d+\.\d+)\s+"
    r"delta_mEh=\s*(?P<delta>[+-]?\d+\.\d+)"
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


def write_trace_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = _trace_paths(run_dir)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "cycle",
                "basis_size",
                "restart_seed_count",
                "cum_new_krylov",
                "electronic_energy_eh",
                "total_energy_eh",
                "delta_mHa",
                "abs_delta_mHa",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_live_status(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "live_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _case_plot_title(case_label: str, run_label: str, fixed_store_keep: int, fixed_propagate_keep: int) -> str:
    return f"{case_label} | {run_label} | Kprop={fixed_propagate_keep} Kstore={fixed_store_keep}"


def plot_run_trace(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    title: str,
    max_cumulative_iterations: int,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    png_path = run_dir / "energy_error_vs_iteration.png"
    pdf_path = run_dir / "energy_error_vs_iteration.pdf"

    xvals = [int(row["cum_new_krylov"]) for row in rows]
    yvals = [max(1.0e-12, float(row["abs_delta_mHa"])) for row in rows]
    latest = rows[-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xvals, yvals, color="#0B5FFF", linewidth=1.6, marker="o", markersize=2.8)
    ax.set_yscale("log")
    ax.set_xlim(0, int(max_cumulative_iterations))
    ax.set_xlabel("Cumulative new Krylov vectors")
    ax.set_ylabel("|Energy error| (mHa)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.axhline(1.0, color="#C0392B", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.annotate(
        f"latest: k={latest['cum_new_krylov']}\n|ΔE|={float(latest['abs_delta_mHa']):.6f} mHa",
        xy=(int(latest["cum_new_krylov"]), max(1.0e-12, float(latest["abs_delta_mHa"]))),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
    )
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _load_existing_trace_rows(run_dir: Path) -> list[dict[str, Any]]:
    _, json_path = _trace_paths(run_dir)
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text(encoding="utf-8"))


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
    if not traces:
        return None

    png_path = case_dir / "energy_error_vs_iteration_comparison.png"
    pdf_path = case_dir / "energy_error_vs_iteration_comparison.pdf"
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
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
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
    if not QP_N2_SETUP_DONOR.exists():
        raise FileNotFoundError(f"Missing N2 donor EZFIO for QP study setup: {QP_N2_SETUP_DONOR}")
    if ezfio_path.exists():
        shutil.rmtree(ezfio_path)
    shutil.copytree(QP_N2_SETUP_DONOR, ezfio_path)
    qp = discover_qp_paths_from_root(qp_root)
    ensure_ezfio_defaults(ezfio_path, qp_root=qp_root)
    _rewrite_n2_geometry_direct(qp_root, ezfio_path, distance_ang=distance_ang)
    write_ezfio_value(ezfio_path, "nuclei/io_nuclear_repulsion", "None")
    write_ezfio_value(ezfio_path, "ao_two_e_ints/io_ao_two_e_integrals", "Write")
    write_ezfio_value(ezfio_path, "mo_two_e_ints/io_mo_two_e_integrals", "Write")
    write_ezfio_value(ezfio_path, "mo_one_e_ints/io_mo_one_e_integrals", "None")
    subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"source {shlex.quote(str(qp.quantum_package_rc.resolve()))} >/dev/null 2>&1 || true && {shlex.quote(str(qp.scf_binary.resolve()))} {shlex.quote(str(ezfio_path.resolve()))}",
        ],
        check=True,
        text=True,
        cwd=str(ezfio_path.parent),
        env={**os.environ, "QP_ROOT": str(qp_root), **qp_parallel_env(8)},
        capture_output=True,
    )
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
                omp_threads=int(payload.get("omp_threads", omp_threads)),
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
    run_qp_setup(
        qp,
        basis=str(molecule["basis"]),
        ezfio=ezfio_path,
        xyz=xyz_path,
        ncore=int(cas_cfg.get("ncore", 0)),
        ncas=int(cas_cfg["ncas"]),
        nmo_total=nmo_total,
    )
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
    fixed_store_keep = int(run_params["fixed_store_keep"])
    fixed_propagate_keep = int(run_params["fixed_propagate_keep"])
    m_cycle = int(run_params["m_cycle"])
    ncycles = int(run_params["ncycles"])
    nroot = int(run_params["nroot"])
    selector_mode = str(run_params.get("selector_mode", "abs_residual"))
    bridge = fixed_budget_bridge_kwargs(
        fixed_store_keep=fixed_store_keep,
        fixed_propagate_keep=fixed_propagate_keep,
        external_keep=max(2048, fixed_store_keep),
    )
    # In the QP plugin, `ccik_basis_size` and `ccik_restart_m_cycle` both set
    # the within-cycle Krylov basis size. For example, `m_cycle = 51` means the
    # first thick restart happens after 50 newly generated vectors because cycle
    # 1 starts from a single seed vector.
    return QPCCIKParams(
        ccik_basis_size=m_cycle,
        ccik_selector_mode=selector_mode,
        ccik_matvec_backend="connected_sd_fullq",
        ccik_conv_tol=1.0e-10,
        ccik_ritz_overlap_tol=1.0e-8,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=False,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=False,
        ccik_max_cycle=ncycles,
        ccik_restart_m_cycle=m_cycle,
        ccik_restart_nroot=nroot,
        ccik_reference_total_energy=float(reference_total_eh),
        ccik_print_iteration_error=True,
        ccik_verbose=False,
        **bridge,
    )


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
        "electronic_energy_eh": float(total_energy_eh - float(ecore)),
        "total_energy_eh": float(total_energy_eh),
        "delta_mHa": float(delta_mha),
        "abs_delta_mHa": abs(float(delta_mha)),
    }


def _run_qp_thick_trace(
    *,
    case_dir: Path,
    case_label: str,
    run_name: str,
    run_label: str,
    prepared: PreparedQpCase,
    run_params: dict[str, Any],
    max_cumulative_iterations: int,
    stop_abs_delta_mha: float | None,
    run_labels: dict[str, str],
) -> dict[str, Any]:
    qp = discover_qp_paths_from_root(prepared.qp_root)
    run_dir = _run_dir(case_dir, run_name)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_ezfio = runtime_dir / prepared.ezfio_path.name
    shutil.copytree(prepared.ezfio_path, runtime_ezfio)
    ensure_ezfio_defaults(runtime_ezfio, qp_root=prepared.qp_root)

    params = _make_qp_thick_params(run_params=run_params, reference_total_eh=prepared.reference_total_eh)
    set_ccik_params(qp, runtime_ezfio, params)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_reference_total_energy", float(prepared.reference_total_eh))
    write_ezfio_value(runtime_ezfio, "ccik/ccik_print_iteration_error", True)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace", False)

    stdout_path = run_dir / "ccik_stdout.log"
    stderr_path = run_dir / "ccik_stderr.log"
    env = {**os.environ, "QP_ROOT": str(prepared.qp_root), **qp_parallel_env(prepared.omp_threads)}
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
            match = QP_THICK_LINE_RE.search(raw)
            if match is None:
                continue
            row = _build_qp_trace_row(
                cycle=int(match.group("cycle")),
                iter_idx=int(match.group("iter")),
                total_energy_eh=float(match.group("total")),
                delta_mha=float(match.group("delta")),
                ecore=prepared.ecore,
                restart_m_cycle=int(run_params["m_cycle"]),
                restart_nroot=int(run_params["nroot"]),
            )
            rows.append(row)
            _refresh_run_outputs(
                case_dir=case_dir,
                case_label=case_label,
                run_name=run_name,
                run_label=run_label,
                rows=rows,
                max_cumulative_iterations=max_cumulative_iterations,
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
    effective_final_total = float(limited_rows[-1]["total_energy_eh"]) if final_total is None else float(final_total)
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
        "below_fci_detected": bool(below_fci_points),
        "min_signed_delta_mHa": float(min(row["delta_mHa"] for row in limited_rows)),
        "improves_over_first_point": float(limited_rows[-1]["abs_delta_mHa"]) < float(limited_rows[0]["abs_delta_mHa"]),
        "early_stop_target_abs_delta_mHa": early_stop_target,
        "early_stop_reached": bool(early_stop_reached),
        "wall_s": wall_s,
        "returncode": returncode,
        "plot_png": str((run_dir / "energy_error_vs_iteration.png")),
        "plot_pdf": str((run_dir / "energy_error_vs_iteration.pdf")),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
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
    return {
        "cycle": int(record.cycle),
        "basis_size": int(record.basis_size),
        "restart_seed_count": int(record.restart_seed_count),
        "cum_new_krylov": int(record.cum_new_krylov),
        "electronic_energy_eh": float(record.electronic_energy_eh),
        "total_energy_eh": total_energy,
        "delta_mHa": _signed_error_mha(total_energy, e_fci_total),
        "abs_delta_mHa": _abs_error_mha(total_energy, e_fci_total),
    }


def _refresh_run_outputs(
    *,
    case_dir: Path,
    case_label: str,
    run_name: str,
    run_label: str,
    rows: list[dict[str, Any]],
    max_cumulative_iterations: int,
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
        max_cumulative_iterations=max_cumulative_iterations,
    )
    latest = limited_rows[-1]
    write_live_status(
        run_dir,
        {
            "label": run_label,
            "cum_new_krylov": int(latest["cum_new_krylov"]),
            "cycle": int(latest["cycle"]),
            "basis_size": int(latest["basis_size"]),
            "total_energy_eh": float(latest["total_energy_eh"]),
            "delta_mHa": float(latest["delta_mHa"]),
            "abs_delta_mHa": float(latest["abs_delta_mHa"]),
        },
    )
    plot_case_comparison(
        case_dir=case_dir,
        case_label=case_label,
        run_labels=run_labels,
        max_cumulative_iterations=max_cumulative_iterations,
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
    stop_abs_delta_mha: float | None,
    run_labels: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    early_stop_reached = False
    t0 = time.perf_counter()

    def _collect(record: ThickRestartIterationRecord) -> None:
        rows.append(_build_trace_row(record, ecore=ecore, e_fci_total=e_fci_total))
        _refresh_run_outputs(
            case_dir=case_dir,
            case_label=case_label,
            run_name=run_name,
            run_label=run_label,
            rows=rows,
            max_cumulative_iterations=max_cumulative_iterations,
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
        "below_fci_detected": bool(below_fci_points),
        "min_signed_delta_mHa": float(min(row["delta_mHa"] for row in limited_rows)),
        "improves_over_first_point": float(limited_rows[-1]["abs_delta_mHa"]) < float(limited_rows[0]["abs_delta_mHa"]),
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
    spec = load_study_input(input_path)
    case_dir = _case_dir_from_input(input_path)
    study_summary_path = case_dir / "study_summary.json"
    if study_summary_path.exists() and not force:
        return json.loads(study_summary_path.read_text(encoding="utf-8"))

    case_label = str(spec["label"])
    max_cumulative_iterations = int(spec["study"]["max_cumulative_iterations"])
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
        max_cumulative_iterations=max_cumulative_iterations,
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
