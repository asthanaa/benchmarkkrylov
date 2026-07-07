#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Run QRTE for Fe cases using local MolVQE-style ROHF/CASSCF references."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "qrte_rohf_molvqe_fe_cases"
DEFAULT_PLOT_DIR = BENCHMARK_ROOT / "plots" / "qrte_rohf_molvqe_fe_cases"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_rohf_molvqe_fe_cases"
DEFAULT_CASE_JSONS = {
    "FeO": BENCHMARK_ROOT
    / "dataset"
    / "zenodo_19051536"
    / "extracted"
    / "data_repository"
    / "MolVQE-21"
    / "FeO.json",
    "feo4_2minus_ceo": BENCHMARK_ROOT
    / "dataset"
    / "zenodo_19051536"
    / "extracted"
    / "data_repository"
    / "MolVQE-21"
    / "feo4_2minus_ceo.json",
}
ACTIVE_ORBITAL_LABELS_BY_CASE = {
    "FeO": ["Fe 3d", "O 2p"],
    "feo4_2minus_ceo": ["Fe 3d"],
}
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


def _set_thread_limits() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _load_case(case_id: str, path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta = payload["meta"]
    molecule = meta["molecule"]
    casscf = meta["casscf"]
    scf_meta = meta.get("scf", {})
    atoms = molecule["atoms"]
    geometry_bohr = [[float(value) for value in atom["xyz"]] for atom in atoms]
    return {
        "case_id": case_id,
        "source_json_path": str(path.relative_to(BENCHMARK_ROOT)),
        "symbols": [str(atom["element"]) for atom in atoms],
        "geometry_bohr": geometry_bohr,
        "basis": str(molecule["basis"]),
        "charge": int(molecule["charge"]),
        "spin": int(molecule["spin"]),
        "symmetry": bool(molecule.get("symmetry", False)),
        "active_electrons": int(casscf["nelec"]),
        "active_orbitals": int(casscf["ncas"]),
        "dataset_scf_method": str(scf_meta.get("method", "")),
        "dataset_scf_converged": bool(scf_meta.get("converged", False)),
        "dataset_scf_energy_hartree": float(scf_meta["energy"]),
        "dataset_casscf_energy_hartree": float(casscf["energy"]),
        "dataset_casscf_converged": bool(casscf["converged"]),
        "dataset_casscf_natorb": bool(casscf["natorb"]),
    }


def _completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def _active_orbital_labels(case_id: str, use_avas: bool) -> list[str] | None:
    if not use_avas:
        return None
    labels = ACTIVE_ORBITAL_LABELS_BY_CASE.get(case_id)
    return None if labels is None else list(labels)


def _hamiltonian_source(case: dict[str, Any], force_natorb: bool) -> str:
    if force_natorb or bool(case["dataset_casscf_natorb"]):
        return "casscf_natorb"
    return "casscf"


def _dt_label(delta_t: float) -> str:
    return f"dt{delta_t:g}".replace(".", "p")


def _import_qcant(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))
    import QCANT
    import QCANT.qulacs_accel as qulacs_accel

    return QCANT, qulacs_accel


def _reference_kwargs(case: dict[str, Any], args: argparse.Namespace, active_orbital_labels: Sequence[str] | None):
    return {
        "basis": case["basis"],
        "charge": int(case["charge"]),
        "spin": int(case["spin"]),
        "active_electrons": int(case["active_electrons"]),
        "active_orbitals": int(case["active_orbitals"]),
        "hamiltonian_source": _hamiltonian_source(case, bool(args.force_natorb)),
        "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
        "active_orbital_labels": active_orbital_labels,
        "allow_unconverged_casscf": False,
        "geometry_unit": "Bohr",
        "molecule_symmetry": bool(case["symmetry"]),
        "scf_conv_tol": float(args.scf_conv_tol),
        "scf_diis_space": int(args.scf_diis_space),
        "scf_level_shift": float(args.scf_level_shift),
        "scf_max_cycle": int(args.scf_max_cycle),
        "scf_newton_if_not_converged": bool(args.newton_if_not_converged),
        "casscf_conv_tol": float(args.casscf_conv_tol),
        "casscf_max_cycle": int(args.casscf_max_cycle),
    }


def _build_local_reference(case: dict[str, Any], args: argparse.Namespace, qulacs_accel, active_orbital_labels):
    kwargs = _reference_kwargs(case, args, active_orbital_labels)
    payload = qulacs_accel._build_hamiltonian_payload(
        symbols=case["symbols"],
        geometry=np.asarray(case["geometry_bohr"], dtype=float),
        hamiltonian_cutoff=1e-20,
        **kwargs,
    )
    return {
        "hamiltonian_cache_path": payload.get("hamiltonian_cache_path"),
        "hamiltonian_loaded_from_cache": bool(payload.get("hamiltonian_loaded_from_cache", False)),
        "hamiltonian_source": kwargs["hamiltonian_source"],
        "local_reference_converged": bool(payload["reference_converged"]),
        "local_reference_energy_hartree": float(payload["reference_energy"]),
        "local_scf_converged": bool(payload.get("scf_converged")),
        "local_scf_energy_hartree": float(payload.get("scf_energy", float("nan"))),
        "qubits": int(payload["qubits"]),
    }


def _first_hit(min_history: np.ndarray, reference_energy: float) -> int | None:
    for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
        if float(energy) - reference_energy <= CHEMICAL_ACCURACY_HARTREE:
            return idx
    return None


def _write_history_csv(
    path: Path,
    min_history: np.ndarray,
    local_reference_energy: float,
    dataset_casscf_energy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "min_energy_hartree",
                "error_vs_local_reference_hartree",
                "error_vs_molvqe_casscf_hartree",
                "within_chemical_accuracy_vs_local_reference",
            ],
        )
        writer.writeheader()
        for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
            local_error = float(energy) - local_reference_energy
            writer.writerow(
                {
                    "iteration": idx,
                    "min_energy_hartree": float(energy),
                    "error_vs_local_reference_hartree": local_error,
                    "error_vs_molvqe_casscf_hartree": float(energy) - dataset_casscf_energy,
                    "within_chemical_accuracy_vs_local_reference": str(local_error <= CHEMICAL_ACCURACY_HARTREE).lower(),
                }
            )


def _plot_history(case_id: str, delta_t: float, run_dir: Path, plot_dir: Path) -> tuple[str, str] | None:
    history_path = run_dir / "min_energy_history.csv"
    summary_path = run_dir / "summary.json"
    if not history_path.exists() or not summary_path.exists():
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    iterations = [int(row["iteration"]) for row in rows]
    errors = [float(row["error_vs_local_reference_hartree"]) for row in rows]

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.semilogy(iterations, errors, marker="o", markersize=3.6, linewidth=1.8, label=f"dt={delta_t:g}")
    ax.axhline(
        CHEMICAL_ACCURACY_HARTREE,
        color="#c43b3b",
        linestyle="--",
        linewidth=1.4,
        label=f"chemical accuracy ({CHEMICAL_ACCURACY_HARTREE:g} Ha)",
    )
    ax.set_xlabel("QRTE iteration")
    ax.set_ylabel("Error vs local reference (Ha)")
    ax.set_title(f"{case_id} QRTE convergence, dt={delta_t:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    text = (
        "ref = {:.12f} Ha\nfinal error = {:.6g} Ha\nsource = {}\nCASSCF converged = {}"
    ).format(
        float(summary["local_reference_energy_hartree"]),
        errors[-1],
        summary["hamiltonian_source"],
        summary["local_reference_converged"],
    )
    ax.text(
        0.985,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#cfcfcf", alpha=0.9, boxstyle="round,pad=0.35"),
    )
    stem = f"{case_id}_qrte_rohf_molvqe_{_dt_label(delta_t)}_convergence"
    png_path = plot_dir / f"{stem}.png"
    pdf_path = plot_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return str(png_path), str(pdf_path)


def _plot_combined(case_id: str, dts: Sequence[float], outdir: Path, plot_dir: Path) -> tuple[str, str] | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    local_reference = None
    hamiltonian_source = None
    for delta_t in dts:
        run_dir = outdir / case_id / _dt_label(delta_t)
        history_path = run_dir / "min_energy_history.csv"
        summary_path = run_dir / "summary.json"
        if not history_path.exists() or not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("status") != "completed":
            continue
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        local_reference = float(summary["local_reference_energy_hartree"])
        hamiltonian_source = summary["hamiltonian_source"]
        series.append(
            (
                delta_t,
                [int(row["iteration"]) for row in rows],
                [float(row["error_vs_local_reference_hartree"]) for row in rows],
            )
        )

    if not series:
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    for delta_t, iterations, errors in series:
        ax.semilogy(iterations, errors, marker="o", markersize=3.4, linewidth=1.7, label=f"dt={delta_t:g}")
    ax.axhline(
        CHEMICAL_ACCURACY_HARTREE,
        color="#c43b3b",
        linestyle="--",
        linewidth=1.4,
        label=f"chemical accuracy ({CHEMICAL_ACCURACY_HARTREE:g} Ha)",
    )
    ax.set_xlabel("QRTE iteration")
    ax.set_ylabel("Error vs local reference (Ha)")
    ax.set_title(f"{case_id} QRTE convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    if local_reference is not None:
        ax.text(
            0.03,
            0.05,
            f"ref = {local_reference:.12f} Ha\nsource = {hamiltonian_source}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="#cfcfcf", alpha=0.9, boxstyle="round,pad=0.35"),
        )

    joined = "_".join(_dt_label(delta_t) for delta_t, _it, _err in series)
    stem = f"{case_id}_qrte_rohf_molvqe_{joined}_convergence"
    png_path = plot_dir / f"{stem}.png"
    pdf_path = plot_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return str(png_path), str(pdf_path)


def _run_case_dt(case: dict[str, Any], delta_t: float, args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    case_id = case["case_id"]
    case_dir = args.outdir / case_id / _dt_label(delta_t)
    summary_path = case_dir / "summary.json"
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed QRTE {case_id} dt={delta_t:g}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    active_orbital_labels = _active_orbital_labels(case_id, not args.no_avas_labels)
    start = time.perf_counter()
    try:
        reference_info = _build_local_reference(case, args, qulacs_accel, active_orbital_labels)
        local_reference_energy = float(reference_info["local_reference_energy_hartree"])
        early_stop_energy = local_reference_energy + CHEMICAL_ACCURACY_HARTREE if args.early_stop_chemical_accuracy else None
        hamiltonian_source = reference_info["hamiltonian_source"]
        reference_kwargs = _reference_kwargs(case, args, active_orbital_labels)

        input_snapshot = {
            **case,
            **reference_info,
            "active_orbital_labels": active_orbital_labels,
            "active_orbital_label_source": "local benchmark AVAS mapping; MolVQE JSON does not store orbital labels",
            "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
            "geometry_source": "MolVQE JSON meta.molecule atoms xyz, used directly with QCANT geometry_unit='Bohr'",
            "local_minus_dataset_casscf_hartree": local_reference_energy - case["dataset_casscf_energy_hartree"],
            "local_minus_dataset_scf_hartree": reference_info["local_scf_energy_hartree"] - case["dataset_scf_energy_hartree"],
            "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
        }
        run_config = {
            "casscf_conv_tol": float(args.casscf_conv_tol),
            "casscf_max_cycle": int(args.casscf_max_cycle),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(delta_t),
            "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
            "early_stop_energy_hartree": early_stop_energy,
            "evolution_mode": args.evolution_mode,
            "force_natorb": bool(args.force_natorb),
            "geometry_unit": "Bohr",
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": hamiltonian_source,
            "max_workers": int(args.max_workers),
            "method": "qrte",
            "n_steps": int(args.n_steps),
            "newton_if_not_converged": bool(args.newton_if_not_converged),
            "qcant_root": str(args.qcant_root.resolve()),
            "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
            "scf_conv_tol": float(args.scf_conv_tol),
            "scf_diis_space": int(args.scf_diis_space),
            "scf_level_shift": float(args.scf_level_shift),
            "scf_max_cycle": int(args.scf_max_cycle),
            "trotter_steps": int(args.trotter_steps),
            "use_avas_labels": not bool(args.no_avas_labels),
        }
        _write_json(case_dir / "input_snapshot.json", input_snapshot)
        _write_json(case_dir / "run_config.json", run_config)

        energies, _basis_states, _times, min_history = QCANT.qrte_qulacs(
            symbols=case["symbols"],
            geometry=np.asarray(case["geometry_bohr"], dtype=float),
            delta_t=float(delta_t),
            n_steps=int(args.n_steps),
            return_min_energy_history=True,
            evolution_mode=args.evolution_mode,
            trotter_steps=int(args.trotter_steps),
            max_workers=int(args.max_workers),
            early_stop_energy=early_stop_energy,
            **reference_kwargs,
        )

        energies = np.asarray(energies, dtype=float)
        min_history = np.asarray(min_history, dtype=float)
        hit_iteration = _first_hit(min_history, local_reference_energy)
        n_steps_completed = int(min_history.size)
        stopped_early = bool(
            args.early_stop_chemical_accuracy
            and hit_iteration is not None
            and n_steps_completed < int(args.n_steps)
        )
        _write_history_csv(
            case_dir / "min_energy_history.csv",
            min_history,
            local_reference_energy,
            float(case["dataset_casscf_energy_hartree"]),
        )

        summary = {
            "status": "completed",
            "case_id": case_id,
            "method": "qrte",
            "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
            "local_reference_energy_hartree": local_reference_energy,
            "local_reference_converged": reference_info["local_reference_converged"],
            "local_scf_energy_hartree": reference_info["local_scf_energy_hartree"],
            "local_scf_converged": reference_info["local_scf_converged"],
            "dataset_scf_energy_hartree": case["dataset_scf_energy_hartree"],
            "dataset_scf_converged": case["dataset_scf_converged"],
            "local_minus_dataset_scf_hartree": reference_info["local_scf_energy_hartree"] - case["dataset_scf_energy_hartree"],
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "dataset_casscf_converged": case["dataset_casscf_converged"],
            "dataset_casscf_natorb": case["dataset_casscf_natorb"],
            "local_minus_dataset_casscf_hartree": local_reference_energy - case["dataset_casscf_energy_hartree"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "active_orbital_labels": active_orbital_labels,
            "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(delta_t),
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_cache_path": reference_info["hamiltonian_cache_path"],
            "hamiltonian_loaded_from_cache": reference_info["hamiltonian_loaded_from_cache"],
            "hamiltonian_source": hamiltonian_source,
            "hit_chemical_accuracy": hit_iteration is not None,
            "hit_iteration": hit_iteration,
            "n_steps": n_steps_completed,
            "n_steps_completed": n_steps_completed,
            "n_steps_requested": int(args.n_steps),
            "plot_paths": None,
            "qubits": int(reference_info["qubits"]),
            "stopped_early": stopped_early,
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        if args.plot:
            plots = _plot_history(case_id, delta_t, case_dir, args.plot_dir)
            summary["plot_paths"] = plots
            _write_json(summary_path, summary)
        print(
            f"DONE QRTE {case_id} dt={delta_t:g} "
            f"hit={hit_iteration} ref={local_reference_energy:.12f}",
            flush=True,
        )
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case_id,
            "method": "qrte",
            "delta_t": float(delta_t),
            "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
            "dataset_scf_energy_hartree": case["dataset_scf_energy_hartree"],
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL QRTE {case_id} dt={delta_t:g}: {exc!r}", flush=True)
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--delta-t", action="append", type=float, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--scf-diis-space", type=int, default=12)
    parser.add_argument("--scf-level-shift", type=float, default=0.0)
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--casscf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--casscf-max-cycle", type=int, default=300)
    parser.add_argument("--newton-if-not-converged", action="store_true")
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--force-natorb", action="store_true")
    parser.add_argument("--no-avas-labels", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(plot=True)
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    selected_case_ids: Sequence[str] = tuple(args.case_id or DEFAULT_CASE_JSONS)
    missing = [case_id for case_id in selected_case_ids if case_id not in DEFAULT_CASE_JSONS]
    if missing:
        raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")
    delta_ts = tuple(args.delta_t or [0.05, 0.1])

    QCANT, qulacs_accel = _import_qcant(args.qcant_root)
    cases = [_load_case(case_id, DEFAULT_CASE_JSONS[case_id]) for case_id in selected_case_ids]
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    for case in cases:
        for delta_t in delta_ts:
            status = _run_case_dt(case, delta_t, args, QCANT, qulacs_accel)
            counts[status] = counts.get(status, 0) + 1
        if args.plot:
            combined = _plot_combined(case["case_id"], delta_ts, args.outdir, args.plot_dir)
            if combined is not None:
                print(f"PLOT {case['case_id']} {combined[0]} {combined[1]}", flush=True)

    print(json.dumps(counts, sort_keys=True), flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
