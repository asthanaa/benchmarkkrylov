#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Run QRTE dt=0.05 for MolVQE cases with explicit active-orbital overrides."""

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


ANGSTROM_PER_BOHR = 0.529177210903
CHEMICAL_ACCURACY_HARTREE = 1.6e-3
DEFAULT_CASSCF_MATCH_TOLERANCE_HARTREE = 1e-5

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_ACTIVE_ORBITAL_OVERRIDES = BENCHMARK_ROOT / "manifests" / "molvqe21_active_orbital_overrides.json"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "qrte_corrected_active_space_dt0.05"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb_corrected_active_space_dt0p05"
DEFAULT_CASE_IDS = (
    "cyclobutadiene_ceo",
    "m_Benzyne_equi_cc_pVTZ_0",
    "m_Benzyne_equi_cc_pVTZ_2",
)


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


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _load_active_orbital_overrides(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"active orbital override file must contain a JSON object: {path}")

    overrides: dict[str, dict[str, Any]] = {}
    for case_id, override in payload.items():
        if not isinstance(override, dict):
            raise ValueError(f"active orbital override for {case_id!r} must be a JSON object")
        indices = [int(index) for index in override["active_orbital_indices"]]
        if len(indices) != len(set(indices)):
            raise ValueError(f"active orbital override for {case_id!r} contains duplicate indices")
        base = int(override.get("active_orbital_index_base", 0))
        if any(index < base for index in indices):
            raise ValueError(f"active orbital override for {case_id!r} is inconsistent with base={base}")
        overrides[str(case_id)] = {
            "active_orbital_indices": indices,
            "active_orbital_index_base": base,
            "selection_method": str(override.get("selection_method", "pyscf_canonical_rohf_mo_indices")),
        }
    return overrides


def _load_source_case(
    case_id: str,
    manifest_row: dict[str, str],
    active_orbital_override: dict[str, Any],
) -> dict[str, Any]:
    source_path = BENCHMARK_ROOT / manifest_row["source_json_path"]
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta = payload["meta"]
    molecule = meta["molecule"]
    casscf = meta["casscf"]
    atoms = molecule["atoms"]
    geometry_bohr = [[float(value) for value in atom["xyz"]] for atom in atoms]
    geometry_angstrom = [
        [float(value) * ANGSTROM_PER_BOHR for value in xyz]
        for xyz in geometry_bohr
    ]
    active_orbitals = int(casscf["ncas"])
    active_indices = list(active_orbital_override["active_orbital_indices"])
    if len(active_indices) != active_orbitals:
        raise ValueError(
            f"{case_id}: active_orbital_indices has length {len(active_indices)}, "
            f"but MolVQE ncas={active_orbitals}"
        )

    return {
        "case_id": case_id,
        "source_name": manifest_row["source_name"],
        "source_json_path": manifest_row["source_json_path"],
        "symbols": [str(atom["element"]) for atom in atoms],
        "geometry_bohr": geometry_bohr,
        "geometry_angstrom": geometry_angstrom,
        "basis": str(molecule["basis"]),
        "charge": int(molecule["charge"]),
        "spin": int(molecule["spin"]),
        "active_electrons": int(casscf["nelec"]),
        "active_orbitals": active_orbitals,
        "active_orbital_indices": active_indices,
        "active_orbital_index_base": int(active_orbital_override["active_orbital_index_base"]),
        "active_orbital_selection_method": str(active_orbital_override["selection_method"]),
        "dataset_scf_method": str(meta.get("scf", {}).get("method", "")),
        "dataset_scf_energy_hartree": float(meta.get("scf", {}).get("energy", "nan")),
        "dataset_casscf_energy_hartree": float(casscf["energy"]),
        "dataset_casscf_converged": bool(casscf["converged"]),
        "dataset_casscf_natorb": bool(casscf["natorb"]),
    }


def _first_hit(min_history: np.ndarray, reference_energy: float) -> int | None:
    for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
        if float(energy) - reference_energy <= CHEMICAL_ACCURACY_HARTREE:
            return idx
    return None


def _write_history_csv(path: Path, min_history: np.ndarray, reference_energy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "min_energy_hartree",
                "error_vs_local_reference_hartree",
                "within_chemical_accuracy",
            ],
        )
        writer.writeheader()
        for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
            error = float(energy) - reference_energy
            writer.writerow(
                {
                    "iteration": idx,
                    "min_energy_hartree": float(energy),
                    "error_vs_local_reference_hartree": error,
                    "within_chemical_accuracy": str(error <= CHEMICAL_ACCURACY_HARTREE).lower(),
                }
            )


def _completed(summary_path: Path, *, reference_only: bool) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "completed":
        return False
    if reference_only:
        return payload.get("method") in {"qrte", "reference_only"}
    return payload.get("method") == "qrte"


def _import_qcant(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))
    import QCANT
    import QCANT.qulacs_accel as qulacs_accel

    return QCANT, qulacs_accel


def _reference_kwargs(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "symbols": case["symbols"],
        "geometry": np.asarray(case["geometry_angstrom"], dtype=float),
        "basis": case["basis"],
        "charge": int(case["charge"]),
        "spin": int(case["spin"]),
        "active_electrons": int(case["active_electrons"]),
        "active_orbitals": int(case["active_orbitals"]),
        "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
        "hamiltonian_source": "casscf_natorb",
        "active_orbital_indices": list(case["active_orbital_indices"]),
        "active_orbital_index_base": int(case["active_orbital_index_base"]),
        "allow_unconverged_casscf": False,
        "scf_conv_tol": None if args.scf_conv_tol is None else float(args.scf_conv_tol),
        "scf_diis_space": int(args.scf_diis_space),
        "scf_level_shift": float(args.scf_level_shift),
        "scf_max_cycle": int(args.scf_max_cycle),
        "scf_newton_if_not_converged": bool(args.scf_newton_if_not_converged),
        "casscf_conv_tol": float(args.casscf_conv_tol),
        "casscf_max_cycle": int(args.casscf_max_cycle),
    }


def _build_local_reference(case: dict[str, Any], args: argparse.Namespace, qulacs_accel):
    payload = qulacs_accel._build_hamiltonian_payload(
        hamiltonian_cutoff=1e-20,
        **_reference_kwargs(case, args),
    )
    reference_energy = float(payload["reference_energy"])
    energy_delta = reference_energy - float(case["dataset_casscf_energy_hartree"])
    energy_delta_abs = abs(energy_delta)
    reference_converged = bool(payload["reference_converged"])
    reference_matches_dataset = energy_delta_abs <= float(args.casscf_match_tolerance)
    if not reference_converged:
        raise RuntimeError(f"{case['case_id']}: corrected CASSCF/natorb did not converge")
    if not reference_matches_dataset:
        raise RuntimeError(
            f"{case['case_id']}: corrected CASSCF/natorb energy differs from MolVQE by "
            f"{energy_delta:.12g} Eh; tolerance is {float(args.casscf_match_tolerance):.3g} Eh"
        )
    return {
        "local_reference_energy_hartree": reference_energy,
        "local_reference_converged": reference_converged,
        "local_minus_dataset_casscf_hartree": energy_delta,
        "abs_local_minus_dataset_casscf_hartree": energy_delta_abs,
        "casscf_match_tolerance_hartree": float(args.casscf_match_tolerance),
        "local_reference_matches_dataset": reference_matches_dataset,
        "hamiltonian_cache_path": payload.get("hamiltonian_cache_path"),
        "hamiltonian_loaded_from_cache": bool(payload.get("hamiltonian_loaded_from_cache", False)),
        "active_space_selection": payload.get("active_space_selection"),
        "effective_active_electrons": int(payload.get("active_electrons", case["active_electrons"])),
        "scf_energy_hartree": float(payload.get("scf_energy", "nan")),
        "scf_converged": payload.get("scf_converged"),
    }


def _run_case(case: dict[str, Any], args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    case_id = case["case_id"]
    case_dir = args.outdir / case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path, reference_only=bool(args.reference_only)) and not args.force:
        print(f"SKIP completed qrte {case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reference_info: dict[str, Any] = {}
    try:
        reference_info = _build_local_reference(case, args, qulacs_accel)
        local_reference_energy = float(reference_info["local_reference_energy_hartree"])
        early_stop_energy = (
            local_reference_energy + CHEMICAL_ACCURACY_HARTREE
            if args.early_stop_chemical_accuracy
            else None
        )

        input_snapshot = {
            **case,
            **reference_info,
            "active_orbital_overrides_path": str(args.active_orbital_overrides),
            "active_space_source": "explicit PySCF canonical MO indices from override JSON",
            "casscf_energy_hartree": local_reference_energy,
            "reference_source": "local corrected CASSCF/natorb Hamiltonian reference",
            "scf_policy": "QCANT/PySCF RHF for spin=0, ROHF for spin>0",
        }
        run_config = {
            "active_orbital_indices": list(case["active_orbital_indices"]),
            "active_orbital_index_base": int(case["active_orbital_index_base"]),
            "active_orbital_selection_method": case["active_orbital_selection_method"],
            "casscf_conv_tol": float(args.casscf_conv_tol),
            "casscf_match_tolerance_hartree": float(args.casscf_match_tolerance),
            "casscf_max_cycle": int(args.casscf_max_cycle),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(args.delta_t),
            "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
            "early_stop_energy_hartree": early_stop_energy,
            "evolution_mode": args.evolution_mode,
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": "casscf_natorb",
            "max_workers": int(args.max_workers),
            "method": "qrte",
            "n_steps": int(args.n_steps),
            "qcant_root": str(args.qcant_root.resolve()),
            "reference_source": "local corrected CASSCF/natorb Hamiltonian reference",
            "scf_conv_tol": None if args.scf_conv_tol is None else float(args.scf_conv_tol),
            "scf_diis_space": int(args.scf_diis_space),
            "scf_level_shift": float(args.scf_level_shift),
            "scf_max_cycle": int(args.scf_max_cycle),
            "scf_newton_if_not_converged": bool(args.scf_newton_if_not_converged),
            "trotter_steps": int(args.trotter_steps),
        }
        _write_json(case_dir / "input_snapshot.json", input_snapshot)
        _write_json(case_dir / "run_config.json", run_config)

        if args.reference_only:
            summary = {
                "status": "completed",
                "case_id": case_id,
                "method": "reference_only",
                **reference_info,
                "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
                "active_orbital_indices": list(case["active_orbital_indices"]),
                "active_orbital_index_base": int(case["active_orbital_index_base"]),
                "hamiltonian_source": "casscf_natorb",
                "elapsed_seconds": float(time.perf_counter() - start),
            }
            _write_json(summary_path, summary)
            print(f"REFERENCE {case_id} ref={local_reference_energy:.12f}", flush=True)
            return "completed"

        energies, _basis_states, _times, min_history = QCANT.qrte_qulacs(
            delta_t=float(args.delta_t),
            n_steps=int(args.n_steps),
            return_min_energy_history=True,
            evolution_mode=args.evolution_mode,
            trotter_steps=int(args.trotter_steps),
            max_workers=int(args.max_workers),
            early_stop_energy=early_stop_energy,
            **_reference_kwargs(case, args),
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
        _write_history_csv(case_dir / "min_energy_history.csv", min_history, local_reference_energy)

        summary = {
            "status": "completed",
            "case_id": case_id,
            "method": "qrte",
            "reference_source": "local corrected CASSCF/natorb Hamiltonian reference",
            "casscf_energy_hartree": local_reference_energy,
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "dataset_casscf_converged": case["dataset_casscf_converged"],
            "dataset_casscf_natorb": case["dataset_casscf_natorb"],
            **reference_info,
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "active_orbital_indices": list(case["active_orbital_indices"]),
            "active_orbital_index_base": int(case["active_orbital_index_base"]),
            "active_orbital_selection_method": case["active_orbital_selection_method"],
            "active_space_source": "explicit PySCF canonical MO indices from override JSON",
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(args.delta_t),
            "hit_iteration": hit_iteration,
            "hit_chemical_accuracy": hit_iteration is not None,
            "n_steps": n_steps_completed,
            "n_steps_requested": int(args.n_steps),
            "n_steps_completed": n_steps_completed,
            "stopped_early": stopped_early,
            "early_stop_energy_hartree": early_stop_energy,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": "casscf_natorb",
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(f"DONE qrte {case_id} hit={hit_iteration} ref={local_reference_energy:.12f}", flush=True)
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case_id,
            "method": "qrte_reference_check" if not reference_info else "qrte",
            "reference_source": "local corrected CASSCF/natorb Hamiltonian reference",
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "active_orbital_indices": list(case["active_orbital_indices"]),
            "active_orbital_index_base": int(case["active_orbital_index_base"]),
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "casscf_match_tolerance_hartree": float(args.casscf_match_tolerance),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(args.delta_t),
            "n_steps_requested": int(args.n_steps),
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": "casscf_natorb",
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL qrte {case_id}: {exc!r}", flush=True)
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--active-orbital-overrides", type=Path, default=DEFAULT_ACTIVE_ORBITAL_OVERRIDES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=36)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument("--casscf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--casscf-max-cycle", type=int, default=200)
    parser.add_argument("--casscf-match-tolerance", type=float, default=DEFAULT_CASSCF_MATCH_TOLERANCE_HARTREE)
    parser.add_argument("--scf-conv-tol", type=float, default=None)
    parser.add_argument("--scf-diis-space", type=int, default=12)
    parser.add_argument("--scf-level-shift", type=float, default=0.5)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--no-scf-newton-if-not-converged", dest="scf_newton_if_not_converged", action="store_false")
    parser.set_defaults(scf_newton_if_not_converged=True)
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)

    selected_case_ids: Sequence[str] = tuple(args.case_id or DEFAULT_CASE_IDS)
    manifest_by_case = _load_manifest(args.manifest)
    overrides_by_case = _load_active_orbital_overrides(args.active_orbital_overrides)
    missing_manifest = [case_id for case_id in selected_case_ids if case_id not in manifest_by_case]
    missing_overrides = [case_id for case_id in selected_case_ids if case_id not in overrides_by_case]
    if missing_manifest:
        raise SystemExit(f"Case id(s) not found in manifest: {', '.join(missing_manifest)}")
    if missing_overrides:
        raise SystemExit(f"Case id(s) not found in active orbital overrides: {', '.join(missing_overrides)}")

    QCANT, qulacs_accel = _import_qcant(args.qcant_root)
    cases = [
        _load_source_case(case_id, manifest_by_case[case_id], overrides_by_case[case_id])
        for case_id in selected_case_ids
    ]
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    for case in cases:
        status = _run_case(case, args, QCANT, qulacs_accel)
        counts[status] = counts.get(status, 0) + 1

    print(json.dumps(counts, sort_keys=True), flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
