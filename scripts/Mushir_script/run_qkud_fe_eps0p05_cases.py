#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Run QKUD eps=0.05 for Fe systems with matched QRTE references."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_qrte_rohf_molvqe_fe_cases as rohf_ref  # noqa: E402
import run_qrte_sfx2c1erohf_dt0p05_cases as sfx_ref  # noqa: E402


BENCHMARK_ROOT = sfx_ref.BENCHMARK_ROOT
CHEMICAL_ACCURACY_HARTREE = sfx_ref.CHEMICAL_ACCURACY_HARTREE
DEFAULT_MANIFEST = sfx_ref.DEFAULT_MANIFEST
DEFAULT_QCANT_ROOT = sfx_ref.DEFAULT_QCANT_ROOT
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "qkud_fe_eps0.05"
DEFAULT_ROHF_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_rohf_molvqe_fe_cases"
DEFAULT_SFX_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb_sfx2c1erohf"
DEFAULT_SFX_UNCONVERGED_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb_sfx2c1erohf_unconverged"
DEFAULT_CASE_IDS = ("FeO", "fenoco3_minus", "feno_2plus")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


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
        return payload.get("method") in {"qkud", "reference_only"}
    return payload.get("method") == "qkud"


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


def _load_case(case_id: str, args: argparse.Namespace) -> dict[str, Any]:
    if case_id == "FeO":
        return rohf_ref._load_case(case_id, rohf_ref.DEFAULT_CASE_JSONS[case_id])

    manifest_by_case = sfx_ref._load_manifest(args.manifest)
    if case_id not in manifest_by_case:
        raise SystemExit(f"Case id not found in manifest: {case_id}")
    return sfx_ref._load_source_case(case_id, manifest_by_case[case_id])


def _rohf_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hamiltonian_cache_dir=args.rohf_hamiltonian_cache_dir,
        force_natorb=False,
        scf_conv_tol=float(args.rohf_scf_conv_tol),
        scf_diis_space=int(args.rohf_scf_diis_space),
        scf_level_shift=float(args.rohf_scf_level_shift),
        scf_max_cycle=int(args.rohf_scf_max_cycle),
        newton_if_not_converged=bool(args.rohf_newton_if_not_converged),
        casscf_conv_tol=float(args.rohf_casscf_conv_tol),
        casscf_max_cycle=int(args.rohf_casscf_max_cycle),
    )


def _sfx_args(case_id: str, args: argparse.Namespace) -> SimpleNamespace:
    cache_dir = (
        args.sfx_unconverged_hamiltonian_cache_dir
        if case_id == "fenoco3_minus"
        else args.sfx_hamiltonian_cache_dir
    )
    return SimpleNamespace(
        hamiltonian_cache_dir=cache_dir,
        allow_unconverged_casscf=(case_id == "fenoco3_minus"),
    )


def _run_feo(case: dict[str, Any], args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    case_id = case["case_id"]
    case_dir = args.outdir / case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path, reference_only=bool(args.reference_only)) and not args.force:
        print(f"SKIP completed qkud {case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    ref_args = _rohf_args(args)
    active_orbital_labels = rohf_ref._active_orbital_labels(case_id, not args.no_avas_labels)
    start = time.perf_counter()
    try:
        reference_info = rohf_ref._build_local_reference(case, ref_args, qulacs_accel, active_orbital_labels)
        local_reference_energy = float(reference_info["local_reference_energy_hartree"])
        early_stop_energy = local_reference_energy + CHEMICAL_ACCURACY_HARTREE if args.early_stop_chemical_accuracy else None
        reference_kwargs = rohf_ref._reference_kwargs(case, ref_args, active_orbital_labels)

        input_snapshot = {
            **case,
            **reference_info,
            "active_orbital_labels": active_orbital_labels,
            "active_orbital_label_source": "local benchmark AVAS mapping; MolVQE JSON does not store orbital labels",
            "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
            "geometry_source": "MolVQE JSON meta.molecule atoms xyz, used directly with QCANT geometry_unit='Bohr'",
            "local_minus_dataset_casscf_hartree": local_reference_energy - case["dataset_casscf_energy_hartree"],
            "local_minus_dataset_scf_hartree": reference_info["local_scf_energy_hartree"] - case["dataset_scf_energy_hartree"],
            "reference_scf": "ROHF",
            "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
        }
        run_config = {
            "casscf_conv_tol": float(ref_args.casscf_conv_tol),
            "casscf_max_cycle": int(ref_args.casscf_max_cycle),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
            "early_stop_energy_hartree": early_stop_energy,
            "epsilon": float(args.epsilon),
            "evolution_mode": args.evolution_mode,
            "geometry_unit": "Bohr",
            "hamiltonian_cache_dir": str(ref_args.hamiltonian_cache_dir),
            "hamiltonian_source": reference_info["hamiltonian_source"],
            "max_workers": int(args.max_workers),
            "method": "qkud",
            "n_steps": int(args.n_steps),
            "newton_if_not_converged": bool(ref_args.newton_if_not_converged),
            "qcant_root": str(args.qcant_root.resolve()),
            "reference_scf": "ROHF",
            "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
            "scf_conv_tol": float(ref_args.scf_conv_tol),
            "scf_diis_space": int(ref_args.scf_diis_space),
            "scf_level_shift": float(ref_args.scf_level_shift),
            "scf_max_cycle": int(ref_args.scf_max_cycle),
            "trotter_steps": int(args.trotter_steps),
            "use_avas_labels": not bool(args.no_avas_labels),
        }
        _write_json(case_dir / "input_snapshot.json", input_snapshot)
        _write_json(case_dir / "run_config.json", run_config)

        if args.reference_only:
            summary = {
                "status": "completed",
                "case_id": case_id,
                "method": "reference_only",
                "reference_scf": "ROHF",
                "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
                **reference_info,
                "active_orbital_labels": active_orbital_labels,
                "elapsed_seconds": float(time.perf_counter() - start),
            }
            _write_json(summary_path, summary)
            print(f"REFERENCE qkud {case_id} ref={local_reference_energy:.12f}", flush=True)
            return "completed"

        energies, _basis_states, min_history = QCANT.qkud_qulacs(
            symbols=case["symbols"],
            geometry=np.asarray(case["geometry_bohr"], dtype=float),
            epsilon=float(args.epsilon),
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
            "method": "qkud",
            "reference_scf": "ROHF",
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
            "epsilon": float(args.epsilon),
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "hamiltonian_cache_dir": str(ref_args.hamiltonian_cache_dir),
            "hamiltonian_cache_path": reference_info["hamiltonian_cache_path"],
            "hamiltonian_loaded_from_cache": reference_info["hamiltonian_loaded_from_cache"],
            "hamiltonian_source": reference_info["hamiltonian_source"],
            "hit_chemical_accuracy": hit_iteration is not None,
            "hit_iteration": hit_iteration,
            "n_steps": n_steps_completed,
            "n_steps_completed": n_steps_completed,
            "n_steps_requested": int(args.n_steps),
            "qubits": int(reference_info["qubits"]),
            "stopped_early": stopped_early,
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(f"DONE qkud {case_id} hit={hit_iteration} ref={local_reference_energy:.12f}", flush=True)
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case_id,
            "method": "qkud",
            "reference_scf": "ROHF",
            "reference_source": "local converged ROHF/CASSCF Hamiltonian reference",
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "epsilon": float(args.epsilon),
            "n_steps_requested": int(args.n_steps),
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL qkud {case_id}: {exc!r}", flush=True)
        return "failed"


def _run_sfx_case(case: dict[str, Any], args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    case_id = case["case_id"]
    case_dir = args.outdir / case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path, reference_only=bool(args.reference_only)) and not args.force:
        print(f"SKIP completed qkud {case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    ref_args = _sfx_args(case_id, args)
    active_orbital_labels = sfx_ref._active_orbital_labels(case_id, not args.no_avas_labels)
    start = time.perf_counter()
    try:
        with sfx_ref._sf_x2c1e_rohf_reference():
            reference_info = sfx_ref._build_local_reference(case, ref_args, qulacs_accel, active_orbital_labels)
            local_reference_energy = float(reference_info["local_reference_energy_hartree"])
            reference_source = (
                "local sfX2C1eROHF final unconverged CASSCF/natorb CASCI energy"
                if ref_args.allow_unconverged_casscf and not reference_info["local_reference_converged"]
                else "local sfX2C1eROHF CASSCF/natorb"
            )
            early_stop_energy = (
                local_reference_energy + CHEMICAL_ACCURACY_HARTREE
                if args.early_stop_chemical_accuracy
                else None
            )

            input_snapshot = {
                **case,
                **reference_info,
                "active_orbital_labels": active_orbital_labels,
                "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
                "casscf_energy_hartree": local_reference_energy,
                "reference_scf": "sfX2C1eROHF",
                "reference_source": reference_source,
                "allow_unconverged_casscf": bool(ref_args.allow_unconverged_casscf),
            }
            run_config = {
                "allow_unconverged_casscf": bool(ref_args.allow_unconverged_casscf),
                "active_orbital_labels": active_orbital_labels,
                "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
                "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
                "early_stop_energy_hartree": early_stop_energy,
                "epsilon": float(args.epsilon),
                "evolution_mode": args.evolution_mode,
                "hamiltonian_cache_dir": str(ref_args.hamiltonian_cache_dir),
                "hamiltonian_source": "casscf_natorb",
                "max_workers": int(args.max_workers),
                "method": "qkud",
                "n_steps": int(args.n_steps),
                "qcant_root": str(args.qcant_root.resolve()),
                "reference_scf": "sfX2C1eROHF",
                "reference_source": reference_source,
                "trotter_steps": int(args.trotter_steps),
            }
            _write_json(case_dir / "input_snapshot.json", input_snapshot)
            _write_json(case_dir / "run_config.json", run_config)

            if args.reference_only:
                summary = {
                    "status": "completed",
                    "case_id": case_id,
                    "method": "reference_only",
                    "reference_scf": "sfX2C1eROHF",
                    "reference_source": reference_source,
                    "allow_unconverged_casscf": bool(ref_args.allow_unconverged_casscf),
                    "casscf_energy_hartree": local_reference_energy,
                    "local_reference_energy_hartree": local_reference_energy,
                    "local_reference_converged": reference_info["local_reference_converged"],
                    "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
                    "local_minus_dataset_casscf_hartree": local_reference_energy - case["dataset_casscf_energy_hartree"],
                    "active_electrons": int(case["active_electrons"]),
                    "active_orbitals": int(case["active_orbitals"]),
                    "active_orbital_labels": active_orbital_labels,
                    "hamiltonian_cache_dir": str(ref_args.hamiltonian_cache_dir),
                    "hamiltonian_cache_path": reference_info["hamiltonian_cache_path"],
                    "hamiltonian_loaded_from_cache": reference_info["hamiltonian_loaded_from_cache"],
                    "hamiltonian_source": "casscf_natorb",
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
                _write_json(summary_path, summary)
                print(f"REFERENCE qkud {case_id} ref={local_reference_energy:.12f}", flush=True)
                return "completed"

            energies, _basis_states, min_history = QCANT.qkud_qulacs(
                symbols=case["symbols"],
                geometry=np.asarray(case["geometry_angstrom"], dtype=float),
                epsilon=float(args.epsilon),
                n_steps=int(args.n_steps),
                active_electrons=int(case["active_electrons"]),
                active_orbitals=int(case["active_orbitals"]),
                basis=case["basis"],
                charge=int(case["charge"]),
                spin=int(case["spin"]),
                return_min_energy_history=True,
                evolution_mode=args.evolution_mode,
                trotter_steps=int(args.trotter_steps),
                max_workers=int(args.max_workers),
                early_stop_energy=early_stop_energy,
                hamiltonian_cache_dir=str(ref_args.hamiltonian_cache_dir),
                hamiltonian_source="casscf_natorb",
                active_orbital_labels=active_orbital_labels,
                allow_unconverged_casscf=bool(ref_args.allow_unconverged_casscf),
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
            "method": "qkud",
            "reference_scf": "sfX2C1eROHF",
            "reference_source": reference_source,
            "allow_unconverged_casscf": bool(ref_args.allow_unconverged_casscf),
            "casscf_energy_hartree": local_reference_energy,
            "local_reference_energy_hartree": local_reference_energy,
            "local_reference_converged": reference_info["local_reference_converged"],
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "dataset_casscf_converged": case["dataset_casscf_converged"],
            "dataset_casscf_natorb": case["dataset_casscf_natorb"],
            "local_minus_dataset_casscf_hartree": local_reference_energy - case["dataset_casscf_energy_hartree"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
            "active_orbital_labels": active_orbital_labels,
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "epsilon": float(args.epsilon),
            "hit_iteration": hit_iteration,
            "hit_chemical_accuracy": hit_iteration is not None,
            "n_steps": n_steps_completed,
            "n_steps_requested": int(args.n_steps),
            "n_steps_completed": n_steps_completed,
            "stopped_early": stopped_early,
            "early_stop_energy_hartree": early_stop_energy,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "hamiltonian_cache_dir": str(ref_args.hamiltonian_cache_dir),
            "hamiltonian_cache_path": reference_info["hamiltonian_cache_path"],
            "hamiltonian_loaded_from_cache": reference_info["hamiltonian_loaded_from_cache"],
            "hamiltonian_source": "casscf_natorb",
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(f"DONE qkud {case_id} hit={hit_iteration} ref={local_reference_energy:.12f}", flush=True)
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case_id,
            "method": "qkud",
            "reference_scf": "sfX2C1eROHF",
            "reference_source": "local sfX2C1eROHF CASSCF/natorb",
            "allow_unconverged_casscf": bool(ref_args.allow_unconverged_casscf),
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "epsilon": float(args.epsilon),
            "n_steps_requested": int(args.n_steps),
            "hamiltonian_cache_dir": str(ref_args.hamiltonian_cache_dir),
            "hamiltonian_source": "casscf_natorb",
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL qkud {case_id}: {exc!r}", flush=True)
        return "failed"


def _run_case(case: dict[str, Any], args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    if case["case_id"] == "FeO":
        return _run_feo(case, args, QCANT, qulacs_accel)
    return _run_sfx_case(case, args, QCANT, qulacs_accel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--rohf-hamiltonian-cache-dir", type=Path, default=DEFAULT_ROHF_CACHE_DIR)
    parser.add_argument("--sfx-hamiltonian-cache-dir", type=Path, default=DEFAULT_SFX_CACHE_DIR)
    parser.add_argument("--sfx-unconverged-hamiltonian-cache-dir", type=Path, default=DEFAULT_SFX_UNCONVERGED_CACHE_DIR)
    parser.add_argument("--rohf-scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--rohf-scf-diis-space", type=int, default=12)
    parser.add_argument("--rohf-scf-level-shift", type=float, default=0.0)
    parser.add_argument("--rohf-scf-max-cycle", type=int, default=200)
    parser.add_argument("--rohf-casscf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--rohf-casscf-max-cycle", type=int, default=300)
    parser.add_argument("--no-rohf-newton-if-not-converged", dest="rohf_newton_if_not_converged", action="store_false")
    parser.set_defaults(rohf_newton_if_not_converged=True)
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--no-avas-labels", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    sfx_ref._set_thread_limits()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.rohf_hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)
    args.sfx_hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)
    args.sfx_unconverged_hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)

    selected_case_ids: Sequence[str] = tuple(args.case_id or DEFAULT_CASE_IDS)
    unknown = [case_id for case_id in selected_case_ids if case_id not in DEFAULT_CASE_IDS]
    if unknown:
        raise SystemExit(f"Unknown case id(s): {', '.join(unknown)}")

    QCANT, qulacs_accel = sfx_ref._import_qcant(args.qcant_root)
    cases = [_load_case(case_id, args) for case_id in selected_case_ids]
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    for case in cases:
        status = _run_case(case, args, QCANT, qulacs_accel)
        counts[status] = counts.get(status, 0) + 1

    print(json.dumps(counts, sort_keys=True), flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
