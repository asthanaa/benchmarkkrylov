#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Run Ferrocene QKUD with sfX2C1eROHF SCF and non-natorb CASSCF."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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

from run_qrte_sfx2c1erohf_casscf_ferrocene_dt0p05 import (  # noqa: E402
    ACTIVE_ORBITAL_LABELS,
    BENCHMARK_ROOT,
    CHEMICAL_ACCURACY_HARTREE,
    DEFAULT_CASE_JSON,
    DEFAULT_CASSCF_MATCH_TOLERANCE_HARTREE,
    DEFAULT_HAMILTONIAN_CACHE_DIR,
    DEFAULT_QCANT_ROOT,
    _build_local_reference,
    _first_hit,
    _import_qcant,
    _load_case,
    _reference_kwargs,
    _set_thread_limits,
    _sf_x2c1e_rohf_reference,
    _write_history_csv,
    _write_json,
)


DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "qkud_sfx2c1erohf_casscf_ferrocene_eps0.05"


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


def _run_case(case: dict[str, Any], args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    case_id = case["case_id"]
    case_dir = args.outdir / case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path, reference_only=bool(args.reference_only)) and not args.force:
        print(f"SKIP completed qkud {case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    active_orbital_labels = None if args.no_avas_labels else list(ACTIVE_ORBITAL_LABELS)
    start = time.perf_counter()
    try:
        with _sf_x2c1e_rohf_reference():
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
                "active_orbital_labels": active_orbital_labels,
                "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
                "active_orbital_label_source": "local benchmark AVAS mapping; MolVQE JSON does not store orbital labels",
                "casscf_energy_hartree": local_reference_energy,
                "reference_scf": "sfX2C1eROHF",
                "reference_source": "local sfX2C1eROHF CASSCF Hamiltonian reference without natural orbitals",
            }
            run_config = {
                "active_orbital_labels": active_orbital_labels,
                "casscf_conv_tol": float(args.casscf_conv_tol),
                "casscf_match_tolerance_hartree": float(args.casscf_match_tolerance),
                "casscf_max_cycle": int(args.casscf_max_cycle),
                "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
                "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
                "early_stop_energy_hartree": early_stop_energy,
                "epsilon": float(args.epsilon),
                "evolution_mode": args.evolution_mode,
                "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
                "hamiltonian_source": "casscf",
                "max_workers": int(args.max_workers),
                "method": "qkud",
                "n_steps": int(args.n_steps),
                "qcant_root": str(args.qcant_root.resolve()),
                "reference_scf": "sfX2C1eROHF",
                "reference_source": "local sfX2C1eROHF CASSCF Hamiltonian reference without natural orbitals",
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
                    "reference_scf": "sfX2C1eROHF",
                    "reference_source": "local sfX2C1eROHF CASSCF Hamiltonian reference without natural orbitals",
                    "casscf_energy_hartree": local_reference_energy,
                    "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
                    "dataset_casscf_natorb": case["dataset_casscf_natorb"],
                    "local_casscf_natorb": False,
                    "active_electrons": int(case["active_electrons"]),
                    "active_orbitals": int(case["active_orbitals"]),
                    "active_orbital_labels": active_orbital_labels,
                    "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
                    **reference_info,
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
                _write_json(summary_path, summary)
                print(f"REFERENCE qkud {case_id} ref={local_reference_energy:.12f}", flush=True)
                return "completed"

            energies, _basis_states, min_history = QCANT.qkud_qulacs(
                epsilon=float(args.epsilon),
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
            "reference_source": "local sfX2C1eROHF CASSCF Hamiltonian reference without natural orbitals",
            "casscf_energy_hartree": local_reference_energy,
            "local_reference_energy_hartree": local_reference_energy,
            "local_reference_converged": reference_info["local_reference_converged"],
            "local_reference_matches_dataset": reference_info["local_reference_matches_dataset"],
            "local_scf_energy_hartree": reference_info["local_scf_energy_hartree"],
            "local_scf_converged": reference_info["local_scf_converged"],
            "dataset_scf_method": case["dataset_scf_method"],
            "dataset_scf_energy_hartree": case["dataset_scf_energy_hartree"],
            "dataset_scf_converged": case["dataset_scf_converged"],
            "local_minus_dataset_scf_hartree": reference_info["local_scf_energy_hartree"] - case["dataset_scf_energy_hartree"],
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "dataset_casscf_converged": case["dataset_casscf_converged"],
            "dataset_casscf_natorb": case["dataset_casscf_natorb"],
            "local_casscf_natorb": False,
            "local_minus_dataset_casscf_hartree": reference_info["local_minus_dataset_casscf_hartree"],
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
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_cache_path": reference_info["hamiltonian_cache_path"],
            "hamiltonian_loaded_from_cache": reference_info["hamiltonian_loaded_from_cache"],
            "hamiltonian_source": "casscf",
            "qubits": int(reference_info["qubits"]),
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
            "reference_source": "local sfX2C1eROHF CASSCF Hamiltonian reference without natural orbitals",
            "dataset_scf_method": case["dataset_scf_method"],
            "dataset_scf_energy_hartree": case["dataset_scf_energy_hartree"],
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "dataset_casscf_natorb": case["dataset_casscf_natorb"],
            "local_casscf_natorb": False,
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "active_orbital_labels": active_orbital_labels,
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "epsilon": float(args.epsilon),
            "n_steps_requested": int(args.n_steps),
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": "casscf",
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL qkud {case_id}: {exc!r}", flush=True)
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, default=DEFAULT_CASE_JSON)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument("--scf-conv-tol", type=float, default=None)
    parser.add_argument("--scf-diis-space", type=int, default=12)
    parser.add_argument("--scf-level-shift", type=float, default=0.5)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--no-scf-newton-if-not-converged", dest="scf_newton_if_not_converged", action="store_false")
    parser.set_defaults(scf_newton_if_not_converged=True)
    parser.add_argument("--casscf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--casscf-max-cycle", type=int, default=300)
    parser.add_argument("--casscf-match-tolerance", type=float, default=DEFAULT_CASSCF_MATCH_TOLERANCE_HARTREE)
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--no-avas-labels", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)

    case = _load_case(args.case_json)
    QCANT, qulacs_accel = _import_qcant(args.qcant_root)
    status = _run_case(case, args, QCANT, qulacs_accel)
    print(json.dumps({status: 1}, sort_keys=True), flush=True)
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
