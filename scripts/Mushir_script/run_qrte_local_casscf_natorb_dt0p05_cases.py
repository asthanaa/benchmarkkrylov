#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Run selected QRTE cases against the local CASSCF/natorb reference."""

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
BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "qrte_mbenzyne_local_casscf_natorb_dt0.05"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb_mbenzyne_localref"
DEFAULT_CASE_IDS = ("m_Benzyne_equi_cc_pVTZ_0", "m_Benzyne_equi_cc_pVTZ_2")
ACTIVE_ORBITAL_LABELS_BY_CASE = {
    "m_Benzyne_equi_cc_pVTZ_0": ["C 2pz"],
    "m_Benzyne_equi_cc_pVTZ_2": ["C 2pz"],
}


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


def _load_source_case(case_id: str, manifest_row: dict[str, str]) -> dict[str, Any]:
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
        "active_orbitals": int(casscf["ncas"]),
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
                "error_vs_casscf_hartree",
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
                    "error_vs_casscf_hartree": error,
                    "error_vs_local_reference_hartree": error,
                    "within_chemical_accuracy": str(error <= CHEMICAL_ACCURACY_HARTREE).lower(),
                }
            )


def _completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def _import_qcant(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))
    import QCANT
    import QCANT.qulacs_accel as qulacs_accel

    return QCANT, qulacs_accel


def _active_orbital_labels(case_id: str) -> list[str] | None:
    labels = ACTIVE_ORBITAL_LABELS_BY_CASE.get(case_id)
    return None if labels is None else list(labels)


def _build_local_reference(case: dict[str, Any], args: argparse.Namespace, qulacs_accel, active_orbital_labels):
    payload = qulacs_accel._build_hamiltonian_payload(
        symbols=case["symbols"],
        geometry=np.asarray(case["geometry_angstrom"], dtype=float),
        basis=case["basis"],
        charge=int(case["charge"]),
        spin=int(case["spin"]),
        active_electrons=int(case["active_electrons"]),
        active_orbitals=int(case["active_orbitals"]),
        hamiltonian_cutoff=1e-20,
        hamiltonian_source="casscf_natorb",
        hamiltonian_cache_dir=str(args.hamiltonian_cache_dir),
        active_orbital_labels=active_orbital_labels,
    )
    return {
        "local_reference_energy_hartree": float(payload["reference_energy"]),
        "local_reference_converged": bool(payload["reference_converged"]),
        "hamiltonian_cache_path": payload.get("hamiltonian_cache_path"),
        "hamiltonian_loaded_from_cache": bool(payload.get("hamiltonian_loaded_from_cache", False)),
    }


def _run_case(case: dict[str, Any], args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    case_id = case["case_id"]
    case_dir = args.outdir / case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed qrte {case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    active_orbital_labels = _active_orbital_labels(case_id)
    start = time.perf_counter()
    try:
        reference_info = _build_local_reference(case, args, qulacs_accel, active_orbital_labels)
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
            "casscf_energy_hartree": local_reference_energy,
            "local_minus_dataset_casscf_hartree": local_reference_energy - case["dataset_casscf_energy_hartree"],
            "reference_source": "local CASSCF/natorb Hamiltonian reference",
            "scf_policy": "QCANT default: RHF for spin=0, ROHF for spin>0",
        }
        run_config = {
            "active_orbital_labels": active_orbital_labels,
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
            "reference_source": "local CASSCF/natorb Hamiltonian reference",
            "scf_policy": "QCANT default: RHF for spin=0, ROHF for spin>0",
            "trotter_steps": int(args.trotter_steps),
        }
        _write_json(case_dir / "input_snapshot.json", input_snapshot)
        _write_json(case_dir / "run_config.json", run_config)

        energies, _basis_states, _times, min_history = QCANT.qrte_qulacs(
            symbols=case["symbols"],
            geometry=np.asarray(case["geometry_angstrom"], dtype=float),
            delta_t=float(args.delta_t),
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
            hamiltonian_cache_dir=str(args.hamiltonian_cache_dir),
            hamiltonian_source="casscf_natorb",
            active_orbital_labels=active_orbital_labels,
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
            "reference_source": "local CASSCF/natorb Hamiltonian reference",
            "casscf_energy_hartree": local_reference_energy,
            "local_reference_energy_hartree": local_reference_energy,
            "local_reference_converged": reference_info["local_reference_converged"],
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "dataset_casscf_converged": case["dataset_casscf_converged"],
            "local_minus_dataset_casscf_hartree": local_reference_energy - case["dataset_casscf_energy_hartree"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
            "active_orbital_labels": active_orbital_labels,
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "scf_policy": "QCANT default: RHF for spin=0, ROHF for spin>0",
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
            "hamiltonian_cache_path": reference_info["hamiltonian_cache_path"],
            "hamiltonian_loaded_from_cache": reference_info["hamiltonian_loaded_from_cache"],
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
            "method": "qrte",
            "reference_source": "local CASSCF/natorb Hamiltonian reference",
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
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
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)

    selected_case_ids: Sequence[str] = tuple(args.case_id or DEFAULT_CASE_IDS)
    manifest_by_case = _load_manifest(args.manifest)
    missing = [case_id for case_id in selected_case_ids if case_id not in manifest_by_case]
    if missing:
        raise SystemExit(f"Case id(s) not found in manifest: {', '.join(missing)}")

    QCANT, qulacs_accel = _import_qcant(args.qcant_root)
    cases = [_load_source_case(case_id, manifest_by_case[case_id]) for case_id in selected_case_ids]
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    for case in cases:
        status = _run_case(case, args, QCANT, qulacs_accel)
        counts[status] = counts.get(status, 0) + 1

    print(json.dumps(counts, sort_keys=True), flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
