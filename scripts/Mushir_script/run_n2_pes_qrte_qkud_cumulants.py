#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Run MolVQE-style N2 PES QRTE/QKUD and two-body cumulant calculations.

This is intentionally separate from the Benchmark-QC runners.  It varies only
the N-N distance while keeping the MolVQE equilibrium N2 settings:
cc-pVTZ, charge 0, spins 0/2, 6 active electrons, 6 active orbitals, and
QCANT's CASSCF natural-orbital Hamiltonian path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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

from compute_2body_cumulant_norms import cumulant_2rdm_norm_from_cache  # noqa: E402


ANGSTROM_PER_BOHR = 0.529177210903
CHEMICAL_ACCURACY_HARTREE = 1.6e-3
BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb_n2_pes"
DEFAULT_QRTE_OUTDIR = BENCHMARK_ROOT / "runs" / "n2_pes_qrte_natorb_dt0.05"
DEFAULT_QKUD_OUTDIR = BENCHMARK_ROOT / "runs" / "n2_pes_qkud_natorb_eps0.05"
DEFAULT_CUMULANT_OUTDIR = BENCHMARK_ROOT / "runs" / "n2_pes_2body_cumulants"
DEFAULT_SUMMARY_CSV = BENCHMARK_ROOT / "runs" / "n2_pes_qrte_qkud_cumulants.csv"
DEFAULT_DISTANCES_ANGSTROM = (0.90, 1.00, 1.10, 1.35, 1.50, 1.70, 2.00, 2.40)
DEFAULT_SPINS = (0, 2)


@dataclass(frozen=True)
class Case:
    case_id: str
    bond_length_angstrom: float
    spin: int

    @property
    def symbols(self) -> list[str]:
        return ["N", "N"]

    @property
    def geometry_angstrom(self) -> list[list[float]]:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, float(self.bond_length_angstrom)]]

    @property
    def geometry_bohr(self) -> list[list[float]]:
        return [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, float(self.bond_length_angstrom) / ANGSTROM_PER_BOHR],
        ]


def _set_thread_limits() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(path_text: str | os.PathLike[str] | None) -> Path | None:
    if path_text in (None, ""):
        return None
    path = Path(str(path_text))
    if not path.is_absolute():
        path = BENCHMARK_ROOT / path
    return path


def _relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path)


def _completed(summary_path: Path, method: str | None = None) -> bool:
    if not summary_path.exists():
        return False
    try:
        payload = _load_json(summary_path)
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "completed":
        return False
    return method is None or payload.get("method") == method


def _bond_label(distance: float) -> str:
    return f"{float(distance):.2f}".replace(".", "p")


def _case_id(distance: float, spin: int) -> str:
    return f"N2_R{_bond_label(distance)}_cc_pVTZ_{int(spin)}"


def _parse_cases(args: argparse.Namespace) -> list[Case]:
    distances = list(args.distance or DEFAULT_DISTANCES_ANGSTROM)
    spins = list(args.spin or DEFAULT_SPINS)
    cases = [
        Case(
            case_id=_case_id(distance, spin),
            bond_length_angstrom=float(distance),
            spin=int(spin),
        )
        for distance in distances
        if abs(float(distance) - 1.20) > 1e-10
        for spin in spins
    ]
    if args.case_id:
        allowed = set(args.case_id)
        cases = [case for case in cases if case.case_id in allowed]
    cases.sort(key=lambda item: (item.bond_length_angstrom, item.spin))
    if args.limit is not None:
        cases = cases[: int(args.limit)]
    if not cases:
        raise SystemExit("No N2 PES cases selected")
    return cases


def _import_qcant(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))
    import QCANT  # noqa: PLC0415
    import QCANT.qulacs_accel as qulacs_accel  # noqa: PLC0415

    return QCANT, qulacs_accel


def _reference_kwargs(case: Case, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "symbols": case.symbols,
        "geometry": np.asarray(case.geometry_angstrom, dtype=float),
        "basis": "cc-pVTZ",
        "charge": 0,
        "spin": int(case.spin),
        "active_electrons": 6,
        "active_orbitals": 6,
        "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
        "hamiltonian_source": "casscf_natorb",
        "active_orbital_labels": None,
        "active_orbital_indices": None,
        "active_orbital_index_base": 0,
        "allow_unconverged_casscf": False,
        "geometry_unit": "Angstrom",
        "molecule_symmetry": False,
        "scf_conv_tol": None if args.scf_conv_tol is None else float(args.scf_conv_tol),
        "scf_diis_space": int(args.scf_diis_space),
        "scf_level_shift": float(args.scf_level_shift),
        "scf_max_cycle": int(args.scf_max_cycle),
        "scf_newton_if_not_converged": bool(args.scf_newton_if_not_converged),
        "casscf_conv_tol": float(args.casscf_conv_tol),
        "casscf_max_cycle": int(args.casscf_max_cycle),
    }


def _build_local_reference(case: Case, args: argparse.Namespace, qulacs_accel) -> dict[str, Any]:
    payload = qulacs_accel._build_hamiltonian_payload(
        hamiltonian_cutoff=1e-20,
        **_reference_kwargs(case, args),
    )
    if not bool(payload["reference_converged"]):
        raise RuntimeError(f"{case.case_id}: CASSCF/natorb reference did not converge")
    return {
        "local_reference_energy_hartree": float(payload["reference_energy"]),
        "local_reference_converged": bool(payload["reference_converged"]),
        "hamiltonian_cache_path": payload.get("hamiltonian_cache_path"),
        "hamiltonian_loaded_from_cache": bool(payload.get("hamiltonian_loaded_from_cache", False)),
        "active_space_selection": payload.get("active_space_selection"),
        "effective_active_electrons": int(payload.get("active_electrons", 6)),
        "scf_energy_hartree": float(payload.get("scf_energy", "nan")),
        "scf_converged": payload.get("scf_converged"),
    }


def _first_hit(min_history: np.ndarray, reference_energy: float) -> int | None:
    for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
        if float(energy) - float(reference_energy) <= CHEMICAL_ACCURACY_HARTREE:
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
            lineterminator="\n",
        )
        writer.writeheader()
        for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
            error = float(energy) - float(reference_energy)
            writer.writerow(
                {
                    "iteration": idx,
                    "min_energy_hartree": float(energy),
                    "error_vs_local_reference_hartree": error,
                    "within_chemical_accuracy": str(error <= CHEMICAL_ACCURACY_HARTREE).lower(),
                }
            )


def _steps_for(args: argparse.Namespace, method: str) -> int:
    if method == "qrte" and args.qrte_steps is not None:
        return int(args.qrte_steps)
    if method == "qkud" and args.qkud_steps is not None:
        return int(args.qkud_steps)
    return int(args.n_steps)


def _case_snapshot(
    case: Case,
    reference_info: dict[str, Any],
    *,
    reference_source: str,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "source_name": "generated N2 PES grid from MolVQE equilibrium N2 settings",
        "source_json_path": None,
        "symbols": case.symbols,
        "geometry_bohr": case.geometry_bohr,
        "geometry_angstrom": case.geometry_angstrom,
        "bond_length_angstrom": float(case.bond_length_angstrom),
        "basis": "cc-pVTZ",
        "charge": 0,
        "spin": int(case.spin),
        "active_electrons": 6,
        "active_orbitals": 6,
        "active_orbital_labels": None,
        "active_orbital_indices": None,
        "active_orbital_index_base": None,
        "active_orbital_selection_method": None,
        "active_space_source": "MolVQE N2 equilibrium active size: 6e/6o; varied N-N distance",
        "casscf_energy_hartree": float(reference_info["local_reference_energy_hartree"]),
        "reference_source": reference_source,
        "scf_policy": "QCANT/PySCF RHF for spin=0, ROHF for spin>0",
        **reference_info,
    }


def _base_run_config(
    args: argparse.Namespace,
    *,
    method: str,
    n_steps: int,
    early_stop_energy: float | None,
) -> dict[str, Any]:
    config = {
        "active_orbital_labels": None,
        "active_orbital_indices": None,
        "active_orbital_index_base": None,
        "active_orbital_selection_method": None,
        "basis": "cc-pVTZ",
        "active_electrons": 6,
        "active_orbitals": 6,
        "casscf_conv_tol": float(args.casscf_conv_tol),
        "casscf_max_cycle": int(args.casscf_max_cycle),
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
        "early_stop_energy_hartree": early_stop_energy,
        "evolution_mode": args.evolution_mode,
        "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
        "hamiltonian_source": "casscf_natorb",
        "max_workers": int(args.max_workers),
        "method": method,
        "n_steps": int(n_steps),
        "qcant_root": str(args.qcant_root.resolve()),
        "reference_source": "local CASSCF/natorb Hamiltonian reference",
        "scf_conv_tol": None if args.scf_conv_tol is None else float(args.scf_conv_tol),
        "scf_diis_space": int(args.scf_diis_space),
        "scf_level_shift": float(args.scf_level_shift),
        "scf_max_cycle": int(args.scf_max_cycle),
        "scf_newton_if_not_converged": bool(args.scf_newton_if_not_converged),
        "trotter_steps": int(args.trotter_steps),
    }
    if method == "qrte":
        config["delta_t"] = float(args.delta_t)
    if method == "qkud":
        config["epsilon"] = float(args.epsilon)
    return config


def _run_krylov_case(case: Case, method: str, args: argparse.Namespace, QCANT, qulacs_accel) -> str:
    outdir = args.qrte_outdir if method == "qrte" else args.qkud_outdir
    case_dir = outdir / case.case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path, method) and not args.force:
        print(f"SKIP completed {method} {case.case_id}", flush=True)
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
        n_steps = _steps_for(args, method)
        _write_json(
            case_dir / "input_snapshot.json",
            _case_snapshot(
                case,
                reference_info,
                reference_source="local CASSCF/natorb Hamiltonian reference",
            ),
        )
        _write_json(
            case_dir / "run_config.json",
            _base_run_config(
                args,
                method=method,
                n_steps=n_steps,
                early_stop_energy=early_stop_energy,
            ),
        )

        if method == "qrte":
            energies, _basis_states, _times, min_history = QCANT.qrte_qulacs(
                delta_t=float(args.delta_t),
                n_steps=n_steps,
                return_min_energy_history=True,
                evolution_mode=args.evolution_mode,
                trotter_steps=int(args.trotter_steps),
                max_workers=int(args.max_workers),
                early_stop_energy=early_stop_energy,
                **_reference_kwargs(case, args),
            )
        elif method == "qkud":
            energies, _basis_states, min_history = QCANT.qkud_qulacs(
                epsilon=float(args.epsilon),
                n_steps=n_steps,
                return_min_energy_history=True,
                evolution_mode=args.evolution_mode,
                trotter_steps=int(args.trotter_steps),
                max_workers=int(args.max_workers),
                early_stop_energy=early_stop_energy,
                **_reference_kwargs(case, args),
            )
        else:
            raise ValueError(f"Unexpected method: {method!r}")

        energies = np.asarray(energies, dtype=float)
        min_history = np.asarray(min_history, dtype=float)
        hit_iteration = _first_hit(min_history, local_reference_energy)
        n_steps_completed = int(min_history.size)
        stopped_early = bool(
            args.early_stop_chemical_accuracy
            and hit_iteration is not None
            and n_steps_completed < n_steps
        )
        _write_history_csv(case_dir / "min_energy_history.csv", min_history, local_reference_energy)
        summary = {
            "status": "completed",
            "case_id": case.case_id,
            "method": method,
            "bond_length_angstrom": float(case.bond_length_angstrom),
            "spin": int(case.spin),
            "reference_source": "local CASSCF/natorb Hamiltonian reference",
            "casscf_energy_hartree": local_reference_energy,
            "local_reference_energy_hartree": local_reference_energy,
            **reference_info,
            "active_electrons": 6,
            "active_orbitals": 6,
            "active_orbital_labels": None,
            "basis": "cc-pVTZ",
            "charge": 0,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "hit_iteration": hit_iteration,
            "hit_chemical_accuracy": hit_iteration is not None,
            "n_steps": n_steps_completed,
            "n_steps_requested": n_steps,
            "n_steps_completed": n_steps_completed,
            "stopped_early": stopped_early,
            "early_stop_energy_hartree": early_stop_energy,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": "casscf_natorb",
            "delta_t": float(args.delta_t) if method == "qrte" else None,
            "epsilon": float(args.epsilon) if method == "qkud" else None,
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(
            f"DONE {method} {case.case_id} R={case.bond_length_angstrom:.2f} "
            f"spin={case.spin} hit={hit_iteration}",
            flush=True,
        )
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case.case_id,
            "method": method,
            "bond_length_angstrom": float(case.bond_length_angstrom),
            "spin": int(case.spin),
            "reference_source": "local CASSCF/natorb Hamiltonian reference",
            **reference_info,
            "active_electrons": 6,
            "active_orbitals": 6,
            "basis": "cc-pVTZ",
            "charge": 0,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "n_steps_requested": _steps_for(args, method),
            "hamiltonian_cache_dir": str(args.hamiltonian_cache_dir),
            "hamiltonian_source": "casscf_natorb",
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL {method} {case.case_id}: {exc!r}", flush=True)
        if not args.keep_going:
            raise
        return "failed"


def _read_hit(outdir: Path, case_id: str) -> tuple[float | str, bool | str, str]:
    summary_path = outdir / case_id / "summary.json"
    if not summary_path.exists():
        return "", "", ""
    try:
        payload = _load_json(summary_path)
    except json.JSONDecodeError:
        return "", "", "unreadable"
    return (
        payload.get("hit_iteration", ""),
        payload.get("hit_chemical_accuracy", ""),
        str(payload.get("status", "")),
    )


def _run_cumulant_case(case: Case, args: argparse.Namespace, qulacs_accel) -> str:
    case_dir = args.cumulant_outdir / case.case_id
    output_path = case_dir / "cumulant_2body.json"
    if _completed(output_path) and not args.force:
        print(f"SKIP completed cumulant {case.case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reference_info: dict[str, Any] = {}
    try:
        reference_info = _build_local_reference(case, args, qulacs_accel)
        cache_path = _resolve_path(reference_info.get("hamiltonian_cache_path"))
        if cache_path is None:
            raise RuntimeError(f"{case.case_id}: missing Hamiltonian cache path")
        qrte_iter, qrte_hit, qrte_status = _read_hit(args.qrte_outdir, case.case_id)
        qkud_iter, qkud_hit, qkud_status = _read_hit(args.qkud_outdir, case.case_id)
        result = cumulant_2rdm_norm_from_cache(
            cache_path,
            conv_tol=float(args.cumulant_conv_tol),
            max_cycle=int(args.cumulant_max_cycle),
        )
        result.update(
            {
                "case_id": case.case_id,
                "bond_length_angstrom": float(case.bond_length_angstrom),
                "spin": int(case.spin),
                "active_electrons": 6,
                "active_orbitals": 6,
                "basis": "cc-pVTZ",
                "charge": 0,
                "reference_source": "local CASSCF/natorb Hamiltonian reference",
                "local_reference_energy_hartree": float(reference_info["local_reference_energy_hartree"]),
                "qrte_dt0p05_chem_accuracy_iteration": qrte_iter,
                "qrte_hit_chemical_accuracy": qrte_hit,
                "qrte_status": qrte_status,
                "qkud_eps0p05_chem_accuracy_iteration": qkud_iter,
                "qkud_hit_chemical_accuracy": qkud_hit,
                "qkud_status": qkud_status,
                "hamiltonian_cache_path": _relative(cache_path),
                "wall_elapsed_seconds": float(time.perf_counter() - start),
            }
        )
        _write_json(output_path, result)
        print(
            f"DONE cumulant {case.case_id} "
            f"||Delta_2||_F^2={result['cumulant_2rdm_frobenius_norm_squared']:.12g}",
            flush=True,
        )
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case.case_id,
            "bond_length_angstrom": float(case.bond_length_angstrom),
            "spin": int(case.spin),
            "reference_source": "local CASSCF/natorb Hamiltonian reference",
            **reference_info,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "wall_elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(output_path, failure)
        print(f"FAIL cumulant {case.case_id}: {exc!r}", flush=True)
        if not args.keep_going:
            raise
        return "failed"


def _run_reference_case(case: Case, args: argparse.Namespace, qulacs_accel) -> str:
    case_dir = args.reference_outdir / case.case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path, "reference_only") and not args.force:
        print(f"SKIP completed reference {case.case_id}", flush=True)
        return "skipped"
    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        reference_info = _build_local_reference(case, args, qulacs_accel)
        _write_json(
            case_dir / "input_snapshot.json",
            _case_snapshot(
                case,
                reference_info,
                reference_source="local CASSCF/natorb Hamiltonian reference",
            ),
        )
        _write_json(
            case_dir / "summary.json",
            {
                "status": "completed",
                "method": "reference_only",
                "case_id": case.case_id,
                "bond_length_angstrom": float(case.bond_length_angstrom),
                "spin": int(case.spin),
                "active_electrons": 6,
                "active_orbitals": 6,
                "basis": "cc-pVTZ",
                "charge": 0,
                "hamiltonian_source": "casscf_natorb",
                **reference_info,
                "elapsed_seconds": float(time.perf_counter() - start),
            },
        )
        print(
            f"REFERENCE {case.case_id} ref={reference_info['local_reference_energy_hartree']:.12f}",
            flush=True,
        )
        return "completed"
    except Exception as exc:  # noqa: BLE001
        _write_json(
            summary_path,
            {
                "status": "failed",
                "method": "reference_only",
                "case_id": case.case_id,
                "bond_length_angstrom": float(case.bond_length_angstrom),
                "spin": int(case.spin),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": float(time.perf_counter() - start),
            },
        )
        print(f"FAIL reference {case.case_id}: {exc!r}", flush=True)
        if not args.keep_going:
            raise
        return "failed"


def _summary_value(path: Path, key: str) -> Any:
    if not path.exists():
        return ""
    try:
        return _load_json(path).get(key, "")
    except json.JSONDecodeError:
        return ""


def _write_summary_csv(cases: Iterable[Case], args: argparse.Namespace) -> None:
    fieldnames = [
        "case_id",
        "bond_length_angstrom",
        "spin",
        "basis",
        "active_electrons",
        "active_orbitals",
        "qrte_status",
        "qrte_hit_iteration",
        "qrte_hit_chemical_accuracy",
        "qrte_steps_completed",
        "qkud_status",
        "qkud_hit_iteration",
        "qkud_hit_chemical_accuracy",
        "qkud_steps_completed",
        "cumulant_status",
        "cumulant_2rdm_frobenius_norm_squared",
        "cumulant_2rdm_frobenius_norm",
        "local_reference_energy_hartree",
        "hamiltonian_cache_path",
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        qrte_summary = args.qrte_outdir / case.case_id / "summary.json"
        qkud_summary = args.qkud_outdir / case.case_id / "summary.json"
        cumulant_json = args.cumulant_outdir / case.case_id / "cumulant_2body.json"
        local_reference = (
            _summary_value(qrte_summary, "local_reference_energy_hartree")
            or _summary_value(qkud_summary, "local_reference_energy_hartree")
            or _summary_value(cumulant_json, "local_reference_energy_hartree")
        )
        cache_path = (
            _summary_value(qrte_summary, "hamiltonian_cache_path")
            or _summary_value(qkud_summary, "hamiltonian_cache_path")
            or _summary_value(cumulant_json, "hamiltonian_cache_path")
        )
        rows.append(
            {
                "case_id": case.case_id,
                "bond_length_angstrom": float(case.bond_length_angstrom),
                "spin": int(case.spin),
                "basis": "cc-pVTZ",
                "active_electrons": 6,
                "active_orbitals": 6,
                "qrte_status": _summary_value(qrte_summary, "status"),
                "qrte_hit_iteration": _summary_value(qrte_summary, "hit_iteration"),
                "qrte_hit_chemical_accuracy": _summary_value(qrte_summary, "hit_chemical_accuracy"),
                "qrte_steps_completed": _summary_value(qrte_summary, "n_steps_completed"),
                "qkud_status": _summary_value(qkud_summary, "status"),
                "qkud_hit_iteration": _summary_value(qkud_summary, "hit_iteration"),
                "qkud_hit_chemical_accuracy": _summary_value(qkud_summary, "hit_chemical_accuracy"),
                "qkud_steps_completed": _summary_value(qkud_summary, "n_steps_completed"),
                "cumulant_status": _summary_value(cumulant_json, "status"),
                "cumulant_2rdm_frobenius_norm_squared": _summary_value(
                    cumulant_json, "cumulant_2rdm_frobenius_norm_squared"
                ),
                "cumulant_2rdm_frobenius_norm": _summary_value(
                    cumulant_json, "cumulant_2rdm_frobenius_norm"
                ),
                "local_reference_energy_hartree": local_reference,
                "hamiltonian_cache_path": _relative(_resolve_path(cache_path)) if cache_path else "",
            }
        )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.summary_csv.with_suffix(args.summary_csv.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(args.summary_csv)
    print(f"Wrote {args.summary_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "reference", "qrte", "qkud", "cumulant", "summary"),
        default="all",
    )
    parser.add_argument("--distance", action="append", type=float, default=None)
    parser.add_argument("--spin", action="append", type=int, choices=DEFAULT_SPINS, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--qrte-outdir", type=Path, default=DEFAULT_QRTE_OUTDIR)
    parser.add_argument("--qkud-outdir", type=Path, default=DEFAULT_QKUD_OUTDIR)
    parser.add_argument("--cumulant-outdir", type=Path, default=DEFAULT_CUMULANT_OUTDIR)
    parser.add_argument(
        "--reference-outdir",
        type=Path,
        default=BENCHMARK_ROOT / "runs" / "n2_pes_casscf_natorb_references",
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--qrte-steps", type=int, default=None)
    parser.add_argument("--qkud-steps", type=int, default=None)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--casscf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--casscf-max-cycle", type=int, default=100)
    parser.add_argument("--scf-conv-tol", type=float, default=None)
    parser.add_argument("--scf-diis-space", type=int, default=12)
    parser.add_argument("--scf-level-shift", type=float, default=0.5)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument(
        "--no-scf-newton-if-not-converged",
        dest="scf_newton_if_not_converged",
        action="store_false",
    )
    parser.set_defaults(scf_newton_if_not_converged=True)
    parser.add_argument("--cumulant-conv-tol", type=float, default=1e-12)
    parser.add_argument("--cumulant-max-cycle", type=int, default=300)
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)
    args.qrte_outdir.mkdir(parents=True, exist_ok=True)
    args.qkud_outdir.mkdir(parents=True, exist_ok=True)
    args.cumulant_outdir.mkdir(parents=True, exist_ok=True)
    args.reference_outdir.mkdir(parents=True, exist_ok=True)

    cases = _parse_cases(args)
    if args.mode == "summary":
        _write_summary_csv(cases, args)
        return

    QCANT, qulacs_accel = _import_qcant(args.qcant_root)
    counts: dict[str, dict[str, int]] = {}

    def record(kind: str, status: str) -> None:
        bucket = counts.setdefault(kind, {"completed": 0, "failed": 0, "skipped": 0})
        bucket[status] = bucket.get(status, 0) + 1

    if args.mode in {"all", "reference"}:
        for case in cases:
            record("reference", _run_reference_case(case, args, qulacs_accel))
    if args.mode in {"all", "qrte"}:
        for case in cases:
            record("qrte", _run_krylov_case(case, "qrte", args, QCANT, qulacs_accel))
    if args.mode in {"all", "qkud"}:
        for case in cases:
            record("qkud", _run_krylov_case(case, "qkud", args, QCANT, qulacs_accel))
    if args.mode in {"all", "cumulant"}:
        for case in cases:
            record("cumulant", _run_cumulant_case(case, args, qulacs_accel))

    _write_summary_csv(cases, args)
    print(json.dumps(counts, sort_keys=True), flush=True)
    if any(bucket.get("failed", 0) for bucket in counts.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
