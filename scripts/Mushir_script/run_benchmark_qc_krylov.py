#!/usr/bin/env python
"""Run QRTE/QKUD on saved Benchmark-QC equilibrium Hamiltonians.

This runner is intentionally self-contained inside benchmarkkrylov. It consumes
Benchmark-QC-style NPZ files with labels/Hs/casci_energies, extracts the chosen
bond-length Hamiltonian, converts the saved PennyLane Pauli terms to a Qulacs
operator, starts from the manifest reference determinant, and writes run
summaries in the same broad shape as the existing Mushir QRTE/QKUD runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
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

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "benchmark_qc_equilibrium_manifest.csv"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


@dataclass(frozen=True)
class Case:
    case_id: str
    system: str
    raw_npz: Path
    equilibrium_npz: Path
    bond_length_angstrom: float
    active_electrons: int
    active_orbitals: int
    spin: int
    n_qubits: int
    reference_bits: list[int]
    reference_energy_hartree: float | None
    label_index: int | None = None


def _set_thread_limits() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BENCHMARK_ROOT / path


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _load_manifest(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference_bits = [int(bit) for bit in json.loads(row["reference_bits"])]
            cases.append(
                Case(
                    case_id=row["case_id"],
                    system=row["system"],
                    raw_npz=_resolve_repo_path(row["raw_npz"]),
                    equilibrium_npz=_resolve_repo_path(row["equilibrium_npz"]),
                    bond_length_angstrom=float(row["bond_length_angstrom"]),
                    active_electrons=int(row["active_electrons"]),
                    active_orbitals=int(row["active_orbitals"]),
                    spin=int(row["spin"]),
                    n_qubits=int(row["n_qubits"]),
                    reference_bits=reference_bits,
                    reference_energy_hartree=_float_or_none(row.get("reference_energy_hartree")),
                    label_index=_int_or_none(row.get("label_index")),
                )
            )
    return cases


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _write_history_csv(path: Path, min_history, reference_energy: float, np) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "min_energy_hartree",
                "error_vs_reference_hartree",
                "within_chemical_accuracy",
            ],
        )
        writer.writeheader()
        for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
            error = float(energy) - float(reference_energy)
            writer.writerow(
                {
                    "iteration": idx,
                    "min_energy_hartree": float(energy),
                    "error_vs_reference_hartree": error,
                    "within_chemical_accuracy": str(
                        abs(error) <= CHEMICAL_ACCURACY_HARTREE
                    ).lower(),
                }
            )


def _first_hit(min_history, reference_energy: float, np) -> int | None:
    for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
        if abs(float(energy) - float(reference_energy)) <= CHEMICAL_ACCURACY_HARTREE:
            return idx
    return None


def _completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def _import_stack(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))

    # The vendored QCANT source tree may not include the generated setuptools-scm
    # _version.py file. Provide the minimal module __init__ expects when running
    # directly from a copied repository on HPC.
    import types

    version_module = types.ModuleType("QCANT._version")
    version_module.__version__ = "benchmark-qc-local"
    sys.modules.setdefault("QCANT._version", version_module)

    from QCANT.qulacs_accel import (  # noqa: PLC0415
        _ProjectionAccumulator,
        _basis_vector_from_bits,
        _build_time_evolution_circuit,
        _evolve_state,
        _import_optional_stack,
        _project_hamiltonian_energies,
        _qml_operator_to_qulacs_operator,
        _sanitize_hamiltonian,
        _state_from_vector,
        _threshold_state,
    )

    stack = _import_optional_stack()
    stack.update(
        {
            "_ProjectionAccumulator": _ProjectionAccumulator,
            "_basis_vector_from_bits": _basis_vector_from_bits,
            "_build_time_evolution_circuit": _build_time_evolution_circuit,
            "_evolve_state": _evolve_state,
            "_project_hamiltonian_energies": _project_hamiltonian_energies,
            "_qml_operator_to_qulacs_operator": _qml_operator_to_qulacs_operator,
            "_sanitize_hamiltonian": _sanitize_hamiltonian,
            "_state_from_vector": _state_from_vector,
            "_threshold_state": _threshold_state,
        }
    )
    return stack


def _nearest_label_index(labels, target: float, np) -> int:
    labels_f = np.asarray(labels, dtype=float)
    return int(np.argmin(np.abs(labels_f - float(target))))


def _load_raw_case(case: Case, stack: dict[str, Any]) -> dict[str, Any]:
    np = stack["np"]
    qml = stack["qml"]
    if not case.raw_npz.exists():
        raise FileNotFoundError(f"Missing raw Benchmark-QC NPZ: {case.raw_npz}")
    data = np.load(case.raw_npz, allow_pickle=True)
    if case.label_index is None:
        idx = _nearest_label_index(data["labels"], case.bond_length_angstrom, np)
    else:
        idx = int(case.label_index)
        if idx < 0 or idx >= len(data["labels"]):
            raise ValueError(
                f"{case.case_id}: label_index {idx} outside raw label range [0, {len(data['labels'])})"
            )
    chosen_label = float(data["labels"][idx])
    if abs(chosen_label - case.bond_length_angstrom) > 1e-8:
        raise ValueError(
            f"{case.case_id}: requested bond {case.bond_length_angstrom} A but nearest saved label is {chosen_label} A"
        )
    raw_terms = data["Hs"][idx]
    if hasattr(raw_terms, "terms"):
        h_qml = raw_terms
        h_terms_for_cache = np.asarray([raw_terms], dtype=object)
        term_count = len(raw_terms.terms()[0])
    else:
        h_terms_for_cache = np.asarray(raw_terms, dtype=object)
        terms = list(raw_terms)
        if not terms:
            raise ValueError(f"{case.case_id}: saved Hamiltonian has no terms")
        h_qml = qml.sum(*terms)
        term_count = len(terms)
    h_qml = stack["_sanitize_hamiltonian"](h_qml, np, qml)
    reference_energy = float(data["casci_energies"][idx])
    return {
        "h_qml": h_qml,
        "h_terms_for_cache": h_terms_for_cache,
        "label_index": idx,
        "chosen_label": chosen_label,
        "reference_energy_hartree": reference_energy,
        "term_count": int(term_count),
        "source_npz": str(case.raw_npz),
    }


def _write_equilibrium_case(case: Case, raw_payload: dict[str, Any], stack: dict[str, Any]) -> None:
    np = stack["np"]
    case.equilibrium_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case.equilibrium_npz,
        case_id=np.asarray(case.case_id),
        system=np.asarray(case.system),
        source_npz=np.asarray(str(case.raw_npz.relative_to(BENCHMARK_ROOT))),
        label_index=int(raw_payload["label_index"]),
        bond_length_angstrom=float(raw_payload["chosen_label"]),
        reference_energy_hartree=float(raw_payload["reference_energy_hartree"]),
        active_electrons=int(case.active_electrons),
        active_orbitals=int(case.active_orbitals),
        spin=int(case.spin),
        n_qubits=int(case.n_qubits),
        reference_bits=np.asarray(case.reference_bits, dtype=int),
        H=np.asarray(raw_payload["h_terms_for_cache"], dtype=object),
        term_count=int(raw_payload["term_count"]),
        format_version=np.asarray("benchmark_qc_equilibrium_v1"),
    )


def _build_payload(case: Case, raw_payload: dict[str, Any], stack: dict[str, Any]) -> dict[str, Any]:
    np = stack["np"]
    if len(case.reference_bits) != case.n_qubits:
        raise ValueError(
            f"{case.case_id}: reference_bits length {len(case.reference_bits)} != n_qubits {case.n_qubits}"
        )
    if case.n_qubits != 2 * case.active_orbitals:
        raise ValueError(
            f"{case.case_id}: n_qubits {case.n_qubits} != 2 * active_orbitals {case.active_orbitals}"
        )
    if sum(case.reference_bits) != case.active_electrons:
        raise ValueError(
            f"{case.case_id}: reference_bits contain {sum(case.reference_bits)} electrons, expected {case.active_electrons}"
        )

    h_qulacs, pauli_terms = stack["_qml_operator_to_qulacs_operator"](
        raw_payload["h_qml"],
        qubits=case.n_qubits,
        qml=stack["qml"],
        operator_cls=stack["Observable"],
    )
    return {
        "h_qulacs": h_qulacs,
        "pauli_terms": pauli_terms,
        "qubits": int(case.n_qubits),
        "hf_bits": np.asarray(case.reference_bits, dtype=int),
        **raw_payload,
    }


def _run_krylov(case: Case, args: argparse.Namespace, payload: dict[str, Any], stack: dict[str, Any]):
    np = stack["np"]
    QuantumState = stack["QuantumState"]
    QuantumCircuit = stack["QuantumCircuit"]
    qg = stack["qg"]
    expm_multiply = stack["expm_multiply"]
    qubits = int(payload["qubits"])
    h_qulacs = payload["h_qulacs"]

    current = stack["_state_from_vector"](
        stack["_basis_vector_from_bits"](payload["hf_bits"], qubits, np),
        qubits=qubits,
        QuantumState=QuantumState,
        np=np,
    )
    current = stack["_threshold_state"](current, args.basis_threshold, QuantumState, np)
    basis_states = [current.copy()]
    min_history = []
    projection_accumulator = stack["_ProjectionAccumulator"](
        h_qulacs,
        np,
        max_workers=args.max_workers,
        capacity=int(args.n_steps) + 1,
    )
    projection_accumulator.append(current)

    sparse_hamiltonian = h_qulacs.get_matrix().tocsr() if args.evolution_mode == "sparse" else None
    if args.method == "qrte":
        circuit = None
        if args.evolution_mode == "trotter":
            circuit = stack["_build_time_evolution_circuit"](
                qubits,
                payload["pauli_terms"],
                float(args.delta_t),
                int(args.trotter_steps),
                qg,
                QuantumCircuit,
                np,
            )
        for _ in range(int(args.n_steps)):
            current = stack["_evolve_state"](
                current,
                evolution_mode=args.evolution_mode,
                sparse_hamiltonian=sparse_hamiltonian,
                circuit=circuit,
                time_step=float(args.delta_t),
                QuantumState=QuantumState,
                np=np,
                expm_multiply=expm_multiply,
            )
            current = stack["_threshold_state"](current, args.basis_threshold, QuantumState, np)
            basis_states.append(current.copy())
            projection_accumulator.append(current)
            e0 = float(projection_accumulator.energies(args.overlap_tol)[0])
            min_history.append(e0)
            if (
                args.early_stop_chemical_accuracy
                and abs(e0 - float(payload["reference_energy_hartree"]))
                <= CHEMICAL_ACCURACY_HARTREE
            ):
                break
    else:
        forward_circuit = None
        backward_circuit = None
        if args.evolution_mode == "trotter":
            forward_circuit = stack["_build_time_evolution_circuit"](
                qubits,
                payload["pauli_terms"],
                float(args.epsilon),
                int(args.trotter_steps),
                qg,
                QuantumCircuit,
                np,
            )
            backward_circuit = stack["_build_time_evolution_circuit"](
                qubits,
                payload["pauli_terms"],
                -float(args.epsilon),
                int(args.trotter_steps),
                qg,
                QuantumCircuit,
                np,
            )
        for _ in range(int(args.n_steps)):
            forward = stack["_evolve_state"](
                current,
                evolution_mode=args.evolution_mode,
                sparse_hamiltonian=sparse_hamiltonian,
                circuit=forward_circuit,
                time_step=float(args.epsilon),
                QuantumState=QuantumState,
                np=np,
                expm_multiply=expm_multiply,
            )
            backward = stack["_evolve_state"](
                current,
                evolution_mode=args.evolution_mode,
                sparse_hamiltonian=sparse_hamiltonian,
                circuit=backward_circuit,
                time_step=-float(args.epsilon),
                QuantumState=QuantumState,
                np=np,
                expm_multiply=expm_multiply,
            )
            vector = (1j * (np.asarray(forward.get_vector(), dtype=complex) - np.asarray(backward.get_vector(), dtype=complex))) / (2.0 * float(args.epsilon))
            if args.normalize_basis:
                norm = np.linalg.norm(vector)
                if norm < 1e-14:
                    raise ValueError("QKUD vector has zero norm")
                vector = vector / norm
            current = stack["_state_from_vector"](vector, qubits=qubits, QuantumState=QuantumState, np=np)
            current = stack["_threshold_state"](current, args.basis_threshold, QuantumState, np)
            basis_states.append(current.copy())
            projection_accumulator.append(current)
            e0 = float(projection_accumulator.energies(args.overlap_tol)[0])
            min_history.append(e0)
            if (
                args.early_stop_chemical_accuracy
                and abs(e0 - float(payload["reference_energy_hartree"]))
                <= CHEMICAL_ACCURACY_HARTREE
            ):
                break

    energies = projection_accumulator.energies(args.overlap_tol)
    basis_array = projection_accumulator.basis_array()
    return np.asarray(energies, dtype=float), basis_array, np.asarray(min_history, dtype=float)


def _case_dir(outdir: Path, case: Case) -> Path:
    return outdir / case.case_id


def _run_case(case: Case, args: argparse.Namespace, stack: dict[str, Any]) -> str:
    case_dir = _case_dir(args.outdir, case)
    summary_path = case_dir / "summary.json"
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed {args.method} {case.case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        raw_payload = _load_raw_case(case, stack)
        _write_equilibrium_case(case, raw_payload, stack)
        payload = _build_payload(case, raw_payload, stack)
        reference_energy = float(payload["reference_energy_hartree"])
        early_stop_energy = reference_energy + CHEMICAL_ACCURACY_HARTREE if args.early_stop_chemical_accuracy else None

        input_snapshot = {
            "case_id": case.case_id,
            "system": case.system,
            "raw_npz": str(case.raw_npz.relative_to(BENCHMARK_ROOT)),
            "equilibrium_npz": str(case.equilibrium_npz.relative_to(BENCHMARK_ROOT)),
            "source_npz": str(case.raw_npz.relative_to(BENCHMARK_ROOT)),
            "label_index": int(payload["label_index"]),
            "bond_length_angstrom": float(payload["chosen_label"]),
            "active_electrons": int(case.active_electrons),
            "active_orbitals": int(case.active_orbitals),
            "spin": int(case.spin),
            "n_qubits": int(case.n_qubits),
            "reference_bits": [int(bit) for bit in case.reference_bits],
            "reference_energy_hartree": reference_energy,
            "reference_energy_source": "Benchmark-QC casci_energies at selected bond length",
            "hamiltonian_source": "Benchmark-QC saved PennyLane qubit Hamiltonian",
            "term_count": int(payload["term_count"]),
        }
        run_config = {
            "method": args.method,
            "n_steps": int(args.n_steps),
            "delta_t": float(args.delta_t),
            "epsilon": float(args.epsilon),
            "evolution_mode": args.evolution_mode,
            "trotter_steps": int(args.trotter_steps),
            "max_workers": None if args.max_workers is None else int(args.max_workers),
            "overlap_tol": float(args.overlap_tol),
            "basis_threshold": float(args.basis_threshold),
            "normalize_basis": bool(args.normalize_basis),
            "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
            "early_stop_energy_hartree": early_stop_energy,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "qcant_root": str(args.qcant_root.resolve()),
            "manifest": str(args.manifest.resolve()),
        }
        _write_json(case_dir / "input_snapshot.json", input_snapshot)
        _write_json(case_dir / "run_config.json", run_config)

        if args.reference_only:
            summary = {
                "status": "completed",
                "case_id": case.case_id,
                "method": "reference_only",
                **input_snapshot,
                "elapsed_seconds": float(time.perf_counter() - start),
            }
            _write_json(summary_path, summary)
            print(f"REFERENCE {case.case_id} ref={reference_energy:.12f}", flush=True)
            return "completed"

        energies, _basis_array, min_history = _run_krylov(case, args, payload, stack)
        hit_iteration = _first_hit(min_history, reference_energy, stack["np"])
        n_steps_completed = int(min_history.size)
        stopped_early = bool(
            args.early_stop_chemical_accuracy
            and hit_iteration is not None
            and n_steps_completed < int(args.n_steps)
        )
        _write_history_csv(case_dir / "min_energy_history.csv", min_history, reference_energy, stack["np"])
        summary = {
            "status": "completed",
            **input_snapshot,
            "method": args.method,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "delta_t": float(args.delta_t),
            "epsilon": float(args.epsilon),
            "evolution_mode": args.evolution_mode,
            "trotter_steps": int(args.trotter_steps),
            "hit_iteration": hit_iteration,
            "hit_chemical_accuracy": hit_iteration is not None,
            "n_steps": n_steps_completed,
            "n_steps_requested": int(args.n_steps),
            "n_steps_completed": n_steps_completed,
            "stopped_early": stopped_early,
            "early_stop_energy_hartree": early_stop_energy,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(f"DONE {args.method} {case.case_id} hit={hit_iteration} ref={reference_energy:.12f}", flush=True)
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case.case_id,
            "method": args.method,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, failure)
        print(f"FAIL {args.method} {case.case_id}: {exc!r}", flush=True)
        return "failed"


def _run_case_parent(case: Case, args: argparse.Namespace) -> str:
    case_dir = _case_dir(args.outdir, case)
    summary_path = case_dir / "summary.json"
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed {args.method} {case.case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-case",
        "--method",
        args.method,
        "--manifest",
        str(args.manifest),
        "--outdir",
        str(args.outdir),
        "--qcant-root",
        str(args.qcant_root),
        "--case-id",
        case.case_id,
        "--n-steps",
        str(args.n_steps),
        "--delta-t",
        str(args.delta_t),
        "--epsilon",
        str(args.epsilon),
        "--evolution-mode",
        args.evolution_mode,
        "--trotter-steps",
        str(args.trotter_steps),
        "--overlap-tol",
        str(args.overlap_tol),
        "--basis-threshold",
        str(args.basis_threshold),
    ]
    if args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.early_stop_chemical_accuracy:
        cmd.append("--early-stop-chemical-accuracy")
    if args.force:
        cmd.append("--force")
    if args.reference_only:
        cmd.append("--reference-only")
    if not args.normalize_basis:
        cmd.append("--no-normalize-basis")

    with (case_dir / "stdout.log").open("w", encoding="utf-8") as stdout_handle:
        with (case_dir / "stderr.log").open("w", encoding="utf-8") as stderr_handle:
            result = subprocess.run(cmd, cwd=str(BENCHMARK_ROOT), stdout=stdout_handle, stderr=stderr_handle, text=True, check=False)

    if result.returncode == 0 and _completed(summary_path):
        print(f"DONE {args.method} {case.case_id}", flush=True)
        return "completed"
    print(f"FAIL {args.method} {case.case_id}; subprocess rc={result.returncode}; see {summary_path}", flush=True)
    return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qrte", "qkud"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--overlap-tol", type=float, default=1e-10)
    parser.add_argument("--basis-threshold", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Only extract/load references and write snapshots.")
    parser.add_argument("--normalize-basis", dest="normalize_basis", action="store_true", default=True)
    parser.add_argument("--no-normalize-basis", dest="normalize_basis", action="store_false")
    parser.add_argument("--single-case", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.manifest = args.manifest if args.manifest.is_absolute() else BENCHMARK_ROOT / args.manifest
    args.outdir = args.outdir if args.outdir.is_absolute() else BENCHMARK_ROOT / args.outdir
    args.qcant_root = args.qcant_root if args.qcant_root.is_absolute() else BENCHMARK_ROOT / args.qcant_root
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = _load_manifest(args.manifest)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
    if not cases:
        raise SystemExit("No Benchmark-QC cases selected")

    counts = {"completed": 0, "skipped": 0, "failed": 0}
    if args.single_case:
        if len(cases) != 1:
            raise SystemExit("--single-case requires exactly one selected case")
        stack = _import_stack(args.qcant_root)
        status = _run_case(cases[0], args, stack)
        counts[status] = counts.get(status, 0) + 1
        print(json.dumps(counts, sort_keys=True), flush=True)
        if status == "failed":
            raise SystemExit(1)
        return

    for case in cases:
        status = _run_case_parent(case, args)
        counts[status] = counts.get(status, 0) + 1

    print(json.dumps(counts, sort_keys=True), flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
