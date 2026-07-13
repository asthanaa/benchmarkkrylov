#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Compute exact-sector two-body cumulant norms for Benchmark-QC cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
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

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / f"matplotlib-{os.environ.get('USER', 'benchmarkkrylov')}"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(TMPDIR / f"python-pycache-{os.environ.get('USER', 'benchmarkkrylov')}"))

import numpy as np

from run_benchmark_qc_krylov import (  # noqa: E402
    BENCHMARK_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_QCANT_ROOT,
    _build_payload,
    _import_stack,
    _load_manifest,
    _load_raw_case,
)


DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "benchmark_qc_2body_cumulants"

FIELDNAMES = [
    "case_id",
    "status",
    "system",
    "bond_length_angstrom",
    "active_electrons",
    "active_orbitals",
    "spin",
    "n_qubits",
    "n_alpha",
    "n_beta",
    "sector_dim",
    "term_count",
    "solver",
    "sector_ground_energy_hartree",
    "reference_energy_hartree",
    "ground_minus_reference_hartree",
    "ground_gap_hartree",
    "cumulant_2rdm_frobenius_norm_squared",
    "cumulant_2rdm_frobenius_norm",
    "gamma_trace",
    "two_rdm_trace",
    "max_rdm_consistency_error",
    "spin_orbital_order",
    "raw_npz",
    "equilibrium_npz",
    "elapsed_seconds",
    "error",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
    tmp_path.replace(path)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path)


def _bits_to_index(bits: list[int]) -> int:
    return int("".join(str(int(bit)) for bit in bits), 2)


def _index_to_bits(index: int, qubits: int) -> list[int]:
    return [int(bit) for bit in format(int(index), f"0{int(qubits)}b")]


def _spin_populations(active_electrons: int, spin: int) -> tuple[int, int]:
    total = int(active_electrons)
    spin_delta = int(spin)
    if (total + spin_delta) % 2 != 0:
        raise ValueError(f"active_electrons={total} and spin={spin_delta} have incompatible parity")
    n_alpha = (total + spin_delta) // 2
    n_beta = total - n_alpha
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(f"invalid spin sector for active_electrons={total}, spin={spin_delta}")
    return n_alpha, n_beta


def _spin_wires(qubits: int, order: str) -> tuple[list[int], list[int]]:
    if qubits % 2 != 0:
        raise ValueError(f"Expected an even number of spin orbitals, got {qubits}")
    n_spatial = qubits // 2
    if order == "interleaved":
        return list(range(0, qubits, 2)), list(range(1, qubits, 2))
    if order == "alpha-first":
        return list(range(n_spatial)), list(range(n_spatial, qubits))
    raise ValueError(f"Unknown spin-orbital order: {order}")


def _sector_indices(
    *,
    qubits: int,
    n_alpha: int,
    n_beta: int,
    spin_orbital_order: str,
) -> np.ndarray:
    alpha_wires, beta_wires = _spin_wires(qubits, spin_orbital_order)
    indices: list[int] = []
    for index in range(2**int(qubits)):
        bits = _index_to_bits(index, qubits)
        if sum(bits[wire] for wire in alpha_wires) != int(n_alpha):
            continue
        if sum(bits[wire] for wire in beta_wires) != int(n_beta):
            continue
        indices.append(index)
    return np.asarray(indices, dtype=int)


def _apply_fermion_ops(
    index: int,
    ops_right_to_left: list[tuple[str, int]],
    *,
    qubits: int,
) -> tuple[int, int] | None:
    bits = _index_to_bits(index, qubits)
    phase = 1
    for action, wire in ops_right_to_left:
        parity = sum(bits[: int(wire)])
        if parity % 2:
            phase *= -1
        if action == "annihilate":
            if bits[wire] == 0:
                return None
            bits[wire] = 0
        elif action == "create":
            if bits[wire] == 1:
                return None
            bits[wire] = 1
        else:
            raise ValueError(f"Unknown fermion action: {action}")
    return _bits_to_index(bits), phase


def _expectation(
    psi: np.ndarray,
    occupied_indices: np.ndarray,
    ops_right_to_left: list[tuple[str, int]],
    *,
    qubits: int,
) -> complex:
    value = 0.0j
    for index in occupied_indices:
        coeff = psi[int(index)]
        if abs(coeff) < 1e-14:
            continue
        transition = _apply_fermion_ops(int(index), ops_right_to_left, qubits=qubits)
        if transition is None:
            continue
        new_index, phase = transition
        value += np.conj(psi[new_index]) * phase * coeff
    return value


def _spinorbital_rdms_from_state(
    psi: np.ndarray,
    occupied_indices: np.ndarray,
    *,
    qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.zeros((qubits, qubits), dtype=complex)
    two_rdm = np.zeros((qubits, qubits, qubits, qubits), dtype=complex)

    for p in range(qubits):
        for q in range(qubits):
            gamma[p, q] = _expectation(
                psi,
                occupied_indices,
                [("annihilate", q), ("create", p)],
                qubits=qubits,
            )

    for p in range(qubits):
        for q in range(qubits):
            for r in range(qubits):
                for s in range(qubits):
                    two_rdm[p, q, r, s] = _expectation(
                        psi,
                        occupied_indices,
                        [
                            ("annihilate", r),
                            ("annihilate", s),
                            ("create", q),
                            ("create", p),
                        ],
                        qubits=qubits,
                    )

    return gamma, two_rdm


def _cumulant_metrics(
    psi: np.ndarray,
    sector_indices: np.ndarray,
    *,
    qubits: int,
    active_electrons: int,
) -> dict[str, float]:
    gamma, two_rdm = _spinorbital_rdms_from_state(psi, sector_indices, qubits=qubits)
    disconnected = (
        np.einsum("pr,qs->pqrs", gamma, gamma)
        - np.einsum("ps,qr->pqrs", gamma, gamma)
    )
    cumulant = two_rdm - disconnected
    norm2 = float(np.real(np.vdot(cumulant, cumulant)))
    gamma_trace = float(np.real(np.trace(gamma)))
    two_trace = float(np.real(np.einsum("pqpq->", two_rdm)))
    max_consistency_error = max(
        abs(gamma_trace - int(active_electrons)),
        abs(two_trace - int(active_electrons) * (int(active_electrons) - 1)),
        float(np.max(np.abs(two_rdm + two_rdm.swapaxes(0, 1)))),
        float(np.max(np.abs(two_rdm + two_rdm.swapaxes(2, 3)))),
    )
    return {
        "cumulant_2rdm_frobenius_norm_squared": norm2,
        "cumulant_2rdm_frobenius_norm": float(math.sqrt(max(norm2, 0.0))),
        "gamma_trace": gamma_trace,
        "two_rdm_trace": two_trace,
        "max_rdm_consistency_error": float(max_consistency_error),
    }


def _read_completed(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return None
    if payload.get("status") == "completed":
        return payload
    return None


def _exact_sector_ground_state(
    case: Any,
    args: argparse.Namespace,
    stack: dict[str, Any],
) -> tuple[float, float | None, np.ndarray, np.ndarray, int]:
    raw_payload = _load_raw_case(case, stack)
    payload = _build_payload(case, raw_payload, stack)
    qubits = int(case.n_qubits)
    n_alpha, n_beta = _spin_populations(case.active_electrons, case.spin)
    sector_indices = _sector_indices(
        qubits=qubits,
        n_alpha=n_alpha,
        n_beta=n_beta,
        spin_orbital_order=args.spin_orbital_order,
    )
    if sector_indices.size == 0:
        raise ValueError(f"No determinants found for {case.case_id}")

    h_matrix = payload["h_qulacs"].get_matrix().tocsr()
    sector_h = h_matrix[sector_indices][:, sector_indices].toarray()
    sector_h = np.asarray(0.5 * (sector_h + sector_h.conj().T), dtype=complex)
    eigvals, eigvecs = np.linalg.eigh(sector_h)
    order = np.argsort(np.real(eigvals), kind="stable")
    ground_position = int(order[0])
    ground_energy = float(np.real(eigvals[ground_position]))
    ground_gap = None
    if order.size > 1:
        ground_gap = float(np.real(eigvals[int(order[1])] - eigvals[ground_position]))

    psi = np.zeros(2**qubits, dtype=complex)
    psi[sector_indices] = np.asarray(eigvecs[:, ground_position], dtype=complex)
    psi_norm = float(np.linalg.norm(psi))
    if psi_norm < 1e-14:
        raise RuntimeError(f"Exact-sector eigenvector collapsed for {case.case_id}")
    psi /= psi_norm
    return ground_energy, ground_gap, psi, sector_indices, int(payload["term_count"])


def compute_case(case: Any, args: argparse.Namespace, stack: dict[str, Any]) -> dict[str, Any]:
    output_path = args.outdir / case.case_id / "cumulant_2body.json"
    if not args.force:
        completed = _read_completed(output_path)
        if completed is not None:
            print(f"SKIP {case.case_id}: {output_path}", flush=True)
            return completed

    print(f"START {case.case_id}: {case.system}", flush=True)
    start = time.perf_counter()
    n_alpha, n_beta = _spin_populations(case.active_electrons, case.spin)
    try:
        ground_energy, ground_gap, psi, sector_indices, term_count = _exact_sector_ground_state(case, args, stack)
        metrics = _cumulant_metrics(
            psi,
            sector_indices,
            qubits=int(case.n_qubits),
            active_electrons=int(case.active_electrons),
        )
        result = {
            "status": "completed",
            "case_id": case.case_id,
            "system": case.system,
            "bond_length_angstrom": float(case.bond_length_angstrom),
            "active_electrons": int(case.active_electrons),
            "active_orbitals": int(case.active_orbitals),
            "spin": int(case.spin),
            "n_qubits": int(case.n_qubits),
            "n_alpha": int(n_alpha),
            "n_beta": int(n_beta),
            "sector_dim": int(sector_indices.size),
            "term_count": int(term_count),
            "solver": "exact-sector-qubit-diagonalization",
            "sector_ground_energy_hartree": float(ground_energy),
            "reference_energy_hartree": float(case.reference_energy_hartree),
            "ground_minus_reference_hartree": float(ground_energy - float(case.reference_energy_hartree)),
            "ground_gap_hartree": "" if ground_gap is None else float(ground_gap),
            "raw_npz": _relative(case.raw_npz),
            "equilibrium_npz": _relative(case.equilibrium_npz),
            "manifest": _relative(args.manifest),
            "spin_orbital_order": args.spin_orbital_order,
            "elapsed_seconds": float(time.perf_counter() - start),
            **metrics,
        }
        _write_json(output_path, result)
        print(
            f"DONE {case.case_id}: ||Delta_2||_F^2="
            f"{result['cumulant_2rdm_frobenius_norm_squared']:.12g}",
            flush=True,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case.case_id,
            "system": case.system,
            "error": repr(exc),
            "raw_npz": _relative(case.raw_npz),
            "equilibrium_npz": _relative(case.equilibrium_npz),
            "manifest": _relative(args.manifest),
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(output_path, failure)
        print(f"FAIL {case.case_id}: {exc!r}", flush=True)
        if not args.keep_going:
            raise
        return failure


def _run_self_test() -> None:
    qubits = 6
    bits = [1, 1, 0, 0, 0, 0]
    psi = np.zeros(2**qubits, dtype=complex)
    index = _bits_to_index(bits)
    psi[index] = 1.0
    metrics = _cumulant_metrics(psi, np.asarray([index], dtype=int), qubits=qubits, active_electrons=2)
    if metrics["cumulant_2rdm_frobenius_norm_squared"] > 1e-12:
        raise RuntimeError(f"Slater determinant self-test failed: {metrics}")
    print("self-test passed: Slater determinant cumulant norm is zero")


def compute(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = _load_manifest(args.manifest)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
    if not cases:
        raise SystemExit("No Benchmark-QC cases selected")
    if args.expected_cases is not None and len(cases) != int(args.expected_cases):
        raise SystemExit(f"Expected {args.expected_cases} Benchmark-QC cases, found {len(cases)}")

    stack = _import_stack(args.qcant_root)
    rows = [compute_case(case, args, stack) for case in cases]
    rows.sort(key=lambda row: str(row.get("case_id", "")))
    _write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--spin-orbital-order", choices=("interleaved", "alpha-first"), default="interleaved")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest if args.manifest.is_absolute() else BENCHMARK_ROOT / args.manifest
    args.outdir = args.outdir if args.outdir.is_absolute() else BENCHMARK_ROOT / args.outdir
    args.qcant_root = args.qcant_root if args.qcant_root.is_absolute() else BENCHMARK_ROOT / args.qcant_root
    if args.output_csv is None:
        args.output_csv = args.outdir / "two_body_cumulants.csv"
    elif not args.output_csv.is_absolute():
        args.output_csv = BENCHMARK_ROOT / args.output_csv
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        _run_self_test()
        if not args.case_id and args.expected_cases is None and not args.force:
            return
    compute(args)


if __name__ == "__main__":
    main()
