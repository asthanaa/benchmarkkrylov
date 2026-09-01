#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Compute reference-matched initial overlap and spectral gap for BenchmarkQC."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
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
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(TMPDIR / f"matplotlib-{os.environ.get('USER', 'benchmarkkrylov')}"),
)
os.environ.setdefault(
    "PYTHONPYCACHEPREFIX",
    str(TMPDIR / f"python-pycache-{os.environ.get('USER', 'benchmarkkrylov')}"),
)

import numpy as np


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_QC_REPO = Path("/home/mushir.kolathingal/BenchmarkQC")
DEFAULT_MANIFEST = (
    DEFAULT_BENCHMARK_QC_REPO
    / "datasets"
    / "benchmarkQC"
    / "source"
    / "manifest.csv"
)
DEFAULT_OVERLAP_OUTDIR = BENCHMARK_ROOT / "runs" / "benchmark_qc_51_initial_state_overlap"
DEFAULT_GAP_OUTDIR = BENCHMARK_ROOT / "runs" / "benchmark_qc_51_initial_state_gap"
HARTREE_TO_EV = 27.211386245988
OVERLAP_SCHEMA = "benchmark-qc.reference-matched-initial-overlap.v1"
GAP_SCHEMA = "benchmark-qc.reference-matched-gap.v1"


@dataclass(frozen=True)
class Case:
    case_id: str
    system: str
    hamiltonian_path: Path
    integral_data_path: Path
    metadata_path: Path
    n_qubits: int
    active_electrons: int
    active_orbitals: int
    spin: int
    reference_energy_hartree: float


def _resolve_dataset_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest.parent.parent / path


def _load_manifest(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cases.append(
                Case(
                    case_id=row["case_id"],
                    system=row.get("source_case_id") or row.get("system", "BenchmarkQC"),
                    hamiltonian_path=_resolve_dataset_path(path, row["hamiltonian_path"]),
                    integral_data_path=_resolve_dataset_path(path, row["integral_data_path"]),
                    metadata_path=_resolve_dataset_path(path, row["metadata_path"]),
                    n_qubits=int(row["qubits"]),
                    active_electrons=int(row["active_electrons"]),
                    active_orbitals=int(row["active_spatial_orbitals"]),
                    spin=int(row["spin"]),
                    reference_energy_hartree=float(row["reference_energy_hartree"]),
                )
            )
    return cases


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tmp_path.replace(path)


def _read_completed(path: Path, *, expected_schema: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("status") != "completed" or payload.get("schema") != expected_schema:
        return None
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path.resolve())


def _spin_populations(active_electrons: int, spin: int) -> tuple[int, int]:
    if (active_electrons + spin) % 2 != 0:
        raise ValueError(
            f"active_electrons={active_electrons} and spin={spin} have incompatible parity"
        )
    n_alpha = (active_electrons + spin) // 2
    n_beta = active_electrons - n_alpha
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(
            f"active_electrons={active_electrons} and spin={spin} imply a negative population"
        )
    return n_alpha, n_beta


def _krylov_reference_bits(case: Case) -> np.ndarray:
    n_alpha, n_beta = _spin_populations(case.active_electrons, case.spin)
    bits = np.zeros(case.n_qubits, dtype=int)
    bits[0 : 2 * n_alpha : 2] = 1
    bits[1 : 2 * n_beta : 2] = 1
    return bits


def _pyscf_string(bits: np.ndarray, *, spin_offset: int) -> int:
    value = 0
    for orbital, occupied in enumerate(bits[spin_offset::2]):
        if int(occupied):
            value |= 1 << orbital
    return value


def _root_convergence_flags(value: Any, *, root_count: int) -> list[bool]:
    array = np.asarray(value, dtype=bool).reshape(-1)
    if array.size == 1:
        return [bool(array[0])] * root_count
    if array.size != root_count:
        raise RuntimeError(
            f"FCI returned {array.size} convergence flags for {root_count} roots"
        )
    return [bool(item) for item in array]


def _load_archive(case: Case, benchmark_qc_repo: Path) -> Any:
    source_root = benchmark_qc_repo / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from benchmark_qc.integral_dataset import (  # noqa: PLC0415
        load_spatial_integral_archive,
    )

    return load_spatial_integral_archive(case.integral_data_path)


def _run_two_root_fci(
    case: Case,
    archive: Any,
    nelec: tuple[int, int],
    args: argparse.Namespace,
    *,
    enforce_target_spin: bool,
) -> dict[str, Any]:
    from pyscf import fci  # noqa: PLC0415

    solver = fci.direct_spin1.FCI()
    solver.conv_tol = float(args.conv_tol)
    solver.conv_tol_residual = float(args.conv_tol_residual)
    solver.max_cycle = int(args.max_cycle)
    solver.max_space = int(args.max_space)
    solver.davidson_only = True
    solver.lindep = float(args.lindep)
    target_spin_square = float(case.spin * (case.spin + 2) / 4.0)
    if enforce_target_spin:
        fci.addons.fix_spin_(
            solver,
            shift=float(args.spin_penalty_shift),
            ss=target_spin_square,
        )

    start = time.perf_counter()
    energies, ci_vectors = solver.kernel(
        archive.one_body_integrals,
        archive.two_body_integrals,
        case.active_orbitals,
        nelec,
        ecore=archive.core_constant,
        nroots=2,
    )
    elapsed = float(time.perf_counter() - start)
    energy_values = np.asarray(energies, dtype=float).reshape(-1)
    vectors = list(ci_vectors) if isinstance(ci_vectors, (list, tuple)) else [ci_vectors]
    if energy_values.size != 2 or len(vectors) != 2:
        raise RuntimeError(
            f"{case.case_id}: expected two FCI roots, got "
            f"{energy_values.size} energies and {len(vectors)} vectors"
        )
    order = np.argsort(energy_values, kind="stable")
    energy_values = energy_values[order]
    vectors = [np.asarray(vectors[int(index)]) for index in order]
    convergence = _root_convergence_flags(solver.converged, root_count=2)
    return {
        "energies": energy_values,
        "vectors": vectors,
        "convergence": convergence,
        "elapsed_seconds": elapsed,
        "enforce_target_spin": enforce_target_spin,
        "solver": (
            "pyscf-direct_spin1-fci-with-spin-penalty"
            if enforce_target_spin
            else "pyscf-direct_spin1-fci"
        ),
    }


def _solve_case(case: Case, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    from pyscf import fci  # noqa: PLC0415

    archive = _load_archive(case, args.benchmark_qc_repo)
    n_alpha, n_beta = _spin_populations(case.active_electrons, case.spin)
    if archive.n_spatial_orbitals != case.active_orbitals:
        raise ValueError(
            f"{case.case_id}: integral archive has {archive.n_spatial_orbitals} orbitals, "
            f"manifest has {case.active_orbitals}"
        )
    if archive.n_qubits != case.n_qubits:
        raise ValueError(
            f"{case.case_id}: integral archive has {archive.n_qubits} qubits, "
            f"manifest has {case.n_qubits}"
        )

    archived_bits = np.asarray(archive.reference_determinant, dtype=int)
    reference_bits = _krylov_reference_bits(case)
    if not np.array_equal(archived_bits, reference_bits):
        raise ValueError(
            f"{case.case_id}: archived reference determinant does not match the "
            "QRTE/QKUD reference bitstring"
        )
    if int(np.sum(reference_bits[0::2])) != n_alpha:
        raise ValueError(f"{case.case_id}: reference determinant has the wrong alpha count")
    if int(np.sum(reference_bits[1::2])) != n_beta:
        raise ValueError(f"{case.case_id}: reference determinant has the wrong beta count")

    target_spin_square = float(case.spin * (case.spin + 2) / 4.0)
    fixed_population = _run_two_root_fci(
        case,
        archive,
        (n_alpha, n_beta),
        args,
        enforce_target_spin=False,
    )
    fixed_population_diff = (
        float(fixed_population["energies"][0]) - case.reference_energy_hartree
    )
    fixed_population_matches = (
        all(fixed_population["convergence"])
        and abs(fixed_population_diff) <= float(args.max_reference_energy_diff)
    )
    if fixed_population_matches:
        selected = fixed_population
        reference_sector = "fixed_n_alpha_n_beta"
        target_spin = None
    else:
        target_spin = _run_two_root_fci(
            case,
            archive,
            (n_alpha, n_beta),
            args,
            enforce_target_spin=True,
        )
        selected = target_spin
        reference_sector = "fixed_total_spin"

    energy_values = selected["energies"]
    vectors = selected["vectors"]
    convergence = selected["convergence"]
    if not all(convergence):
        raise RuntimeError(
            f"{case.case_id}: selected {reference_sector} FCI roots did not both "
            f"converge: {convergence}; fixed-population convergence was "
            f"{fixed_population['convergence']}"
        )

    root_spin_squares: list[float] = []
    root_multiplicities: list[float] = []
    for index, vector in enumerate(vectors):
        spin_square, multiplicity = fci.spin_op.spin_square0(
            vector,
            case.active_orbitals,
            (n_alpha, n_beta),
        )
        root_spin_squares.append(float(spin_square))
        root_multiplicities.append(float(multiplicity))
        spin_error = abs(float(spin_square) - target_spin_square)
        if reference_sector == "fixed_total_spin" and spin_error > float(args.spin_square_atol):
            raise RuntimeError(
                f"{case.case_id}: root {index} has <S^2>={float(spin_square):.12g}, "
                f"target={target_spin_square:.12g}, error={spin_error:.3e}"
            )

    ground_energy = float(energy_values[0])
    first_excited_energy = float(energy_values[1])
    gap = float(first_excited_energy - ground_energy)
    if gap < -float(args.energy_atol):
        raise RuntimeError(f"{case.case_id}: negative sorted spectral gap {gap}")
    ground_minus_reference = ground_energy - case.reference_energy_hartree
    if abs(ground_minus_reference) > float(args.max_reference_energy_diff):
        raise RuntimeError(
            f"{case.case_id}: ground energy {ground_energy:.15f} differs from "
            f"reference {case.reference_energy_hartree:.15f} by "
            f"{ground_minus_reference:.3e} Ha"
        )

    alpha_string = _pyscf_string(reference_bits, spin_offset=0)
    beta_string = _pyscf_string(reference_bits, spin_offset=1)
    alpha_position = int(
        fci.cistring.str2addr(case.active_orbitals, n_alpha, alpha_string)
    )
    beta_position = int(
        fci.cistring.str2addr(case.active_orbitals, n_beta, beta_string)
    )
    ground_vector = vectors[0]
    expected_shape = (
        math.comb(case.active_orbitals, n_alpha),
        math.comb(case.active_orbitals, n_beta),
    )
    if ground_vector.shape != expected_shape:
        raise RuntimeError(
            f"{case.case_id}: ground CI shape {ground_vector.shape} != {expected_shape}"
        )
    amplitude = complex(ground_vector[alpha_position, beta_position])
    overlap_weight = float(abs(amplitude) ** 2)
    if overlap_weight <= 0.0:
        raise RuntimeError(f"{case.case_id}: reference determinant has zero ground overlap")

    sector_dim = int(math.prod(expected_shape))
    common = {
        "status": "completed",
        "case_id": case.case_id,
        "system": case.system,
        "manifest": _display_path(args.manifest),
        "hamiltonian_path": _display_path(case.hamiltonian_path),
        "integral_data_path": _display_path(case.integral_data_path),
        "metadata_path": _display_path(case.metadata_path),
        "hamiltonian_representation": "validated spatial integrals equivalent to saved JW Hamiltonian",
        "spin_orbital_order": "interleaved",
        "solver": selected["solver"],
        "reference_sector": reference_sector,
        "reference_sector_selection": (
            "Use the fixed-N_alpha,N_beta ground state when it matches the stored "
            "BenchmarkQC reference; otherwise use the manifest target-S^2 ground state."
        ),
        "conv_tol": float(args.conv_tol),
        "conv_tol_residual": float(args.conv_tol_residual),
        "max_cycle": int(args.max_cycle),
        "max_space": int(args.max_space),
        "davidson_only": True,
        "lindep": float(args.lindep),
        "spin_penalty_shift": float(args.spin_penalty_shift),
        "target_spin_square": target_spin_square,
        "root_spin_squares": root_spin_squares,
        "root_multiplicities": root_multiplicities,
        "spin_square_atol": float(args.spin_square_atol),
        "roots_converged": convergence,
        "fixed_population_roots_converged": fixed_population["convergence"],
        "fixed_population_ground_energy_hartree": float(
            fixed_population["energies"][0]
        ),
        "fixed_population_ground_minus_reference_hartree": float(
            fixed_population_diff
        ),
        "active_electrons": int(case.active_electrons),
        "active_orbitals": int(case.active_orbitals),
        "spin": int(case.spin),
        "qubits": int(case.n_qubits),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "sector_dim": sector_dim,
        "reference_bits": [int(bit) for bit in reference_bits],
        "reference_bitstring": "".join(str(int(bit)) for bit in reference_bits),
        "reference_alpha_string": int(alpha_string),
        "reference_beta_string": int(beta_string),
        "reference_alpha_position": int(alpha_position),
        "reference_beta_position": int(beta_position),
        "reference_sector_position": int(alpha_position * expected_shape[1] + beta_position),
        "reference_energy_hartree": float(case.reference_energy_hartree),
        "sector_ground_energy_hartree": ground_energy,
        "ground_minus_reference_hartree": float(ground_minus_reference),
        "amplitude_real": float(amplitude.real),
        "amplitude_imag": float(amplitude.imag),
        "overlap_weight": overlap_weight,
        "negative_log10_overlap": float(-math.log10(overlap_weight)),
        "largest_determinant_probability": float(np.max(np.abs(ground_vector) ** 2)),
        "reference_is_largest_determinant": bool(
            overlap_weight >= float(np.max(np.abs(ground_vector) ** 2)) - 1.0e-12
        ),
        "elapsed_seconds": float(
            fixed_population["elapsed_seconds"]
            + (0.0 if target_spin is None else target_spin["elapsed_seconds"])
        ),
    }
    overlap = {
        **common,
        "schema": OVERLAP_SCHEMA,
    }
    gap_result = {
        **common,
        "schema": GAP_SCHEMA,
        "nroots_requested": 2,
        "nroots_returned": 2,
        "energies_hartree": [ground_energy, first_excited_energy],
        "first_excited_energy_hartree": first_excited_energy,
        "gap_hartree": gap,
        "gap_millihartree": float(1000.0 * gap),
        "gap_ev": float(HARTREE_TO_EV * gap),
    }
    return overlap, gap_result


def compute_case(case: Case, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    overlap_path = args.overlap_outdir / case.case_id / "initial_overlap.json"
    gap_path = args.gap_outdir / case.case_id / "sector_gap.json"
    if not args.force:
        overlap = _read_completed(overlap_path, expected_schema=OVERLAP_SCHEMA)
        gap = _read_completed(gap_path, expected_schema=GAP_SCHEMA)
        if overlap is not None and gap is not None:
            print(f"SKIP {case.case_id}: completed outputs already exist", flush=True)
            return overlap, gap

    print(f"START {case.case_id}", flush=True)
    try:
        overlap, gap = _solve_case(case, args)
        _write_json(overlap_path, overlap)
        _write_json(gap_path, gap)
        (args.gap_outdir / case.case_id / "gap_overlap_failure.json").unlink(
            missing_ok=True
        )
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case.case_id,
            "error": repr(exc),
            "manifest": _display_path(args.manifest),
            "integral_data_path": _display_path(case.integral_data_path),
        }
        _write_json(args.gap_outdir / case.case_id / "gap_overlap_failure.json", failure)
        raise

    print(
        f"DONE {case.case_id}: gap={gap['gap_hartree']:.12g} Ha, "
        f"overlap={overlap['overlap_weight']:.12g}",
        flush=True,
    )
    return overlap, gap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-qc-repo", type=Path, default=DEFAULT_BENCHMARK_QC_REPO)
    parser.add_argument("--overlap-outdir", type=Path, default=DEFAULT_OVERLAP_OUTDIR)
    parser.add_argument("--gap-outdir", type=Path, default=DEFAULT_GAP_OUTDIR)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--conv-tol-residual", type=float, default=1.0e-11)
    parser.add_argument("--max-cycle", type=int, default=1000)
    parser.add_argument("--max-space", type=int, default=100)
    parser.add_argument("--lindep", type=float, default=1.0e-24)
    parser.add_argument("--spin-penalty-shift", type=float, default=0.5)
    parser.add_argument("--spin-square-atol", type=float, default=1.0e-7)
    parser.add_argument("--max-reference-energy-diff", type=float, default=1.0e-7)
    parser.add_argument("--energy-atol", type=float, default=1.0e-10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.benchmark_qc_repo = args.benchmark_qc_repo.resolve()
    args.overlap_outdir = (
        args.overlap_outdir
        if args.overlap_outdir.is_absolute()
        else BENCHMARK_ROOT / args.overlap_outdir
    )
    args.gap_outdir = (
        args.gap_outdir if args.gap_outdir.is_absolute() else BENCHMARK_ROOT / args.gap_outdir
    )
    return args


def main() -> None:
    args = parse_args()
    cases = _load_manifest(args.manifest)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested.difference(case.case_id for case in cases)
        if missing:
            raise SystemExit(f"Unknown BenchmarkQC case IDs: {sorted(missing)}")
    if args.expected_cases is not None and len(cases) != int(args.expected_cases):
        raise SystemExit(f"Expected {args.expected_cases} cases, selected {len(cases)}")
    if not cases:
        raise SystemExit("No BenchmarkQC cases selected")

    for case in cases:
        compute_case(case, args)


if __name__ == "__main__":
    main()
