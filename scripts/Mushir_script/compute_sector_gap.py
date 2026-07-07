#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Compute the fixed-sector FCI gap from a cached active-space Hamiltonian."""

from __future__ import annotations

import argparse
import json
import os
import sys
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

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_OVERLAP_DIR = BENCHMARK_ROOT / "runs" / "initial_state_overlap"
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "initial_state_gap"
HARTREE_TO_EV = 27.211386245988


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = BENCHMARK_ROOT / path
    return path


def _sector_dim_from_occupations(n_spatial: int, n_alpha: int, n_beta: int) -> int:
    import math

    return int(math.comb(int(n_spatial), int(n_alpha)) * math.comb(int(n_spatial), int(n_beta)))


def _as_float_list(values: Any) -> list[float]:
    import numpy as np

    array = np.asarray(values, dtype=float).reshape(-1)
    return [float(value) for value in array]


def compute(args: argparse.Namespace) -> dict[str, Any]:
    qcant_root = args.qcant_root.resolve()
    if str(qcant_root) not in sys.path:
        sys.path.insert(0, str(qcant_root))
    from QCANT import qulacs_accel as qa  # noqa: PLC0415

    stack = qa._import_optional_stack()
    np = stack["np"]
    from pyscf import fci  # noqa: PLC0415

    overlap_path = args.overlap_json
    if overlap_path is None:
        if args.case_id is None:
            raise ValueError("Provide either --case-id or --overlap-json")
        overlap_path = args.overlap_dir / args.case_id / "initial_overlap.json"
    overlap_path = overlap_path.resolve()
    overlap = _load_json(overlap_path)
    if overlap.get("status") != "completed":
        raise ValueError(f"Overlap JSON is not completed: {overlap_path}")

    case_id = str(overlap["case_id"])
    cache_path = _resolve_path(overlap["hamiltonian_cache_path"])
    integral_payload = qa._load_hamiltonian_integral_cache(cache_path, np)
    if integral_payload is None:
        raise FileNotFoundError(f"Hamiltonian cache not found or unreadable: {cache_path}")

    norb = int(integral_payload["ncas"])
    nelecas = tuple(int(value) for value in integral_payload.get("nelecas", []))
    if len(nelecas) != 2:
        raise ValueError(f"PySCF FCI solver needs spin-resolved nelecas, got {nelecas!r}")
    n_alpha, n_beta = nelecas
    sector_dim = _sector_dim_from_occupations(norb, n_alpha, n_beta)
    if int(overlap["sector_dim"]) != sector_dim:
        raise ValueError(f"Sector dimension mismatch for {case_id}: overlap={overlap['sector_dim']} cache={sector_dim}")

    h1 = np.asarray(integral_payload["h1ecas"], dtype=float)
    # QCANT stores two-electron integrals after a swap for PennyLane's convention.
    # PySCF FCI expects the restored PySCF ordering.
    eri = np.swapaxes(np.asarray(integral_payload["two_mo"], dtype=float), 1, 3)
    ecore = float(integral_payload["ecore"])

    start = time.perf_counter()
    energies, _ci_vectors = fci.direct_spin1.kernel(
        h1,
        eri,
        norb,
        nelecas,
        ecore=ecore,
        nroots=int(args.nroots),
        max_cycle=int(args.max_cycle),
        conv_tol=float(args.conv_tol),
    )
    elapsed = float(time.perf_counter() - start)
    energy_values = sorted(_as_float_list(energies))
    if len(energy_values) < 2:
        raise RuntimeError(f"Expected at least two FCI roots for {case_id}, got {len(energy_values)}")

    ground_energy = float(energy_values[0])
    first_excited_energy = float(energy_values[1])
    gap = float(first_excited_energy - ground_energy)
    overlap_ground_energy = float(overlap["sector_ground_energy_hartree"])
    ground_energy_diff = float(ground_energy - overlap_ground_energy)
    if abs(ground_energy_diff) > float(args.max_ground_energy_diff):
        raise RuntimeError(
            f"Ground energy mismatch for {case_id}: nroots E0={ground_energy} "
            f"overlap E0={overlap_ground_energy} diff={ground_energy_diff}"
        )

    return {
        "status": "completed",
        "case_id": case_id,
        "overlap_path": str(overlap_path.relative_to(BENCHMARK_ROOT)),
        "overlap_run_dir": str(overlap.get("run_dir", "")),
        "hamiltonian_cache_path": str(cache_path),
        "hamiltonian_source": str(integral_payload["hamiltonian_source"]),
        "reference_energy_hartree": float(integral_payload["reference_energy"]),
        "reference_converged": bool(integral_payload["reference_converged"]),
        "nroots_requested": int(args.nroots),
        "nroots_returned": int(len(energy_values)),
        "energies_hartree": energy_values,
        "sector_ground_energy_hartree": ground_energy,
        "first_excited_energy_hartree": first_excited_energy,
        "gap_hartree": gap,
        "gap_millihartree": float(1000.0 * gap),
        "gap_ev": float(HARTREE_TO_EV * gap),
        "overlap_sector_ground_energy_hartree": overlap_ground_energy,
        "ground_energy_minus_overlap_hartree": ground_energy_diff,
        "overlap_weight": float(overlap["overlap_weight"]),
        "negative_log10_overlap": float(overlap["negative_log10_overlap"]),
        "qubits": int(overlap["qubits"]),
        "n_spatial_orbitals": int(norb),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "sector_dim": int(sector_dim),
        "solver": "pyscf-fci",
        "conv_tol": float(args.conv_tol),
        "max_cycle": int(args.max_cycle),
        "elapsed_seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--overlap-json", type=Path, default=None)
    parser.add_argument("--overlap-dir", type=Path, default=DEFAULT_OVERLAP_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--nroots", type=int, default=2)
    parser.add_argument("--conv-tol", type=float, default=1e-12)
    parser.add_argument("--max-cycle", type=int, default=300)
    parser.add_argument("--max-ground-energy-diff", type=float, default=1e-7)
    args = parser.parse_args()
    if int(args.nroots) < 2:
        raise SystemExit("--nroots must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    result = compute(args)
    output = args.output
    if output is None:
        output = args.outdir / str(result["case_id"]) / "sector_gap.json"
    _write_json(output, result)
    print(f"Wrote {output}")
    for key in (
        "case_id",
        "status",
        "sector_dim",
        "sector_ground_energy_hartree",
        "first_excited_energy_hartree",
        "gap_hartree",
        "gap_ev",
        "elapsed_seconds",
    ):
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
