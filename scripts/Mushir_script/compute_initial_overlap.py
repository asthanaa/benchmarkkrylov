#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Compute the initial determinant overlap with the sector ground state."""

from __future__ import annotations

import argparse
import itertools
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
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "initial_state_overlap"


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


def _resolve_cache_dir(summary: dict[str, Any], run_config: dict[str, Any], run_dir: Path) -> Path:
    raw = run_config.get("hamiltonian_cache_dir") or summary.get("hamiltonian_cache_dir")
    if raw is None:
        raise ValueError(f"No hamiltonian_cache_dir found in {run_dir}")
    path = Path(str(raw))
    if not path.is_absolute():
        path = BENCHMARK_ROOT / path
    return path


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = BENCHMARK_ROOT / path
    return path


def _build_sector_indices(hf_bits: Any, qa: Any, np: Any) -> tuple[Any, int, int, int]:
    bits = np.asarray(hf_bits, dtype=int)
    qubits = int(bits.size)
    n_spatial = qubits // 2
    n_alpha = int(np.sum(bits[0::2]))
    n_beta = int(np.sum(bits[1::2]))
    hf_index = int(qa._bits_to_index(bits))

    sector_indices: list[int] = []
    for alpha_occ in itertools.combinations(range(n_spatial), n_alpha):
        alpha_wires = [2 * index for index in alpha_occ]
        for beta_occ in itertools.combinations(range(n_spatial), n_beta):
            sector_bits = np.zeros(qubits, dtype=int)
            sector_bits[alpha_wires] = 1
            sector_bits[[2 * index + 1 for index in beta_occ]] = 1
            sector_indices.append(int(qa._bits_to_index(sector_bits)))

    sector = np.asarray(sorted(sector_indices), dtype=int)
    matches = np.where(sector == hf_index)[0]
    if matches.size != 1:
        raise ValueError("Initial determinant was not found exactly once in the fixed occupation sector")
    return sector, int(matches[0]), n_alpha, n_beta


def _diagonalize_sector(
    h_sector: Any,
    *,
    solver: str,
    dense_threshold: int,
    eigsh_tol: float,
    eigsh_maxiter: int,
):
    import numpy as np

    if solver == "auto":
        solver = "dense" if h_sector.shape[0] <= int(dense_threshold) else "eigsh"

    if solver == "dense":
        from scipy.linalg import eigh

        matrix = h_sector.toarray() if hasattr(h_sector, "toarray") else np.asarray(h_sector)
        evals, evecs = eigh(matrix)
        return float(evals[0]), evecs[:, 0], solver

    if solver == "eigsh":
        from scipy.sparse.linalg import eigsh

        evals, evecs = eigsh(
            h_sector,
            k=1,
            which="SA",
            tol=float(eigsh_tol),
            maxiter=int(eigsh_maxiter),
        )
        order = np.argsort(evals)
        return float(evals[order[0]]), evecs[:, order[0]], solver

    raise ValueError("solver must be one of {'auto', 'dense', 'eigsh', 'pyscf-fci'}")


def _sector_dim_from_occupations(n_spatial: int, n_alpha: int, n_beta: int) -> int:
    import math

    return int(math.comb(int(n_spatial), int(n_alpha)) * math.comb(int(n_spatial), int(n_beta)))


def _diagonalize_cache_with_pyscf_fci(cache_path: Path, *, qa: Any) -> dict[str, Any]:
    stack = qa._import_optional_stack()
    np = stack["np"]
    from pyscf import fci  # noqa: PLC0415

    integral_payload = qa._load_hamiltonian_integral_cache(cache_path, np)
    if integral_payload is None:
        raise FileNotFoundError(f"Hamiltonian cache not found or unreadable: {cache_path}")

    norb = int(integral_payload["ncas"])
    nelecas = tuple(int(value) for value in integral_payload.get("nelecas", []))
    if len(nelecas) != 2:
        raise ValueError(f"PySCF FCI solver needs spin-resolved nelecas, got {nelecas!r}")
    n_alpha, n_beta = nelecas
    h1 = np.asarray(integral_payload["h1ecas"], dtype=float)
    # QCANT stores two-electron integrals after a swap for PennyLane's convention.
    # PySCF FCI expects the restored PySCF ordering.
    eri = np.swapaxes(np.asarray(integral_payload["two_mo"], dtype=float), 1, 3)
    ecore = float(integral_payload["ecore"])

    energy, ci_vector = fci.direct_spin1.kernel(
        h1,
        eri,
        norb,
        nelecas,
        ecore=ecore,
        max_cycle=200,
        conv_tol=1e-12,
    )
    alpha_strings = fci.cistring.gen_strings4orblist(range(norb), n_alpha)
    beta_strings = fci.cistring.gen_strings4orblist(range(norb), n_beta)
    alpha_ref = sum(1 << index for index in range(n_alpha))
    beta_ref = sum(1 << index for index in range(n_beta))
    alpha_matches = np.where(alpha_strings == alpha_ref)[0]
    beta_matches = np.where(beta_strings == beta_ref)[0]
    if alpha_matches.size != 1 or beta_matches.size != 1:
        raise ValueError("Reference occupation determinant not found in PySCF FCI string basis")

    alpha_position = int(alpha_matches[0])
    beta_position = int(beta_matches[0])
    amplitude = ci_vector[alpha_position, beta_position]
    qubits = 2 * norb
    hf_bits = qa._spin_resolved_hf_bits(nelecas, qubits, np)
    sector_dim = _sector_dim_from_occupations(norb, n_alpha, n_beta)

    return {
        "energy": float(energy),
        "amplitude": amplitude,
        "hf_bits": hf_bits,
        "qubits": int(qubits),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "sector_dim": int(sector_dim),
        "hf_sector_position": int(alpha_position * len(beta_strings) + beta_position),
        "hf_alpha_string_position": alpha_position,
        "hf_beta_string_position": beta_position,
        "reference_energy": float(integral_payload["reference_energy"]),
        "hamiltonian_source": str(integral_payload["hamiltonian_source"]),
        "reference_converged": bool(integral_payload["reference_converged"]),
    }


def _payload_from_cache_path(cache_path: Path, *, hamiltonian_cutoff: float, qa: Any) -> dict[str, Any]:
    stack = qa._import_optional_stack()
    np = stack["np"]
    qml = stack["qml"]
    Observable = stack["Observable"]

    integral_payload = qa._load_hamiltonian_integral_cache(cache_path, np)
    if integral_payload is None:
        raise FileNotFoundError(f"Hamiltonian cache not found or unreadable: {cache_path}")
    integral_payload = dict(integral_payload)
    integral_payload["cache_path"] = str(cache_path)
    integral_payload["loaded_from_cache"] = True

    qubits = 2 * int(integral_payload["ncas"])
    hf_bits = qa._spin_resolved_hf_bits(integral_payload.get("nelecas", []), qubits, np)
    if hf_bits is None:
        hf_bits = np.asarray(qml.qchem.hf_state(int(integral_payload["effective_active_electrons"]), qubits), dtype=int)

    h_fermionic = qml.qchem.fermionic_observable(
        np.array([float(integral_payload["ecore"])]),
        integral_payload["h1ecas"],
        integral_payload["two_mo"],
        cutoff=float(hamiltonian_cutoff),
    )
    h_qml = qml.jordan_wigner(h_fermionic)
    h_qml = qa._sanitize_hamiltonian(h_qml, np=np, qml=qml)
    h_qulacs, pauli_terms = qa._qml_operator_to_qulacs_operator(
        h_qml,
        qubits=qubits,
        qml=qml,
        operator_cls=Observable,
    )
    return {
        **stack,
        "h_qml": h_qml,
        "h_qulacs": h_qulacs,
        "pauli_terms": pauli_terms,
        "qubits": int(qubits),
        "hf_bits": hf_bits,
        "active_electrons": int(integral_payload["effective_active_electrons"]),
        "active_orbital_index_base": int(integral_payload.get("active_orbital_index_base", 0)),
        "active_orbital_indices": list(integral_payload.get("active_orbital_indices", [])),
        "active_space_selection": integral_payload.get("active_space_selection"),
        "hamiltonian_cache_path": str(cache_path),
        "hamiltonian_loaded_from_cache": True,
        "hamiltonian_source": str(integral_payload["hamiltonian_source"]),
        "reference_energy": float(integral_payload["reference_energy"]),
        "reference_converged": bool(integral_payload["reference_converged"]),
    }


def _minimal_payload_from_cache_path(cache_path: Path, qa: Any) -> dict[str, Any]:
    stack = qa._import_optional_stack()
    np = stack["np"]
    qml = stack["qml"]
    integral_payload = qa._load_hamiltonian_integral_cache(cache_path, np)
    if integral_payload is None:
        raise FileNotFoundError(f"Hamiltonian cache not found or unreadable: {cache_path}")

    qubits = 2 * int(integral_payload["ncas"])
    hf_bits = qa._spin_resolved_hf_bits(integral_payload.get("nelecas", []), qubits, np)
    if hf_bits is None:
        hf_bits = np.asarray(qml.qchem.hf_state(int(integral_payload["effective_active_electrons"]), qubits), dtype=int)
    return {
        "np": np,
        "qubits": int(qubits),
        "hf_bits": hf_bits,
        "hamiltonian_cache_path": str(cache_path),
        "hamiltonian_loaded_from_cache": True,
        "hamiltonian_source": str(integral_payload["hamiltonian_source"]),
        "reference_energy": float(integral_payload["reference_energy"]),
        "reference_converged": bool(integral_payload["reference_converged"]),
    }


def compute(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    qcant_root = args.qcant_root.resolve()
    if str(qcant_root) not in sys.path:
        sys.path.insert(0, str(qcant_root))
    from QCANT import qulacs_accel as qa  # noqa: PLC0415

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    snapshot_path = run_dir / "input_snapshot.json"
    run_config_path = run_dir / "run_config.json"
    summary = _load_json(summary_path)
    snapshot = _load_json(snapshot_path)
    run_config = _load_json(run_config_path)

    cache_dir = args.hamiltonian_cache_dir.resolve() if args.hamiltonian_cache_dir is not None else _resolve_cache_dir(summary, run_config, run_dir)
    expected_cache_path = _path_or_none(summary.get("hamiltonian_cache_path"))

    active_orbital_labels = snapshot.get("active_orbital_labels")
    if active_orbital_labels is not None:
        active_orbital_labels = [str(label) for label in active_orbital_labels]
    active_orbital_indices = snapshot.get("active_orbital_indices")
    if active_orbital_indices is not None:
        active_orbital_indices = [int(index) for index in active_orbital_indices]
    active_orbital_index_base = int(snapshot.get("active_orbital_index_base") or 0)

    start = time.perf_counter()
    loaded_directly_from_summary_cache = False
    if expected_cache_path is not None and expected_cache_path.exists() and not args.force_build_payload:
        if args.dry_run or args.solver == "pyscf-fci":
            payload = _minimal_payload_from_cache_path(expected_cache_path, qa)
        else:
            payload = _payload_from_cache_path(expected_cache_path, hamiltonian_cutoff=float(args.hamiltonian_cutoff), qa=qa)
        loaded_directly_from_summary_cache = True
    else:
        payload = qa._build_hamiltonian_payload(
            symbols=list(snapshot["symbols"]),
            geometry=np.asarray(snapshot["geometry_angstrom"], dtype=float),
            basis=str(snapshot["basis"]),
            charge=int(snapshot["charge"]),
            spin=int(snapshot["spin"]),
            active_electrons=int(snapshot["active_electrons"]),
            active_orbitals=int(snapshot["active_orbitals"]),
            hamiltonian_cutoff=float(args.hamiltonian_cutoff),
            hamiltonian_source=str(run_config.get("hamiltonian_source", summary.get("hamiltonian_source", "casscf_natorb"))),
            hamiltonian_cache_dir=str(cache_dir),
            active_orbital_labels=active_orbital_labels,
            active_orbital_indices=active_orbital_indices,
            active_orbital_index_base=active_orbital_index_base,
            allow_unconverged_casscf=bool(run_config.get("allow_unconverged_casscf", summary.get("allow_unconverged_casscf", False))),
        )

    actual_cache_path = _path_or_none(payload.get("hamiltonian_cache_path"))
    if args.require_cache and not bool(payload.get("hamiltonian_loaded_from_cache", False)):
        raise RuntimeError(
            "Hamiltonian was not loaded from cache. Refusing to recompute because that may change the reference "
            "Hamiltonian; rerun with --allow-recompute if this is intentional."
        )
    if expected_cache_path is not None and actual_cache_path is not None and expected_cache_path.resolve() != actual_cache_path.resolve():
        raise RuntimeError(
            f"Loaded cache path does not match summary cache path: {actual_cache_path} != {expected_cache_path}"
        )

    sector_indices, hf_sector_position, n_alpha, n_beta = _build_sector_indices(payload["hf_bits"], qa, np)

    if args.dry_run:
        return {
            "status": "dry_run",
            "case_id": summary.get("case_id", snapshot.get("case_id")),
            "run_dir": str(run_dir.relative_to(BENCHMARK_ROOT)),
            "qubits": int(payload["qubits"]),
            "full_hilbert_dim": int(2 ** int(payload["qubits"])),
            "sector_dim": int(sector_indices.size),
            "n_alpha": int(n_alpha),
            "n_beta": int(n_beta),
            "hf_bits": "".join(str(int(bit)) for bit in np.asarray(payload["hf_bits"], dtype=int)),
            "hf_full_basis_index": int(qa._bits_to_index(payload["hf_bits"])),
            "hf_sector_position": int(hf_sector_position),
            "hamiltonian_cache_path": "" if actual_cache_path is None else str(actual_cache_path),
            "hamiltonian_loaded_from_cache": bool(payload.get("hamiltonian_loaded_from_cache", False)),
            "loaded_directly_from_summary_cache": loaded_directly_from_summary_cache,
        }

    if args.solver == "pyscf-fci":
        if actual_cache_path is None:
            raise RuntimeError("The PySCF FCI solver requires a Hamiltonian cache path.")
        fci_result = _diagonalize_cache_with_pyscf_fci(actual_cache_path, qa=qa)
        amplitude = fci_result["amplitude"]
        overlap_weight = float(abs(amplitude) ** 2)
        elapsed = float(time.perf_counter() - start)
        reference_energy = float(fci_result["reference_energy"])
        result = {
            "status": "completed",
            "case_id": summary.get("case_id", snapshot.get("case_id")),
            "run_dir": str(run_dir.relative_to(BENCHMARK_ROOT)),
            "summary_path": str(summary_path.relative_to(BENCHMARK_ROOT)),
            "input_snapshot_path": str(snapshot_path.relative_to(BENCHMARK_ROOT)),
            "run_config_path": str(run_config_path.relative_to(BENCHMARK_ROOT)),
            "method": summary.get("method"),
            "hit_iteration": summary.get("hit_iteration"),
            "hit_chemical_accuracy": bool(summary.get("hit_chemical_accuracy", False)),
            "reference_source": summary.get("reference_source", snapshot.get("reference_source", "")),
            "hamiltonian_source": fci_result["hamiltonian_source"],
            "hamiltonian_cache_path": str(actual_cache_path),
            "hamiltonian_loaded_from_cache": bool(payload.get("hamiltonian_loaded_from_cache", False)),
            "loaded_directly_from_summary_cache": loaded_directly_from_summary_cache,
            "reference_energy_hartree": reference_energy,
            "sector_ground_energy_hartree": float(fci_result["energy"]),
            "energy_minus_reference_hartree": float(fci_result["energy"] - reference_energy),
            "qubits": int(fci_result["qubits"]),
            "full_hilbert_dim": int(2 ** int(fci_result["qubits"])),
            "sector_dim": int(fci_result["sector_dim"]),
            "n_alpha": int(fci_result["n_alpha"]),
            "n_beta": int(fci_result["n_beta"]),
            "hf_bits": "".join(str(int(bit)) for bit in np.asarray(fci_result["hf_bits"], dtype=int)),
            "hf_full_basis_index": int(qa._bits_to_index(fci_result["hf_bits"])),
            "hf_sector_position": int(fci_result["hf_sector_position"]),
            "hf_alpha_string_position": int(fci_result["hf_alpha_string_position"]),
            "hf_beta_string_position": int(fci_result["hf_beta_string_position"]),
            "amplitude_real": float(np.real(amplitude)),
            "amplitude_imag": float(np.imag(amplitude)),
            "overlap_weight": overlap_weight,
            "negative_log10_overlap": float(-np.log10(overlap_weight)),
            "solver": "pyscf-fci",
            "eigensolve_residual": None,
            "elapsed_seconds": elapsed,
        }
        return result

    h_full = payload["h_qulacs"].get_matrix().tocsr()
    h_sector = h_full[sector_indices, :][:, sector_indices].tocsr()
    ground_energy, ground_vector, solver_used = _diagonalize_sector(
        h_sector,
        solver=args.solver,
        dense_threshold=int(args.dense_threshold),
        eigsh_tol=float(args.eigsh_tol),
        eigsh_maxiter=int(args.eigsh_maxiter),
    )
    amplitude = ground_vector[hf_sector_position]
    overlap_weight = float(abs(amplitude) ** 2)
    residual = float(np.linalg.norm(h_sector @ ground_vector - ground_energy * ground_vector))

    elapsed = float(time.perf_counter() - start)
    reference_energy = float(payload["reference_energy"])
    result = {
        "status": "completed",
        "case_id": summary.get("case_id", snapshot.get("case_id")),
        "run_dir": str(run_dir.relative_to(BENCHMARK_ROOT)),
        "summary_path": str(summary_path.relative_to(BENCHMARK_ROOT)),
        "input_snapshot_path": str(snapshot_path.relative_to(BENCHMARK_ROOT)),
        "run_config_path": str(run_config_path.relative_to(BENCHMARK_ROOT)),
        "method": summary.get("method"),
        "hit_iteration": summary.get("hit_iteration"),
        "hit_chemical_accuracy": bool(summary.get("hit_chemical_accuracy", False)),
        "reference_source": summary.get("reference_source", snapshot.get("reference_source", "")),
        "hamiltonian_source": payload["hamiltonian_source"],
        "hamiltonian_cache_path": "" if actual_cache_path is None else str(actual_cache_path),
        "hamiltonian_loaded_from_cache": bool(payload.get("hamiltonian_loaded_from_cache", False)),
        "loaded_directly_from_summary_cache": loaded_directly_from_summary_cache,
        "reference_energy_hartree": reference_energy,
        "sector_ground_energy_hartree": float(ground_energy),
        "energy_minus_reference_hartree": float(ground_energy - reference_energy),
        "qubits": int(payload["qubits"]),
        "full_hilbert_dim": int(h_full.shape[0]),
        "sector_dim": int(h_sector.shape[0]),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "hf_bits": "".join(str(int(bit)) for bit in np.asarray(payload["hf_bits"], dtype=int)),
        "hf_full_basis_index": int(qa._bits_to_index(payload["hf_bits"])),
        "hf_sector_position": int(hf_sector_position),
        "amplitude_real": float(np.real(amplitude)),
        "amplitude_imag": float(np.imag(amplitude)),
        "overlap_weight": overlap_weight,
        "negative_log10_overlap": float(-np.log10(overlap_weight)),
        "solver": solver_used,
        "eigensolve_residual": residual,
        "elapsed_seconds": elapsed,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run case directory containing summary/input_snapshot/run_config JSON.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=None)
    parser.add_argument("--hamiltonian-cutoff", type=float, default=1e-20)
    parser.add_argument("--solver", choices=("auto", "dense", "eigsh", "pyscf-fci"), default="auto")
    parser.add_argument("--dense-threshold", type=int, default=1000)
    parser.add_argument("--eigsh-tol", type=float, default=1e-11)
    parser.add_argument("--eigsh-maxiter", type=int, default=50000)
    parser.add_argument("--allow-recompute", action="store_true", help="Allow Hamiltonian recomputation if cache is missing.")
    parser.add_argument("--force-build-payload", action="store_true", help="Use QCANT's full payload builder instead of directly loading the summary cache path.")
    parser.add_argument("--dry-run", action="store_true", help="Build/load the Hamiltonian and report sector metadata without diagonalizing.")
    args = parser.parse_args()
    args.require_cache = not bool(args.allow_recompute)
    return args


def main() -> None:
    args = parse_args()
    result = compute(args)
    case_id = str(result["case_id"])
    output = args.output
    if output is None:
        output = args.outdir / case_id / ("initial_overlap_dry_run.json" if args.dry_run else "initial_overlap.json")
    _write_json(output, result)
    print(f"Wrote {output}")
    for key in (
        "case_id",
        "status",
        "sector_dim",
        "n_alpha",
        "n_beta",
        "hf_bits",
        "sector_ground_energy_hartree",
        "overlap_weight",
        "negative_log10_overlap",
        "solver",
        "elapsed_seconds",
    ):
        if key in result:
            print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
