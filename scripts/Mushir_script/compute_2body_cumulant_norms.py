#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Compute active-space two-body cumulant Frobenius norms from cached Hamiltonians."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

import numpy as np


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION_CSV = (
    BENCHMARK_ROOT / "plots" / "qkud_eps0p05_vs_qrte_dt0p05_iterations_vs_entropy.csv"
)
DEFAULT_OVERLAP_DIR = BENCHMARK_ROOT / "runs" / "initial_state_overlap"
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "two_body_cumulant_norms" / "qkud_eps0p05_converged"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb"

FIELDNAMES = [
    "case_id",
    "h0.25",
    "qkud_eps0p05_chem_accuracy_iteration",
    "qrte_dt0p05_chem_accuracy_iteration",
    "qkud_only_hit",
    "cumulant_2rdm_frobenius_norm_squared",
    "cumulant_2rdm_frobenius_norm",
    "fci_energy_hartree",
    "reference_energy_hartree",
    "fci_minus_reference_hartree",
    "reference_converged",
    "n_spatial_orbitals",
    "n_alpha",
    "n_beta",
    "sector_dim",
    "gamma_trace",
    "two_rdm_trace",
    "max_rdm_consistency_error",
    "hamiltonian_cache_path",
    "summary_path",
    "elapsed_seconds",
]


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


def _load_selection_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.selection_csv.exists():
        raise FileNotFoundError(f"Selection CSV not found: {args.selection_csv}")

    rows: list[dict[str, Any]] = []
    with args.selection_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("qkud_hit_chemical_accuracy") != "true":
                continue
            if args.case_id and row["case_id"] not in set(args.case_id):
                continue
            rows.append(dict(row))

    rows.sort(key=lambda row: (float(row.get("h0.25") or 0.0), row["case_id"]))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    return rows


def _read_existing_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "completed":
        return None
    return payload


def _hash_cache_path_from_run_files(summary_path: Path) -> Path:
    input_snapshot_path = summary_path.parent / "input_snapshot.json"
    run_config_path = summary_path.parent / "run_config.json"
    if not input_snapshot_path.exists():
        raise FileNotFoundError(f"Missing input snapshot: {input_snapshot_path}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")

    snapshot = _load_json(input_snapshot_path)
    config = _load_json(run_config_path)
    cache_dir = _resolve_path(config.get("hamiltonian_cache_dir")) or DEFAULT_HAMILTONIAN_CACHE_DIR
    hamiltonian_source = str(config.get("hamiltonian_source", "casscf_natorb"))
    if hamiltonian_source == "casscf-natural":
        hamiltonian_source = "casscf_natorb"

    active_orbital_labels = config.get("active_orbital_labels")
    if active_orbital_labels is None:
        active_orbital_labels = snapshot.get("active_orbital_labels") or []

    payload: dict[str, Any] = {
        "active_electrons": int(snapshot["active_electrons"]),
        "active_orbitals": int(snapshot["active_orbitals"]),
        "basis": str(snapshot["basis"]),
        "charge": int(snapshot["charge"]),
        "geometry_angstrom": np.asarray(snapshot["geometry_angstrom"], dtype=float).round(12).tolist(),
        "hamiltonian_cutoff": 1e-20,
        "hamiltonian_source": hamiltonian_source,
        "active_orbital_labels": [str(label) for label in active_orbital_labels],
        "spin": int(snapshot["spin"]),
        "symbols": [str(symbol) for symbol in snapshot["symbols"]],
    }

    active_orbital_indices = config.get("active_orbital_indices")
    if active_orbital_indices is None:
        active_orbital_indices = snapshot.get("active_orbital_indices")
    if active_orbital_indices not in (None, []):
        payload["active_orbital_indices"] = [int(index) for index in active_orbital_indices]
        payload["active_orbital_index_base"] = int(
            config.get("active_orbital_index_base", snapshot.get("active_orbital_index_base", 0))
        )

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.npz"


def _cache_path_from_overlap(case_id: str, overlap_dir: Path) -> Path | None:
    overlap_path = overlap_dir / case_id / "initial_overlap.json"
    if not overlap_path.exists():
        return None
    overlap = _load_json(overlap_path)
    if overlap.get("status") != "completed":
        return None
    cache_path = _resolve_path(overlap.get("hamiltonian_cache_path"))
    if cache_path is None:
        return None
    return cache_path


def _resolve_cache_path(row: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    case_id = str(row["case_id"])
    summary_path = _resolve_path(row.get("qkud_summary_path"))
    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(f"Missing QKUD summary for {case_id}: {summary_path}")

    if args.prefer_overlap_cache:
        cache_path = _cache_path_from_overlap(case_id, args.overlap_dir)
        if cache_path is not None:
            return cache_path, summary_path

    summary = _load_json(summary_path)
    for payload in (summary, _load_json(summary_path.parent / "input_snapshot.json")):
        cache_path = _resolve_path(payload.get("hamiltonian_cache_path"))
        if cache_path is not None:
            return cache_path, summary_path

    cache_path = _hash_cache_path_from_run_files(summary_path)
    return cache_path, summary_path


def _load_hamiltonian_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hamiltonian cache file not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {
            "ecore": float(payload["ecore"]),
            "effective_active_electrons": int(payload["effective_active_electrons"]),
            "h1ecas": np.asarray(payload["h1ecas"], dtype=float),
            "hamiltonian_source": str(payload["hamiltonian_source"].item()),
            "ncas": int(payload["ncas"]),
            "nelecas": tuple(int(value) for value in np.asarray(payload["nelecas"], dtype=int).tolist()),
            "reference_energy": float(payload["reference_energy"]),
            "reference_converged": bool(payload["reference_converged"]),
            "two_mo": np.asarray(payload["two_mo"], dtype=float),
        }


def _sector_dim(norb: int, n_alpha: int, n_beta: int) -> int:
    return int(math.comb(int(norb), int(n_alpha)) * math.comb(int(norb), int(n_beta)))


def build_spinorbital_rdms_from_pyscf(
    dm1a: np.ndarray,
    dm1b: np.ndarray,
    dm2aa: np.ndarray,
    dm2ab: np.ndarray,
    dm2bb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gamma[p,q]=<p^dag q> and Gamma[p,q,r,s]=<p^dag q^dag s r>.

    PySCF's make_rdm12s documents dm1[p,q]=<q^dag p> and
    dm2[p,q,r,s]=<p^dag r^dag s q>.  The transposes below put the arrays into
    the cumulant convention and fill all mixed-spin blocks by antisymmetry.
    """

    n = int(dm1a.shape[0])
    nso = 2 * n
    alpha = slice(0, n)
    beta = slice(n, nso)

    gamma = np.zeros((nso, nso), dtype=np.result_type(dm1a, dm1b))
    Gamma = np.zeros((nso, nso, nso, nso), dtype=np.result_type(dm2aa, dm2ab, dm2bb))

    gamma[alpha, alpha] = dm1a.T
    gamma[beta, beta] = dm1b.T

    to_cumulant_order = lambda dm2: dm2.transpose(0, 2, 1, 3)
    Gamma[alpha, alpha, alpha, alpha] = to_cumulant_order(dm2aa)
    Gamma[beta, beta, beta, beta] = to_cumulant_order(dm2bb)

    gamma_ab = to_cumulant_order(dm2ab)
    Gamma[alpha, beta, alpha, beta] = gamma_ab
    Gamma[alpha, beta, beta, alpha] = -gamma_ab.transpose(0, 1, 3, 2)
    Gamma[beta, alpha, alpha, beta] = -gamma_ab.transpose(1, 0, 2, 3)
    Gamma[beta, alpha, beta, alpha] = gamma_ab.transpose(1, 0, 3, 2)
    return gamma, Gamma


def cumulant_2rdm_norm_from_cache(
    cache_path: Path,
    *,
    conv_tol: float,
    max_cycle: int,
) -> dict[str, Any]:
    from pyscf import fci  # noqa: PLC0415

    integral_payload = _load_hamiltonian_cache(cache_path)
    norb = int(integral_payload["ncas"])
    nelecas = tuple(int(value) for value in integral_payload["nelecas"])
    if len(nelecas) != 2:
        raise ValueError(f"Expected spin-resolved nelecas, got {nelecas!r}")
    n_alpha, n_beta = nelecas

    h1 = np.asarray(integral_payload["h1ecas"], dtype=float)
    # QCANT stores two-electron integrals after a swap for PennyLane's convention.
    # PySCF FCI expects the restored PySCF ordering.
    eri = np.swapaxes(np.asarray(integral_payload["two_mo"], dtype=float), 1, 3)
    ecore = float(integral_payload["ecore"])

    start = time.perf_counter()
    energy, ci = fci.direct_spin1.kernel(
        h1,
        eri,
        norb,
        nelecas,
        ecore=ecore,
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
    )
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_spin1.make_rdm12s(ci, norb, nelecas)
    gamma, Gamma = build_spinorbital_rdms_from_pyscf(dm1a, dm1b, dm2aa, dm2ab, dm2bb)

    disconnected = (
        np.einsum("pr,qs->pqrs", gamma, gamma)
        - np.einsum("ps,qr->pqrs", gamma, gamma)
    )
    Delta = Gamma - disconnected
    norm2 = float(np.real(np.vdot(Delta, Delta)))

    n_electrons = int(n_alpha + n_beta)
    gamma_trace = float(np.real(np.trace(gamma)))
    two_trace = float(np.real(np.einsum("pqpq->", Gamma)))
    max_consistency_error = max(
        abs(gamma_trace - n_electrons),
        abs(two_trace - n_electrons * (n_electrons - 1)),
        float(np.max(np.abs(Gamma + Gamma.swapaxes(0, 1)))),
        float(np.max(np.abs(Gamma + Gamma.swapaxes(2, 3)))),
    )

    return {
        "status": "completed",
        "solver": "pyscf-fci",
        "fci_energy_hartree": float(energy),
        "reference_energy_hartree": float(integral_payload["reference_energy"]),
        "fci_minus_reference_hartree": float(energy - float(integral_payload["reference_energy"])),
        "reference_converged": bool(integral_payload["reference_converged"]),
        "hamiltonian_source": str(integral_payload["hamiltonian_source"]),
        "n_spatial_orbitals": int(norb),
        "qubits": int(2 * norb),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "sector_dim": _sector_dim(norb, n_alpha, n_beta),
        "gamma_trace": gamma_trace,
        "two_rdm_trace": two_trace,
        "max_rdm_consistency_error": float(max_consistency_error),
        "cumulant_2rdm_frobenius_norm_squared": norm2,
        "cumulant_2rdm_frobenius_norm": float(math.sqrt(max(norm2, 0.0))),
        "conv_tol": float(conv_tol),
        "max_cycle": int(max_cycle),
        "elapsed_seconds": float(time.perf_counter() - start),
    }


def _row_float_or_empty(row: dict[str, Any], key: str) -> float | str:
    value = row.get(key, "")
    if value in ("", None):
        return ""
    return float(value)


def _result_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    return {field: result.get(field, "") for field in FIELDNAMES}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_result_for_csv(row) for row in rows)
    tmp_path.replace(path)


def _run_self_test() -> None:
    from pyscf import fci  # noqa: PLC0415

    for norb, nelecas in ((3, (2, 1)), (4, (2, 2))):
        na = len(fci.cistring.gen_strings4orblist(range(norb), nelecas[0]))
        nb = len(fci.cistring.gen_strings4orblist(range(norb), nelecas[1]))
        ci = np.zeros((na, nb))
        ci[0, 0] = 1.0
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_spin1.make_rdm12s(ci, norb, nelecas)
        gamma, Gamma = build_spinorbital_rdms_from_pyscf(dm1a, dm1b, dm2aa, dm2ab, dm2bb)
        disconnected = (
            np.einsum("pr,qs->pqrs", gamma, gamma)
            - np.einsum("ps,qr->pqrs", gamma, gamma)
        )
        norm2 = float(np.real(np.vdot(Gamma - disconnected, Gamma - disconnected)))
        if norm2 > 1e-12:
            raise RuntimeError(f"Slater determinant self-test failed for norb={norb}: {norm2}")
    print("self-test passed: Slater determinant cumulant norm is zero")


def compute(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _load_selection_rows(args)
    if args.expected_cases is not None and args.limit is None and not args.case_id:
        if len(rows) != int(args.expected_cases):
            raise SystemExit(f"Expected {args.expected_cases} selected cases, found {len(rows)}")
    if not rows:
        raise SystemExit("No selected cases")

    results: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["case_id"])
        case_output = args.outdir / case_id / "cumulant_2body.json"
        existing = None if args.force else _read_existing_result(case_output)
        if existing is not None:
            print(f"SKIP {case_id}: {case_output}")
            results.append(existing)
            continue

        cache_path, summary_path = _resolve_cache_path(row, args)
        print(f"START {case_id}: {cache_path}", flush=True)
        start = time.perf_counter()
        try:
            result = cumulant_2rdm_norm_from_cache(
                cache_path,
                conv_tol=float(args.conv_tol),
                max_cycle=int(args.max_cycle),
            )
            result.update(
                {
                    "case_id": case_id,
                    "h0.25": _row_float_or_empty(row, "h0.25"),
                    "qkud_eps0p05_chem_accuracy_iteration": _row_float_or_empty(
                        row, "qkud_eps0p05_chem_accuracy_iteration"
                    ),
                    "qrte_dt0p05_chem_accuracy_iteration": _row_float_or_empty(
                        row, "qrte_dt0p05_chem_accuracy_iteration"
                    ),
                    "qkud_only_hit": row.get("qkud_only_hit", ""),
                    "selection_csv": _relative(args.selection_csv),
                    "summary_path": _relative(summary_path),
                    "hamiltonian_cache_path": _relative(cache_path),
                    "wall_elapsed_seconds": float(time.perf_counter() - start),
                }
            )
            _write_json(case_output, result)
            results.append(result)
            print(
                f"DONE {case_id}: ||Delta_2||_F^2="
                f"{result['cumulant_2rdm_frobenius_norm_squared']:.12g}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "status": "failed",
                "case_id": case_id,
                "selection_csv": _relative(args.selection_csv),
                "summary_path": _relative(summary_path),
                "hamiltonian_cache_path": _relative(cache_path),
                "error": repr(exc),
                "wall_elapsed_seconds": float(time.perf_counter() - start),
            }
            _write_json(case_output, failure)
            print(f"FAIL {case_id}: {exc!r}", flush=True)
            if not args.keep_going:
                raise

    results.sort(key=lambda item: (float(item.get("h0.25") or 0.0), str(item["case_id"])))
    _write_csv(args.output_csv, results)
    print(f"Wrote {args.output_csv}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--overlap-dir", type=Path, default=DEFAULT_OVERLAP_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-cases", type=int, default=27)
    parser.add_argument("--conv-tol", type=float, default=1e-12)
    parser.add_argument("--max-cycle", type=int, default=300)
    parser.add_argument("--prefer-overlap-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.output_csv is None:
        args.output_csv = args.outdir / "two_body_cumulant_norms.csv"
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        _run_self_test()
        if args.case_id is None and args.limit is None and not args.force:
            return
    compute(args)


if __name__ == "__main__":
    main()
