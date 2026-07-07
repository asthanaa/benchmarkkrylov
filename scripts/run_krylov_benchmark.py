#!/usr/bin/env python
"""Run restartable QRTE/QKUD Qulacs benchmarks from the MolVQE-21 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
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


CHEMICAL_ACCURACY_HARTREE = 1.6e-3
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"
DEFAULT_HAMILTONIAN_CACHE_DIR = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb"
DEFAULT_ACTIVE_ORBITAL_OVERRIDES = BENCHMARK_ROOT / "manifests" / "molvqe21_active_orbital_overrides.json"
AVAS_LABELS_BY_CASE = {
    "Al3_minus": ["Al 3p"],
    "CO_equi_cc_pVTZ_0": ["C 2p", "O 2p"],
    "CO_equi_cc_pVTZ_2": ["C 2p", "O 2p"],
    "FeO": ["Fe 3d", "O 2p"],
    "Ferrocene_ceo": ["Fe 3d"],
    "cyclobutadiene_ceo": ["C 2pz"],
    "feno_2plus": ["Fe 3d", "N 2p", "O 2p"],
    "fenoco3_minus": ["Fe 3d"],
    "feo4_2minus_ceo": ["Fe 3d"],
    "hexatriene_equi_cc_pVTZ_0": ["C 2pz"],
    "hexatriene_equi_cc_pVTZ_2": ["C 2pz"],
    "m_Benzyne_equi_cc_pVTZ_0": ["C 2pz"],
    "m_Benzyne_equi_cc_pVTZ_2": ["C 2pz"],
}


def _set_thread_limits() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_active_orbital_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"active orbital override file must contain a JSON object: {path}")

    overrides: dict[str, dict[str, Any]] = {}
    for case_id, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"active orbital override for {case_id!r} must be a JSON object")
        raw_indices = value.get("active_orbital_indices")
        if raw_indices is None:
            continue
        if not isinstance(raw_indices, list):
            raise ValueError(f"active_orbital_indices for {case_id!r} must be a JSON list")
        indices = [int(index) for index in raw_indices]
        if len(indices) != len(set(indices)):
            raise ValueError(f"active_orbital_indices for {case_id!r} contains duplicates")
        base = int(value.get("active_orbital_index_base", 0))
        if any(index < base for index in indices):
            raise ValueError(f"active_orbital_indices for {case_id!r} are inconsistent with base={base}")
        overrides[str(case_id)] = {
            "active_orbital_indices": indices,
            "active_orbital_index_base": base,
            "selection_method": str(value.get("selection_method", "pyscf_canonical_rohf_mo_indices")),
        }
    return overrides


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _write_history_csv(path: Path, min_history: np.ndarray, casscf_energy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "min_energy_hartree",
                "error_vs_casscf_hartree",
                "within_chemical_accuracy",
            ],
        )
        writer.writeheader()
        for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
            error = float(energy) - casscf_energy
            writer.writerow(
                {
                    "iteration": idx,
                    "min_energy_hartree": float(energy),
                    "error_vs_casscf_hartree": error,
                    "within_chemical_accuracy": str(error <= CHEMICAL_ACCURACY_HARTREE).lower(),
                }
            )


def _first_hit(min_history: np.ndarray, casscf_energy: float) -> int | None:
    for idx, energy in enumerate(np.asarray(min_history, dtype=float), start=1):
        if float(energy) - casscf_energy <= CHEMICAL_ACCURACY_HARTREE:
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


def _active_orbital_labels(case_id: str) -> list[str] | None:
    labels = AVAS_LABELS_BY_CASE.get(case_id)
    return None if labels is None else list(labels)


def _active_orbital_override(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any] | None:
    overrides = getattr(args, "active_orbital_overrides_by_case", {})
    override = overrides.get(row["case_id"])
    if override is None:
        return None

    indices = list(override["active_orbital_indices"])
    active_orbitals = int(row["active_orbitals"])
    if len(indices) != active_orbitals:
        raise ValueError(
            f"active_orbital_indices for {row['case_id']!r} has length {len(indices)}, "
            f"but active_orbitals={active_orbitals}"
        )
    return dict(override)


def _import_qcant(qcant_root: Path):
    sys.path.insert(0, str(qcant_root.resolve()))
    try:
        import qulacs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Qulacs is required for this benchmark; install QCANT with the [qulacs] extra.") from exc
    import QCANT

    return QCANT


def _case_paths(row: dict[str, str], args: argparse.Namespace) -> tuple[Path, Path]:
    case_id = row["case_id"]
    case_dir = args.outdir / case_id
    summary_path = case_dir / "summary.json"
    return case_dir, summary_path


def _prepare_case(row: dict[str, str], args: argparse.Namespace) -> tuple[Path, Path]:
    case_id = row["case_id"]
    case_dir, summary_path = _case_paths(row, args)
    if _completed(summary_path) and not args.force:
        return case_dir, summary_path

    case_dir.mkdir(parents=True, exist_ok=True)
    symbols = _json_loads(row["symbols"])
    geometry_angstrom = np.asarray(_json_loads(row["geometry_angstrom"]), dtype=float)
    casscf_energy = float(row["casscf_energy_hartree"])
    active_orbital_labels = _active_orbital_labels(case_id)
    active_orbital_override = _active_orbital_override(row, args)
    active_orbital_indices = None if active_orbital_override is None else active_orbital_override["active_orbital_indices"]
    active_orbital_index_base = None if active_orbital_override is None else active_orbital_override["active_orbital_index_base"]
    active_orbital_selection_method = None if active_orbital_override is None else active_orbital_override["selection_method"]

    input_snapshot = {
        "case_id": case_id,
        "source_name": row["source_name"],
        "source_json_path": row["source_json_path"],
        "symbols": symbols,
        "geometry_bohr": _json_loads(row["geometry_bohr"]),
        "geometry_angstrom": geometry_angstrom.tolist(),
        "basis": row["basis"],
        "charge": int(row["charge"]),
        "spin": int(row["spin"]),
        "active_electrons": int(row["active_electrons"]),
        "active_orbitals": int(row["active_orbitals"]),
        "casscf_energy_hartree": casscf_energy,
        "h0.25": float(row["h0.25"]),
        "active_orbital_labels": active_orbital_labels,
        "active_orbital_indices": active_orbital_indices,
        "active_orbital_index_base": active_orbital_index_base,
        "active_orbital_selection_method": active_orbital_selection_method,
        "active_orbital_override": active_orbital_override,
        "active_orbital_overrides_path": str(args.active_orbital_overrides),
        "adapt_chem_accuracy_iteration": _int_or_none(row["adapt_chem_accuracy_iteration"]),
        "has_adapt_chem_hit": row["has_adapt_chem_hit"] == "true",
    }
    run_config = {
        "method": args.method,
        "n_steps": int(args.n_steps),
        "delta_t": float(args.delta_t),
        "epsilon": float(args.epsilon),
        "evolution_mode": args.evolution_mode,
        "trotter_steps": int(args.trotter_steps),
        "max_workers": int(args.max_workers),
        "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
        "early_stop_energy_hartree": casscf_energy + CHEMICAL_ACCURACY_HARTREE if args.early_stop_chemical_accuracy else None,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "qcant_root": str(args.qcant_root.resolve()),
        "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir.resolve()),
        "hamiltonian_source": args.hamiltonian_source,
        "active_orbital_labels": active_orbital_labels,
        "active_orbital_indices": active_orbital_indices,
        "active_orbital_index_base": active_orbital_index_base,
        "active_orbital_selection_method": active_orbital_selection_method,
        "active_orbital_override": active_orbital_override,
        "active_orbital_overrides_path": str(args.active_orbital_overrides),
    }

    _write_json(case_dir / "input_snapshot.json", input_snapshot)
    _write_json(case_dir / "run_config.json", run_config)
    return case_dir, summary_path


def _run_case_body(row: dict[str, str], args: argparse.Namespace, QCANT) -> str:
    case_id = row["case_id"]
    case_dir, summary_path = _prepare_case(row, args)
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed {args.method} {case_id}", flush=True)
        return "skipped"

    symbols = _json_loads(row["symbols"])
    geometry_angstrom = np.asarray(_json_loads(row["geometry_angstrom"]), dtype=float)
    casscf_energy = float(row["casscf_energy_hartree"])
    active_orbital_labels = _active_orbital_labels(case_id)
    active_orbital_override = _active_orbital_override(row, args)
    active_orbital_indices = None if active_orbital_override is None else active_orbital_override["active_orbital_indices"]
    active_orbital_index_base = 0 if active_orbital_override is None else active_orbital_override["active_orbital_index_base"]
    active_orbital_selection_method = None if active_orbital_override is None else active_orbital_override["selection_method"]
    early_stop_energy = casscf_energy + CHEMICAL_ACCURACY_HARTREE if args.early_stop_chemical_accuracy else None
    start = time.perf_counter()
    try:
        if args.method == "qrte":
            energies, _basis_states, _times, min_history = QCANT.qrte_qulacs(
                symbols=symbols,
                geometry=geometry_angstrom,
                delta_t=float(args.delta_t),
                n_steps=int(args.n_steps),
                active_electrons=int(row["active_electrons"]),
                active_orbitals=int(row["active_orbitals"]),
                basis=row["basis"],
                charge=int(row["charge"]),
                spin=int(row["spin"]),
                return_min_energy_history=True,
                evolution_mode=args.evolution_mode,
                trotter_steps=int(args.trotter_steps),
                max_workers=int(args.max_workers),
                early_stop_energy=early_stop_energy,
                hamiltonian_cache_dir=None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
                hamiltonian_source=args.hamiltonian_source,
                active_orbital_labels=active_orbital_labels,
                active_orbital_indices=active_orbital_indices,
                active_orbital_index_base=active_orbital_index_base,
            )
        else:
            energies, _basis_states, min_history = QCANT.qkud_qulacs(
                symbols=symbols,
                geometry=geometry_angstrom,
                epsilon=float(args.epsilon),
                n_steps=int(args.n_steps),
                active_electrons=int(row["active_electrons"]),
                active_orbitals=int(row["active_orbitals"]),
                basis=row["basis"],
                charge=int(row["charge"]),
                spin=int(row["spin"]),
                return_min_energy_history=True,
                evolution_mode=args.evolution_mode,
                trotter_steps=int(args.trotter_steps),
                max_workers=int(args.max_workers),
                early_stop_energy=early_stop_energy,
                hamiltonian_cache_dir=None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
                hamiltonian_source=args.hamiltonian_source,
                active_orbital_labels=active_orbital_labels,
                active_orbital_indices=active_orbital_indices,
                active_orbital_index_base=active_orbital_index_base,
            )

        min_history = np.asarray(min_history, dtype=float)
        energies = np.asarray(energies, dtype=float)
        hit_iteration = _first_hit(min_history, casscf_energy)
        n_steps_completed = int(min_history.size)
        stopped_early = bool(args.early_stop_chemical_accuracy and hit_iteration is not None and n_steps_completed < int(args.n_steps))
        _write_history_csv(case_dir / "min_energy_history.csv", min_history, casscf_energy)

        summary = {
            "status": "completed",
            "case_id": case_id,
            "method": args.method,
            "h0.25": float(row["h0.25"]),
            "casscf_energy_hartree": casscf_energy,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "hit_iteration": hit_iteration,
            "hit_chemical_accuracy": hit_iteration is not None,
            "n_steps": n_steps_completed,
            "n_steps_requested": int(args.n_steps),
            "n_steps_completed": n_steps_completed,
            "stopped_early": stopped_early,
            "early_stop_energy_hartree": early_stop_energy,
            "final_projected_ground_energy_hartree": float(energies[0]) if energies.size else None,
            "final_min_history_energy_hartree": float(min_history[-1]) if min_history.size else None,
            "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
            "hamiltonian_source": args.hamiltonian_source,
            "active_orbital_labels": active_orbital_labels,
            "active_orbital_indices": active_orbital_indices,
            "active_orbital_index_base": active_orbital_index_base if active_orbital_indices is not None else None,
            "active_orbital_selection_method": active_orbital_selection_method,
            "active_orbital_override": active_orbital_override,
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(f"DONE {args.method} {case_id} hit={hit_iteration}", flush=True)
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case_id,
            "method": args.method,
            "h0.25": float(row["h0.25"]),
            "casscf_energy_hartree": casscf_energy,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "n_steps": int(args.n_steps),
            "n_steps_requested": int(args.n_steps),
            "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
            "early_stop_energy_hartree": early_stop_energy,
            "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
            "hamiltonian_source": args.hamiltonian_source,
            "active_orbital_labels": active_orbital_labels,
            "active_orbital_indices": active_orbital_indices,
            "active_orbital_index_base": active_orbital_index_base if active_orbital_indices is not None else None,
            "active_orbital_selection_method": active_orbital_selection_method,
            "active_orbital_override": active_orbital_override,
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL {args.method} {case_id}: {exc!r}", flush=True)
        return "failed"


def _run_case_parent(row: dict[str, str], args: argparse.Namespace) -> str:
    case_id = row["case_id"]
    case_dir, summary_path = _prepare_case(row, args)
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed {args.method} {case_id}", flush=True)
        return "skipped"

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
        case_id,
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
        "--max-workers",
        str(args.max_workers),
        "--hamiltonian-source",
        args.hamiltonian_source,
        "--active-orbital-overrides",
        str(args.active_orbital_overrides),
    ]
    if args.hamiltonian_cache_dir is not None:
        cmd.extend(["--hamiltonian-cache-dir", str(args.hamiltonian_cache_dir)])
    if args.early_stop_chemical_accuracy:
        cmd.append("--early-stop-chemical-accuracy")
    if args.force:
        cmd.append("--force")

    with (case_dir / "stdout.log").open("w", encoding="utf-8") as stdout_handle:
        with (case_dir / "stderr.log").open("w", encoding="utf-8") as stderr_handle:
            result = subprocess.run(cmd, stdout=stdout_handle, stderr=stderr_handle, text=True, check=False)

    if result.returncode == 0 and _completed(summary_path):
        print(f"DONE {args.method} {case_id}", flush=True)
        return "completed"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                status = json.load(handle).get("status")
            if status == "failed":
                print(f"FAIL {args.method} {case_id}; see {summary_path}", flush=True)
                return "failed"
        except json.JSONDecodeError:
            pass

    failure = {
        "status": "failed",
        "case_id": case_id,
        "method": args.method,
        "h0.25": float(row["h0.25"]),
        "casscf_energy_hartree": float(row["casscf_energy_hartree"]),
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "n_steps": int(args.n_steps),
        "n_steps_requested": int(args.n_steps),
        "early_stop_chemical_accuracy": bool(args.early_stop_chemical_accuracy),
        "hamiltonian_cache_dir": None if args.hamiltonian_cache_dir is None else str(args.hamiltonian_cache_dir),
        "hamiltonian_source": args.hamiltonian_source,
        "active_orbital_labels": _active_orbital_labels(case_id),
        "active_orbital_override": _active_orbital_override(row, args),
        "error": f"single-case subprocess exited with code {result.returncode}",
    }
    _write_json(summary_path, failure)
    print(f"FAIL {args.method} {case_id}; subprocess rc={result.returncode}", flush=True)
    return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qrte", "qkud"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--row-index", type=int, default=None, help="Run one zero-based manifest row index.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--evolution-mode", choices=("sparse", "trotter"), default="trotter")
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=36)
    parser.add_argument("--hamiltonian-cache-dir", type=Path, default=DEFAULT_HAMILTONIAN_CACHE_DIR)
    parser.add_argument("--active-orbital-overrides", type=Path, default=DEFAULT_ACTIVE_ORBITAL_OVERRIDES)
    parser.add_argument(
        "--hamiltonian-source",
        choices=("casci", "casscf", "casscf_natorb", "casscf-natural", "molecular"),
        default="casscf_natorb",
    )
    parser.add_argument("--early-stop-chemical-accuracy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--single-case", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.hamiltonian_cache_dir is not None:
        args.hamiltonian_cache_dir.mkdir(parents=True, exist_ok=True)
    args.active_orbital_overrides_by_case = _load_active_orbital_overrides(args.active_orbital_overrides)
    rows = _load_manifest(args.manifest)
    if args.row_index is not None:
        if args.row_index < 0 or args.row_index >= len(rows):
            raise SystemExit(f"--row-index {args.row_index} outside manifest range 0-{len(rows) - 1}")
        rows = [rows[args.row_index]]
    if args.case_id:
        allowed = set(args.case_id)
        rows = [row for row in rows if row["case_id"] in allowed]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise SystemExit("No manifest rows selected")

    counts = {"completed": 0, "skipped": 0, "failed": 0}
    if args.single_case:
        if len(rows) != 1:
            raise SystemExit("--single-case requires exactly one selected --case-id")
        QCANT = _import_qcant(args.qcant_root)
        status = _run_case_body(rows[0], args, QCANT)
        counts[status] = counts.get(status, 0) + 1
        print(json.dumps(counts, sort_keys=True), flush=True)
        if status == "failed":
            raise SystemExit(1)
        return

    for row in rows:
        status = _run_case_parent(row, args)
        counts[status] = counts.get(status, 0) + 1

    print(json.dumps(counts, sort_keys=True), flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
