#!/usr/bin/env python
"""Plot QRTE and QKUD convergence histories for one benchmark case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "benchmarkkrylov-matplotlib"))
os.environ["XDG_CACHE_HOME"] = str(TMPDIR / "benchmarkkrylov-xdg-cache")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QRTE_RUN_DIR = BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.05"
DEFAULT_QKUD_RUN_DIR = BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.05"
DEFAULT_PLOT_DIR = BENCHMARK_ROOT / "plots"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "case"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_history(path: Path) -> tuple[list[int], list[float], list[float]]:
    iterations: list[int] = []
    energies: list[float] = []
    errors: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            iterations.append(int(row["iteration"]))
            energies.append(float(row["min_energy_hartree"]))
            errors.append(abs(float(row["error_vs_casscf_hartree"])))
    return iterations, energies, errors


def _dataset_reference_energy(case_dir: Path) -> float:
    snapshot_path = case_dir / "input_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing input snapshot file: {snapshot_path}")
    snapshot = _load_json(snapshot_path)
    source_path = BENCHMARK_ROOT / snapshot["source_json_path"]
    source = _load_json(source_path)
    return float(source["meta"]["casscf"]["energy"])


def _local_hamiltonian_reference_energy(case_dir: Path) -> float:
    snapshot_path = case_dir / "input_snapshot.json"
    config_path = case_dir / "run_config.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing input snapshot file: {snapshot_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config file: {config_path}")
    snapshot = _load_json(snapshot_path)
    config = _load_json(config_path)

    hamiltonian_source = str(config.get("hamiltonian_source", "casscf_natorb"))
    if hamiltonian_source == "casscf-natural":
        hamiltonian_source = "casscf_natorb"
    cache_dir_raw = config.get("hamiltonian_cache_dir")
    if cache_dir_raw is None:
        cache_dir = BENCHMARK_ROOT / "hamiltonian_cache" / "casscf_natorb"
    else:
        cache_dir = Path(str(cache_dir_raw))
        if not cache_dir.is_absolute():
            cache_dir = BENCHMARK_ROOT / cache_dir

    active_orbital_labels = config.get("active_orbital_labels")
    if active_orbital_labels is None:
        active_orbital_labels = snapshot.get("active_orbital_labels") or []
    payload = {
        "active_electrons": int(snapshot["active_electrons"]),
        "active_orbitals": int(snapshot["active_orbitals"]),
        "basis": str(snapshot["basis"]),
        "charge": int(snapshot["charge"]),
        "geometry_angstrom": [
            [round(float(value), 12) for value in xyz]
            for xyz in snapshot["geometry_angstrom"]
        ],
        "hamiltonian_cutoff": 1e-20,
        "hamiltonian_source": hamiltonian_source,
        "active_orbital_labels": [str(label) for label in active_orbital_labels],
        "spin": int(snapshot["spin"]),
        "symbols": [str(symbol) for symbol in snapshot["symbols"]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_path = cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing Hamiltonian cache file: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as payload_npz:
        return float(payload_npz["reference_energy"])


def _load_case(run_dir: Path, case_id: str, reference_source: str) -> dict[str, Any]:
    case_dir = run_dir / case_id
    history_path = case_dir / "min_energy_history.csv"
    summary_path = case_dir / "summary.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    iterations, energies, errors = _load_history(history_path)
    summary = _load_json(summary_path)
    if reference_source == "dataset-json":
        reference_energy = _dataset_reference_energy(case_dir)
        errors = [abs(energy - reference_energy) for energy in energies]
        reference_label = "MolVQE JSON CASSCF/natorb"
    elif reference_source == "local-hamiltonian":
        reference_energy = _local_hamiltonian_reference_energy(case_dir)
        errors = [abs(energy - reference_energy) for energy in energies]
        reference_label = "local cached Hamiltonian CASSCF/natorb"
    elif reference_source == "summary":
        reference_energy = float(summary["casscf_energy_hartree"])
        errors = [abs(energy - reference_energy) for energy in energies]
        reference_label = "summary CASSCF"
    elif reference_source == "history":
        reference_energy = float(summary["casscf_energy_hartree"])
        reference_label = "history CSV CASSCF"
    else:
        raise ValueError(f"unknown reference source: {reference_source}")
    return {
        "iterations": iterations,
        "energies": energies,
        "errors": errors,
        "reference_energy": reference_energy,
        "reference_label": reference_label,
        "summary": summary,
        "history_path": history_path,
        "summary_path": summary_path,
    }


def _series_label(method_label: str, summary: dict[str, Any]) -> str:
    hit_iteration = summary.get("hit_iteration")
    if summary.get("hit_chemical_accuracy") and hit_iteration is not None:
        return f"{method_label} (hit at k={hit_iteration})"
    return f"{method_label} (no hit)"


def _plot(
    *,
    case_id: str,
    qrte: dict[str, Any],
    qkud: dict[str, Any],
    output_png: Path,
    output_pdf: Path | None,
    qrte_label: str,
    qkud_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        qrte["iterations"],
        qrte["errors"],
        marker="o",
        markersize=4.0,
        linewidth=1.8,
        color="#0072B2",
        label=_series_label(qrte_label, qrte["summary"]),
    )
    ax.plot(
        qkud["iterations"],
        qkud["errors"],
        marker="s",
        markersize=3.8,
        linewidth=1.8,
        color="#D55E00",
        label=_series_label(qkud_label, qkud["summary"]),
    )

    chemical_accuracy = float(
        qrte["summary"].get(
            "chemical_accuracy_hartree",
            qkud["summary"].get("chemical_accuracy_hartree", 0.0016),
        )
    )
    ax.axhline(
        chemical_accuracy,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"chemical accuracy ({chemical_accuracy:g} Ha)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Krylov iteration")
    ax.set_ylabel(f"|E_min(k) - E_CASSCF| (Hartree)\nreference: {qrte['reference_label']}")
    ax.set_title(f"{case_id}: QRTE vs QKUD convergence")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    if output_pdf is not None:
        fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", help="Benchmark case directory name, e.g. FeO.")
    parser.add_argument("--qrte-run-dir", type=Path, default=DEFAULT_QRTE_RUN_DIR)
    parser.add_argument("--qkud-run-dir", type=Path, default=DEFAULT_QKUD_RUN_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--output-stem",
        help="Output basename without extension. Defaults to a case/run-derived name.",
    )
    parser.add_argument(
        "--reference-source",
        choices=("history", "summary", "dataset-json", "local-hamiltonian"),
        default="history",
        help="Reference energy used to compute plotted errors.",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG output.")
    args = parser.parse_args()

    qrte = _load_case(args.qrte_run_dir, args.case_id, args.reference_source)
    qkud = _load_case(args.qkud_run_dir, args.case_id, args.reference_source)

    stem = args.output_stem
    if stem is None:
        stem = (
            f"{_slugify(args.case_id)}_"
            f"{_slugify(args.qrte_run_dir.name)}_"
            f"{_slugify(args.qkud_run_dir.name)}_convergence"
        )
    output_png = args.plot_dir / f"{stem}.png"
    output_pdf = None if args.no_pdf else args.plot_dir / f"{stem}.pdf"

    _plot(
        case_id=args.case_id,
        qrte=qrte,
        qkud=qkud,
        output_png=output_png,
        output_pdf=output_pdf,
        qrte_label=args.qrte_run_dir.name,
        qkud_label=args.qkud_run_dir.name,
    )

    print(f"Wrote {output_png}")
    if output_pdf is not None:
        print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
