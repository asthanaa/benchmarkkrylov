#!/usr/bin/env python
"""Build the Fig. 1b-style MolVQE-21 benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ANGSTROM_PER_BOHR = 0.529177210903
CHEMICAL_ACCURACY_HARTREE = 1.6e-3
RENYI_ALPHA = 0.25

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BENCHMARK_ROOT / "dataset" / "zenodo_19051536" / "extracted" / "data_repository"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"


FIELDNAMES = [
    "case_id",
    "source_name",
    "source_json_path",
    "symbols",
    "geometry_bohr",
    "geometry_angstrom",
    "basis",
    "charge",
    "spin",
    "active_electrons",
    "active_orbitals",
    "casscf_energy_hartree",
    "renyi_alpha",
    "h0.25",
    "chemical_accuracy_hartree",
    "adapt_chem_accuracy_iteration",
    "has_adapt_chem_hit",
    "final_num_iterations",
    "final_reason",
]


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _case_id(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return cleaned or "case"


def _renyi_entropy(ci_vector: list[float], alpha: float = RENYI_ALPHA) -> float:
    if alpha == 1.0:
        raise ValueError("alpha=1 requires the Shannon entropy limit")
    probabilities = [float(coeff) ** 2 for coeff in ci_vector]
    if sum(probabilities) <= 0.0:
        raise ValueError("CI vector has zero norm")
    probabilities = [prob for prob in probabilities if prob > 0.0]
    moment = sum(prob**alpha for prob in probabilities)
    return float(math.log(moment) / (1.0 - alpha))


def _adapt_chemical_accuracy_iteration(spans: list[dict[str, Any]], casscf_energy: float) -> int | None:
    for span in spans:
        if span.get("type") != "adapt_iter":
            continue
        attrs = span.get("attrs", {})
        energy = attrs.get("energy_after_vqe")
        iteration = attrs.get("iter")
        if energy is None or iteration is None:
            continue
        if float(energy) - casscf_energy <= CHEMICAL_ACCURACY_HARTREE:
            return int(iteration) + 1
    return None


def _build_row(path: Path) -> dict[str, str | int | float | bool | None]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta = payload["meta"]
    molecule = meta["molecule"]
    casscf = meta["casscf"]
    atoms = molecule["atoms"]

    symbols = [atom["element"] for atom in atoms]
    geometry_bohr = [[float(value) for value in atom["xyz"]] for atom in atoms]
    geometry_angstrom = [
        [float(value) * ANGSTROM_PER_BOHR for value in xyz]
        for xyz in geometry_bohr
    ]

    casscf_energy = float(casscf["energy"])
    adapt_hit = _adapt_chemical_accuracy_iteration(payload.get("spans", []), casscf_energy)
    final = payload.get("final", {})

    return {
        "case_id": _case_id(path.stem),
        "source_name": path.name,
        "source_json_path": path.relative_to(BENCHMARK_ROOT).as_posix(),
        "symbols": _compact_json(symbols),
        "geometry_bohr": _compact_json(geometry_bohr),
        "geometry_angstrom": _compact_json(geometry_angstrom),
        "basis": str(molecule["basis"]),
        "charge": int(molecule["charge"]),
        "spin": int(molecule["spin"]),
        "active_electrons": int(casscf["nelec"]),
        "active_orbitals": int(casscf["ncas"]),
        "casscf_energy_hartree": casscf_energy,
        "renyi_alpha": RENYI_ALPHA,
        "h0.25": _renyi_entropy([float(value) for value in casscf["ci_vector"]]),
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "adapt_chem_accuracy_iteration": "" if adapt_hit is None else adapt_hit,
        "has_adapt_chem_hit": str(adapt_hit is not None).lower(),
        "final_num_iterations": "" if final.get("num_iterations") is None else int(final["num_iterations"]),
        "final_reason": "" if final.get("reason") is None else str(final["reason"]),
    }


def build_manifest(data_root: Path = DATA_ROOT, output: Path = DEFAULT_OUTPUT) -> list[dict[str, Any]]:
    molvqe_root = data_root / "MolVQE-21"
    if not molvqe_root.is_dir():
        raise FileNotFoundError(f"MolVQE-21 directory not found: {molvqe_root}")

    rows = [_build_row(path) for path in sorted(molvqe_root.glob("*.json"))]
    rows.sort(
        key=lambda row: (
            int(row["active_orbitals"]),
            len(json.loads(str(row["symbols"]))),
            int(row["active_electrons"]),
            float(row["h0.25"]),
            str(row["source_name"]),
        )
    )
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id collision while building manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_manifest(args.data_root, args.output)
    adapt_hits = sum(row["has_adapt_chem_hit"] == "true" for row in rows)
    print(f"Wrote {args.output}")
    print(f"MolVQE-21 top-level cases: {len(rows)}")
    print(f"ADAPT chemical-accuracy hits: {adapt_hits}")


if __name__ == "__main__":
    main()
