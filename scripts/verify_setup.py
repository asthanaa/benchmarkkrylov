#!/usr/bin/env python
"""Verify staged data, manifest, and benchmark prerequisites."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


EXPECTED_MD5 = "942d188ea71104a04649ba0d09232d62"
EXPECTED_JSON_COUNT = 101
EXPECTED_MOLVQE21_TOP_LEVEL_COUNT = 29
EXPECTED_KNOWN_HITS = {
    "H4 near equi._sto3g_0.json": "3",
    "H8 near equi._sto3g_0.json": "55",
    "H9 near equi._sto3g_1.json": "142",
    "H10 near equi._sto3g_0.json": "",
}
ANGSTROM_PER_BOHR = 0.529177210903
RENYI_ALPHA = 0.25

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
RAW_ARCHIVE = BENCHMARK_ROOT / "dataset" / "zenodo_19051536" / "raw" / "data_repository.7z"
DATA_ROOT = BENCHMARK_ROOT / "dataset" / "zenodo_19051536" / "extracted" / "data_repository"
MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
QCANT_ROOT = BENCHMARK_ROOT / "vendor" / "QCANT"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"OK {message}")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _renyi_entropy(ci_vector: list[float]) -> float:
    probabilities = [float(coeff) ** 2 for coeff in ci_vector]
    probabilities = [prob for prob in probabilities if prob > 0.0]
    moment = sum(prob**RENYI_ALPHA for prob in probabilities)
    return float(math.log(moment) / (1.0 - RENYI_ALPHA))


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verify_entropy_overlap() -> None:
    checked = 0
    max_diff = 0.0
    for path in DATA_ROOT.rglob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        casscf = payload.get("meta", {}).get("casscf", {})
        stored = casscf.get("renyi_entropy_0.25")
        ci_vector = casscf.get("ci_vector")
        if stored is None or ci_vector is None:
            continue
        computed = _renyi_entropy([float(value) for value in ci_vector])
        max_diff = max(max_diff, abs(computed - float(stored)))
        checked += 1
    _check(checked > 0, "entropy overlap records are present")
    _check(max_diff < 1e-3, f"computed h0.25 matches stored values within JSON rounding; max diff {max_diff:.3e}")


def _verify_imports() -> None:
    sys.path.insert(0, str(QCANT_ROOT.resolve()))
    import QCANT  # noqa: F401
    import qulacs  # noqa: F401

    _check(True, "QCANT and qulacs import successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-imports", action="store_true")
    args = parser.parse_args()

    _check(RAW_ARCHIVE.exists(), "raw Zenodo archive exists")
    _check(_md5(RAW_ARCHIVE) == EXPECTED_MD5, "raw Zenodo archive md5 matches")
    _check(DATA_ROOT.is_dir(), "extracted data_repository exists")
    _check(sum(1 for _ in DATA_ROOT.rglob("*.json")) == EXPECTED_JSON_COUNT, "full extract has 101 JSON files")
    _check(
        sum(1 for _ in (DATA_ROOT / "MolVQE-21").glob("*.json")) == EXPECTED_MOLVQE21_TOP_LEVEL_COUNT,
        "MolVQE-21 top-level set has 29 JSON files",
    )

    rows = _load_manifest(MANIFEST)
    _check(len(rows) == EXPECTED_MOLVQE21_TOP_LEVEL_COUNT, "manifest has 29 rows")
    rows_by_name = {row["source_name"]: row for row in rows}
    for source_name, expected_hit in EXPECTED_KNOWN_HITS.items():
        actual_hit = rows_by_name[source_name]["adapt_chem_accuracy_iteration"]
        _check(actual_hit == expected_hit, f"{source_name} ADAPT hit is {expected_hit or 'unresolved'}")

    _verify_entropy_overlap()
    _check((BENCHMARK_ROOT / "provenance" / "qcant_commit.txt").exists(), "QCANT commit provenance exists")

    if not args.skip_imports:
        _verify_imports()


if __name__ == "__main__":
    main()
