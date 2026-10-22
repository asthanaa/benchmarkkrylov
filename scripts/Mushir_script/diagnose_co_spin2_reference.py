#!/usr/bin/env python
"""Compare CO triplet CASSCF starting spaces without changing benchmark caches."""

import csv
import hashlib
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pyscf
from pyscf import fci, gto, lib, mcscf, scf
from pyscf.fci import cistring
from pyscf.mcscf import avas

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/co_spin2_reference_diagnostic"
CASE = "CO_equi_cc_pVTZ_2"


def main():
    lib.num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "manifests/molvqe21_fig1b_manifest.csv").open(newline="") as handle:
        entry = next(row for row in csv.DictReader(handle) if row["case_id"] == CASE)
    raw_path = ROOT / entry["source_json_path"]
    raw = json.loads(raw_path.read_text())
    with (ROOT / "hamiltonian_cache/correct_27_systems/manifest.csv").open(newline="") as handle:
        mapping = next(row for row in csv.DictReader(handle) if row["case_id"] == CASE)
    cache_path = ROOT / mapping["consolidated_cache_path"]
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    assert digest == mapping["sha256"]
    with np.load(cache_path, allow_pickle=False) as cache:
        cached_reference = float(cache["reference_energy"])
        h1 = cache["h1ecas"]
        eri = np.swapaxes(cache["two_mo"], 1, 3)
        ncas = int(cache["ncas"])
        nelecas = tuple(int(n) for n in cache["nelecas"])
        ecore = float(cache["ecore"])
        cached_occ = cache["mo_occ"].tolist()
    energy, ci = fci.direct_spin1.kernel(h1, eri, ncas, nelecas, ecore=ecore, tol=1e-12)
    result = {
        "case_id": CASE, "pyscf_version": pyscf.__version__, "threads": 1,
        "dataset_path": str(raw_path.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "cache_path": str(cache_path.relative_to(ROOT)), "cache_sha256": digest,
        "dataset_scf_energy_hartree": raw["meta"]["scf"]["energy"],
        "dataset_casscf_energy_hartree": raw["meta"]["casscf"]["energy"],
        "cached_reference_energy_hartree": cached_reference,
        "cached_ground_energy_hartree": float(energy),
        "cached_ground_spin_square": float(fci.spin_op.spin_square(ci, ncas, nelecas)[0]),
        "cached_natural_occupations": cached_occ,
        "calculations": [],
    }
    dataset_ci = np.asarray(raw["meta"]["casscf"]["ci_vector"]).reshape(ci.shape)
    alpha = cistring.make_strings(range(ncas), nelecas[0])
    beta = cistring.make_strings(range(ncas), nelecas[1])
    # Diagonal occupations need coefficient magnitudes only; do not infer
    # off-diagonal density elements from the dataset's nonnegative coefficients.
    result["dataset_ci_norm_squared"] = float(np.sum(dataset_ci**2))
    result["dataset_ci_diagonal_occupations"] = [
        float(sum(dataset_ci[i, j]**2 * ((int(a) >> k & 1) + (int(b) >> k & 1))
                  for i, a in enumerate(alpha) for j, b in enumerate(beta)))
        for k in range(ncas)]

    def save():
        (OUT / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    print("Cached ground:", energy, "S2:", result["cached_ground_spin_square"], flush=True)
    mol = gto.M(atom=list(zip(json.loads(entry["symbols"]), json.loads(entry["geometry_angstrom"]))),
                unit="Angstrom", basis=entry["basis"], charge=int(entry["charge"]),
                spin=2, symmetry=False, verbose=4, output=str(OUT / "pyscf.log"))
    mf = scf.ROHF(mol)
    mf.level_shift, mf.diis_space, mf.max_cycle = .5, 12, 100
    mf.chkfile = str(OUT / "rohf.chk")
    mf.kernel()
    assert mf.converged
    result["recomputed_scf_energy_hartree"] = float(mf.e_tot)
    result["recomputed_scf_spin_square"] = float(mf.spin_square()[0])
    avas_ncas, avas_nelec, avas_mo = avas.avas(mf, ["C 2p", "O 2p"], canonicalize=False,
                                             ncore=4, openshell_option=2, verbose=4)
    avas_total = int(sum(avas_nelec)) if isinstance(avas_nelec, tuple) else int(avas_nelec)
    result["avas"] = {"ncas": int(avas_ncas), "nelec": avas_total,
                      "builder_falls_back_to_canonical": (int(avas_ncas), avas_total) != (6, 6)}
    save()
    print("ROHF:", mf.e_tot, "AVAS:", result["avas"], flush=True)
    starts = [("canonical", mf.mo_coeff)]
    if (int(avas_ncas), avas_total) == (6, 6):
        starts.append(("avas", avas_mo))
    # The paper does not specify AVAS's frozen-core or canonicalization choices.
    # Test a full-space AVAS ordering as a guess for the recorded CAS(6e,6o).
    # This deliberately crops the AVAS recommendation; record that distinction.
    result["full_space_avas"] = []
    for canonicalize in (True, False):
        full_ncas, full_nelec, full_mo = avas.avas(
            mf, ["C 2p", "O 2p"], ncore=0, canonicalize=canonicalize, verbose=4)
        result["full_space_avas"].append({"ncas": int(full_ncas), "nelec": int(full_nelec),
                                          "canonicalize": canonicalize})
        starts.append((f"full_avas_cropped_canonicalize_{canonicalize}", full_mo))
    final_orbitals = {}
    for name, mo in starts:
        start = time.monotonic()
        cas = mcscf.CASSCF(mf, 6, (4, 2))
        cas.natorb, cas.max_cycle, cas.conv_tol = True, 100, 1e-10
        cas.chkfile = str(OUT / f"{name}_casscf.chk")
        cas.kernel(mo)
        final_orbitals[name] = cas.mo_coeff
        record = {"starting_orbitals": name, "energy_hartree": float(cas.e_tot),
                  "converged": bool(cas.converged),
                  "spin_square": float(cas.fcisolver.spin_square(cas.ci, 6, (4, 2))[0]),
                  "natural_occupations": np.asarray(cas.mo_occ)[4:10].tolist(),
                  "minus_dataset_mha": 1000 * (float(cas.e_tot) - result["dataset_casscf_energy_hartree"]),
                  "minus_cache_mha": 1000 * (float(cas.e_tot) - cached_reference),
                  "elapsed_seconds": time.monotonic() - start}
        result["calculations"].append(record)
        save()
        print(json.dumps(record), flush=True)
    ovlp = mf.get_ovlp()
    result["final_active_space_overlaps_with_local_avas_solution"] = {
        name: np.linalg.svd(final_orbitals["avas"][:, 4:10].T @ ovlp @ mo[:, 4:10],
                            compute_uv=False).tolist()
        for name, mo in final_orbitals.items()}
    result["interpretation"] = (
        "Both starts from canonical orbitals and AVAS with ncore=4 reproduce the local cache. "
        "AVAS with ncore=0 recommends CAS(8e,7o); passing its orbital order to CAS(6e,6o) "
        "reproduces the dataset reference, for both canonicalization choices. "
        "This establishes a reproducible preparation-dependent CASSCF solution difference, "
        "but does not prove which settings the dataset authors actually used.")
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == digest
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == result["dataset_sha256"]
    save()
    print("Saved", OUT / "diagnostic.json", flush=True)


if __name__ == "__main__":
    main()
