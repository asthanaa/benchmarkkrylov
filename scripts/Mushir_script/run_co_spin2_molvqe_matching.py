#!/usr/bin/env python
"""Prepare and run three CO triplet jobs using the MolVQE-matching CAS solution.

Export the saved full-space AVAS/cropped CAS(6e,6o) checkpoint once, validate
its exact sector ground energy, and pin that Hamiltonian for QRTE and QKUD.
W_MF retains the earlier canonical-HF-window determinant-energy convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np

import run_qrte_fock_pi_over_w as qrte
import run_qrte_mean_field_active_pi_over_w_small10 as mf
from compare_active_fock_width_small10 import canonical_window
from compute_initial_overlap import _payload_from_cache_path

ROOT = qrte.ROOT
CASE_ID = "CO_equi_cc_pVTZ_2"
DEFAULT_OUTDIR = ROOT / "runs/co_spin2_molvqe_matching"
REFERENCE_DIR = DEFAULT_OUTDIR / "reference"
CACHE_PATH = ROOT / "hamiltonian_cache/co_spin2_molvqe_matching/CO_equi_cc_pVTZ_2.npz"
DIAGNOSTIC_DIR = ROOT / "runs/co_spin2_reference_diagnostic"
CHECKPOINT = DIAGNOSTIC_DIR / "full_avas_cropped_canonicalize_False_casscf.chk"
TASKS = (
    {"name": "qrte_dt0p05_trotter1", "method": "qrte", "delta_t": .05, "trotter_order": 1},
    {"name": "qrte_pi_over_wmf_trotter2", "method": "qrte", "delta_t": None, "trotter_order": 2},
    {"name": "qkud_eps0p05_trotter1", "method": "qkud", "epsilon": .05, "trotter_order": 1},
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_dataset():
    with qrte.DEFAULT_MANIFEST.open(newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row["case_id"] == CASE_ID)
    path = ROOT / row["source_json_path"]
    raw = qrte.load_json(path)
    if raw["meta"]["molecule"]["spin"] != 2 or raw["meta"]["casscf"]["ncas"] != 6:
        raise ValueError("Unexpected dataset case or active space")
    return row, raw, path


def prepare_reference():
    """Reuse the diagnosed orbitals; never rerun SCF or CASSCF."""
    reference_path = REFERENCE_DIR / "reference.json"
    if reference_path.exists():
        return load_reference()
    if CACHE_PATH.exists():
        raise ValueError("Unrecorded Hamiltonian export exists; inspect it before preparing again")
    from pyscf import ao2mo, fci, lib, mcscf, scf
    from scipy.linalg import eigvalsh

    lib.num_threads(1)
    row, raw, dataset_path = read_dataset()
    diagnostic = qrte.load_json(DIAGNOSTIC_DIR / "diagnostic.json")
    saved = next(item for item in diagnostic["calculations"]
                 if item["starting_orbitals"] == "full_avas_cropped_canonicalize_False")
    if not saved["converged"] or abs(saved["spin_square"] - 2) > 1e-8:
        raise ValueError("Diagnosed CASSCF solution is not a converged triplet")
    if diagnostic["dataset_sha256"] != digest(dataset_path):
        raise ValueError("Dataset changed after the CASSCF diagnostic")
    rohf_path = DIAGNOSTIC_DIR / "rohf.chk"
    mol = lib.chkfile.load_mol(str(rohf_path))
    if (mol.spin, mol.charge, mol.nelectron, mol.nao_nr()) != (2, 0, 14, 60):
        raise ValueError("Unexpected checkpoint molecule")
    if (mol.basis != row["basis"] or mol.atom_charges().tolist() != [6, 8]
            or not np.allclose(mol.atom_coords(), np.asarray(json.loads(row["geometry_bohr"])), atol=1e-9, rtol=0)):
        raise ValueError("Checkpoint geometry or basis differs from the dataset")
    reference_scf = scf.ROHF(mol)
    reference_scf.__dict__.update(lib.chkfile.load(str(rohf_path), "scf"))
    reference_scf.converged = True
    if abs(reference_scf.e_tot - raw["meta"]["scf"]["energy"]) > 1e-8:
        raise ValueError("ROHF checkpoint differs from the dataset SCF reference")
    checkpoint_mol = lib.chkfile.load_mol(str(CHECKPOINT))
    if checkpoint_mol.dumps() != mol.dumps():
        # Output/chkfile settings can differ; verify the physical inputs directly.
        if (checkpoint_mol.spin != mol.spin or checkpoint_mol.basis != mol.basis
                or not np.allclose(checkpoint_mol.atom_coords(), mol.atom_coords(), atol=1e-12)):
            raise ValueError("CASSCF and ROHF checkpoints describe different molecules")
    data = lib.chkfile.load(str(CHECKPOINT), "mcscf")
    if (int(data["ncore"]), int(data["ncas"]), tuple(data["nelecas"])) != (4, 6, (4, 2)):
        raise ValueError("Unexpected checkpoint inactive/active partition")
    if abs(float(data["e_tot"]) - saved["energy_hartree"]) > 1e-12:
        raise ValueError("Checkpoint energy differs from the diagnosed second solution")
    mo = np.asarray(data["mo_coeff"])
    if not np.allclose(mo.T @ reference_scf.get_ovlp() @ mo, np.eye(60), atol=1e-8):
        raise ValueError("Saved CASSCF orbitals are not orthonormal")
    cas = mcscf.CASSCF(reference_scf, 6, (4, 2))
    h1, ecore = cas.get_h1eff(mo)
    eri = ao2mo.restore(1, cas.get_h2eff(mo), 6)
    addresses, matrix = fci.direct_spin1.pspace(h1, eri, 6, (4, 2), np=225)
    if not np.array_equal(np.sort(addresses), np.arange(225)):
        raise ValueError("Exact validation did not cover the complete sector")
    spectrum = eigvalsh(matrix) + ecore
    ground, ci = fci.direct_spin1.kernel(h1, eri, 6, (4, 2), ecore=ecore, tol=1e-12)
    spin_square = float(fci.spin_op.spin_square(ci, 6, (4, 2))[0])
    target = float(data["e_tot"])
    if (abs(spectrum[0] - target) > 1e-8 or abs(ground - target) > 1e-8
            or abs(target - raw["meta"]["casscf"]["energy"]) > 1e-8
            or abs(spin_square - 2) > 1e-8):
        raise ValueError("Exported Hamiltonian does not reproduce the MolVQE-matching triplet")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("xb") as handle:
        np.savez_compressed(handle, ecore=float(ecore), effective_active_electrons=6,
                            h1ecas=h1, two_mo=np.swapaxes(eri, 1, 3), ncas=6,
                            nelecas=np.array([4, 2]), mo_occ=data["mo_occ"],
                            hamiltonian_source="casscf_natorb", reference_energy=target,
                            reference_converged=True, scf_energy=float(reference_scf.e_tot),
                            scf_converged=True, active_space_selection="full_avas_order_cropped_to_cas6e6o")
    case = {"case_id": CASE_ID, "symbols": json.loads(row["symbols"]),
            "geometry_angstrom": json.loads(row["geometry_angstrom"]),
            "geometry_bohr": json.loads(row["geometry_bohr"]), "basis": row["basis"],
            "charge": 0, "spin": 2, "active_electrons": 6, "active_orbitals": 6,
            "nelecas": [4, 2], "active_orbital_indices": [], "active_orbital_index_base": 0,
            "active_orbital_labels": ["C 2p", "O 2p"],
            "hamiltonian_source": "casscf_natorb", "hamiltonian_cache_path": str(CACHE_PATH),
            "hamiltonian_sha256": digest(CACHE_PATH), "local_reference_energy_hartree": target,
            "local_reference_converged": True,
            "dataset_casscf_energy_hartree": float(raw["meta"]["casscf"]["energy"]),
            "reference_source": "MolVQE-matching full-space AVAS/cropped CAS(6e,6o) checkpoint",
            "reference_settings": {"reference_scf": "ROHF"},
            "avas_ncore": 0, "avas_canonicalize": False, "avas_openshell_option": 2,
            "avas_recommended_active_electrons": 8, "avas_recommended_active_orbitals": 7}
    fock = {"reference_scf": "ROHF", "fock_operator": "ROHF effective Roothaan spatial Fock",
            "full_scf_orbital_energies_hartree": np.asarray(reference_scf.mo_energy).tolist(),
            "full_scf_orbital_occupations": np.asarray(reference_scf.mo_occ).tolist()}
    estimate = canonical_window(case, fock)
    reference = {"status": "completed", "case": case, "dataset_path": str(dataset_path),
                 "dataset_sha256": digest(dataset_path), "checkpoint_path": str(CHECKPOINT),
                 "checkpoint_sha256": digest(CHECKPOINT), "rohf_checkpoint_path": str(rohf_path),
                 "rohf_checkpoint_sha256": digest(rohf_path), "scf_converged": True,
                 "scf_energy_hartree": float(reference_scf.e_tot),
                 "exact_sector_ground_energy_hartree": float(spectrum[0]),
                 "exact_sector_highest_energy_hartree": float(spectrum[-1]),
                 "exact_sector_ground_spin_square": spin_square,
                 "new_scf_or_casscf_calculations": False,
                 "active_orbital_estimate": estimate, **fock}
    qrte.write_json(reference_path, reference)
    return load_reference()


def load_reference():
    reference = qrte.load_json(REFERENCE_DIR / "reference.json")
    case = reference["case"]
    if reference["status"] != "completed" or case["case_id"] != CASE_ID:
        raise ValueError("Invalid prepared reference")
    for path, expected in [(case["hamiltonian_cache_path"], case["hamiltonian_sha256"]),
                           (reference["dataset_path"], reference["dataset_sha256"]),
                           (reference["checkpoint_path"], reference["checkpoint_sha256"]),
                           (reference["rohf_checkpoint_path"], reference["rohf_checkpoint_sha256"])]:
        if digest(path) != expected:
            raise ValueError(f"Prepared reference source changed: {path}")
    with np.load(case["hamiltonian_cache_path"], allow_pickle=False) as cache:
        if (int(cache["ncas"]), tuple(cache["nelecas"]), bool(cache["reference_converged"])) != (6, (4, 2), True):
            raise ValueError("Prepared cache sector or convergence changed")
        if abs(float(cache["reference_energy"]) - case["local_reference_energy_hartree"]) > 1e-12:
            raise ValueError("Prepared cache and accuracy reference differ")
    if abs(reference["exact_sector_ground_energy_hartree"] - case["local_reference_energy_hartree"]) > 1e-8:
        raise ValueError("Prepared exact ground energy and stopping target differ")
    return reference


def spectrum_builder(case, case_dir, args):
    reference = load_reference()
    if case != reference["case"]:
        raise ValueError("Run case differs from the prepared Hamiltonian")
    path = REFERENCE_DIR / "reference.json"
    return {"spectrum_scope": mf.SPECTRUM_SCOPE,
            "active_orbital_estimate": reference["active_orbital_estimate"],
            "reference_scf": "ROHF", "source_mean_field_reference_path": str(path),
            "source_mean_field_reference_sha256": digest(path), "new_reference_calculation": False}


def configurations(args, reference):
    width = mf.width_from_reference(spectrum_builder(reference["case"], None, args))
    common = {"case_id": CASE_ID, "hamiltonian_cache_path": str(CACHE_PATH),
              "hamiltonian_sha256": reference["case"]["hamiltonian_sha256"],
              "reference_source": reference["case"]["reference_source"],
              "reference_energy_hartree": reference["case"]["local_reference_energy_hartree"],
              "chemical_accuracy_hartree": qrte.CHEMICAL_ACCURACY,
              "n_steps": args.n_steps, "overlap_tol": args.overlap_tol,
              "max_workers": args.max_workers, "evolution_mode": "trotter", "trotter_steps": 1,
              "qcant_root": str(args.qcant_root.resolve()),
              "mean_field_active_bandwidth_hartree": width["mean_field_active_bandwidth_hartree"],
              "mean_field_timestep_convention": width["delta_t_rule"],
              "prepared_reference_sha256": digest(REFERENCE_DIR / "reference.json")}
    configs = []
    for index, task in enumerate(TASKS):
        config = {**common, **task, "task_index": index,
                  "output_directory": str(args.outdir / task["name"] / CASE_ID)}
        if task["method"] == "qrte":
            config["delta_t"] = task["delta_t"] if task["delta_t"] is not None else width["delta_t"]
        configs.append(config)
    return configs


def run_qkud(case, config, args):
    case_dir = Path(config["output_directory"])
    summary_path = case_dir / "summary.json"
    if summary_path.exists() and not args.force:
        saved = qrte.load_json(summary_path)
        if saved.get("status") == "completed" and saved.get("run_profile") == config:
            print("SKIP completed", config["name"])
            return
        raise ValueError("Existing QKUD result has different settings or failed; inspect before --force")
    case_dir.mkdir(parents=True, exist_ok=True)
    qrte.write_json(case_dir / "input_snapshot.json", case)
    qrte.write_json(case_dir / "run_config.json", config)
    sys.path.insert(0, str(args.qcant_root.resolve()))
    from QCANT import qulacs_accel as qa

    start = time.perf_counter()
    try:
        payload = _payload_from_cache_path(Path(case["hamiltonian_cache_path"]), hamiltonian_cutoff=1e-20, qa=qa)
        with qrte.cached_hamiltonian(qa, payload):
            # QKUD's circuit builder defaults to first order for both signs.
            energies, _states, history = qa.qkud_qulacs(
                symbols=case["symbols"], geometry=np.asarray(case["geometry_angstrom"]),
                basis=case["basis"], charge=0, spin=2, active_electrons=6, active_orbitals=6,
                epsilon=.05, n_steps=args.n_steps, overlap_tol=args.overlap_tol,
                evolution_mode="trotter", trotter_steps=1, max_workers=args.max_workers,
                early_stop_energy=case["local_reference_energy_hartree"] + qrte.CHEMICAL_ACCURACY,
                return_min_energy_history=True, hamiltonian_source="casscf_natorb")
        history = np.asarray(history, dtype=float)
        if not history.size or not np.all(np.isfinite(history)):
            raise RuntimeError("QKUD returned an empty or nonfinite history")
        errors = history - case["local_reference_energy_hartree"]
        if np.min(errors) < -qrte.CHEMICAL_ACCURACY:
            raise RuntimeError("QKUD fell below the prepared Hamiltonian reference")
        hit = next((i for i, error in enumerate(errors, 1) if abs(error) <= qrte.CHEMICAL_ACCURACY), None)
        with (case_dir / "min_energy_history.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["iteration", "min_energy_hartree", "error_vs_local_reference_hartree",
                             "error_vs_molvqe_casscf_hartree", "within_chemical_accuracy"])
            for index, (energy, error) in enumerate(zip(history, errors), 1):
                writer.writerow([index, float(energy), float(error),
                                 float(energy - case["dataset_casscf_energy_hartree"]),
                                 str(abs(error) <= qrte.CHEMICAL_ACCURACY).lower()])
        qrte.write_json(summary_path, {**case, **config, "run_profile": config, "status": "completed",
                                      "casscf_energy_hartree": case["local_reference_energy_hartree"],
                                      "n_steps_requested": args.n_steps, "n_steps_completed": int(history.size),
                                      "hit_iteration": hit, "hit_chemical_accuracy": hit is not None,
                                      "final_min_history_energy_hartree": float(history[-1]),
                                      "final_projected_ground_energy_hartree": float(np.min(energies)),
                                      "final_error_vs_local_reference_hartree": float(errors[-1]),
                                      "elapsed_seconds": time.perf_counter() - start})
        print(f"DONE {config['name']} iterations={history.size} hit={hit}", flush=True)
    except Exception as exc:
        qrte.write_json(summary_path, {"status": "failed", "case_id": CASE_ID,
                                      "run_profile": config, "error": repr(exc)})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, choices=range(3))
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--qcant-root", type=Path, default=ROOT / "vendor/QCANT")
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--overlap-tol", type=float, default=1e-10)
    parser.add_argument("--max-workers", type=int, default=18)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for key in ("n_steps", "max_workers", "overlap_tol"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive and finite")
    if args.task_index is None and not (args.prepare_only or args.dry_run):
        parser.error("Choose --task-index 0, 1, or 2, or use --prepare-only/--dry-run")
    args.outdir = args.outdir.resolve()
    return args


def main():
    args = parse_args()
    reference = prepare_reference() if args.prepare_only else load_reference()
    configs = configurations(args, reference)
    if args.prepare_only:
        qrte.write_json(args.outdir / "run_plan.json", {"case": reference["case"], "tasks": configs})
    if args.prepare_only or args.dry_run:
        for config in configs:
            step = f"dt={config['delta_t']:.12g}" if config["method"] == "qrte" else "eps=0.05"
            print(f"{config['task_index']}: {config['name']} {step} order={config['trotter_order']} slices=1")
        print(f"Reference: {reference['case']['local_reference_energy_hartree']:.14f} Eh; "
              f"exact ground: {reference['exact_sector_ground_energy_hartree']:.14f} Eh")
        print(f"W_MF={configs[0]['mean_field_active_bandwidth_hartree']:.12g} Eh; "
              f"maximum iterations={args.n_steps}; overlap tolerance={args.overlap_tol:g}")
        print("Prepared/validated only; no QRTE, QKUD, or Slurm jobs launched.")
        return
    config = configs[args.task_index]
    if config["method"] == "qkud":
        run_qkud(reference["case"], config, args)
    else:
        args.outdir = args.outdir / config["name"]
        args.delta_t = config["delta_t"] if args.task_index == 0 else None
        args.evolution_mode, args.trotter_steps = "trotter", 1
        args.trotter_order = config["trotter_order"]
        args.scf_match_tolerance, args.reference_only = 1e-6, False
        qrte.run_case(reference["case"], args, spectrum_builder=spectrum_builder,
                      width_builder=mf.width_from_reference, spectrum_scope=mf.SPECTRUM_SCOPE,
                      spectrum_filename=mf.SPECTRUM_FILENAME)


if __name__ == "__main__":
    main()
