#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Run ROHF and CASSCF references directly from MolVQE JSON files."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = BENCHMARK_ROOT / "runs" / "rohf_molvqe_fe_cases"
DEFAULT_CASE_JSONS = {
    "FeO": BENCHMARK_ROOT
    / "dataset"
    / "zenodo_19051536"
    / "extracted"
    / "data_repository"
    / "MolVQE-21"
    / "FeO.json",
    "feo4_2minus_ceo": BENCHMARK_ROOT
    / "dataset"
    / "zenodo_19051536"
    / "extracted"
    / "data_repository"
    / "MolVQE-21"
    / "feo4_2minus_ceo.json",
}
ACTIVE_ORBITAL_LABELS_BY_CASE = {
    "FeO": ["Fe 3d", "O 2p"],
    "feo4_2minus_ceo": ["Fe 3d"],
}


def _set_thread_limits() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _load_case(case_id: str, path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta = payload["meta"]
    molecule = meta["molecule"]
    casscf = meta["casscf"]
    scf_meta = meta.get("scf", {})
    atoms = molecule["atoms"]
    geometry_bohr = [[float(value) for value in atom["xyz"]] for atom in atoms]
    return {
        "case_id": case_id,
        "source_json_path": str(path.relative_to(BENCHMARK_ROOT)),
        "symbols": [str(atom["element"]) for atom in atoms],
        "geometry_bohr": geometry_bohr,
        "basis": str(molecule["basis"]),
        "charge": int(molecule["charge"]),
        "spin": int(molecule["spin"]),
        "symmetry": bool(molecule.get("symmetry", False)),
        "active_electrons": int(casscf["nelec"]),
        "active_orbitals": int(casscf["ncas"]),
        "dataset_scf_method": str(scf_meta.get("method", "")),
        "dataset_scf_converged": bool(scf_meta.get("converged", False)),
        "dataset_scf_energy_hartree": float(scf_meta["energy"]),
        "dataset_casscf_energy_hartree": float(casscf["energy"]),
        "dataset_casscf_converged": bool(casscf["converged"]),
        "dataset_casscf_natorb": bool(casscf["natorb"]),
    }


def _completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def _active_orbital_labels(case_id: str, use_avas: bool) -> list[str] | None:
    if not use_avas:
        return None
    labels = ACTIVE_ORBITAL_LABELS_BY_CASE.get(case_id)
    return None if labels is None else list(labels)


def _build_mol(case: dict[str, Any], log_path: Path, pyscf_verbose: int):
    from pyscf import gto

    atom = [
        (case["symbols"][idx], tuple(float(value) for value in xyz))
        for idx, xyz in enumerate(case["geometry_bohr"])
    ]
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Bohr"
    mol.basis = case["basis"]
    mol.charge = int(case["charge"])
    mol.spin = int(case["spin"])
    mol.symmetry = bool(case["symmetry"])
    mol.verbose = int(pyscf_verbose)
    mol.output = str(log_path)
    mol.build()
    return mol


def _run_rohf(mol, args: argparse.Namespace):
    from pyscf import scf

    mf = scf.ROHF(mol)
    mf.conv_tol = float(args.scf_conv_tol)
    mf.diis_space = int(args.scf_diis_space)
    mf.level_shift = float(args.scf_level_shift)
    mf.max_cycle = int(args.scf_max_cycle)
    mf.kernel()
    if not mf.converged and args.newton_if_not_converged:
        mf = scf.newton(mf).run()
    return mf


def _initial_active_orbitals(mf, mol, case: dict[str, Any], active_orbital_labels: Sequence[str] | None):
    if not active_orbital_labels:
        return None, None

    from pyscf.mcscf import avas

    active_electrons = int(case["active_electrons"])
    active_orbitals = int(case["active_orbitals"])
    ncore = int((int(mol.nelectron) - active_electrons) // 2)
    avas_ncas, avas_nelecas, mo0 = avas.avas(
        mf,
        list(active_orbital_labels),
        canonicalize=False,
        ncore=ncore,
        openshell_option=2,
        verbose=0,
    )
    try:
        avas_nelecas_tuple = tuple(int(value) for value in avas_nelecas)
    except TypeError:
        avas_nelecas_tuple = (int(avas_nelecas),)
    avas_nelec = int(sum(avas_nelecas_tuple))
    avas_info = {
        "active_orbital_labels": list(active_orbital_labels),
        "avas_ncas": int(avas_ncas),
        "avas_nelecas": avas_nelecas_tuple,
        "avas_nelec_total": avas_nelec,
        "avas_used_as_initial_guess": int(avas_ncas) == active_orbitals and avas_nelec == active_electrons,
    }
    if not avas_info["avas_used_as_initial_guess"]:
        return None, avas_info
    return mo0, avas_info


def _run_casscf(mf, mo0, case: dict[str, Any], args: argparse.Namespace):
    from pyscf import mcscf

    natorb = bool(case["dataset_casscf_natorb"])
    if args.force_natorb:
        natorb = True

    mc = mcscf.CASSCF(mf, int(case["active_orbitals"]), int(case["active_electrons"]))
    mc.natorb = natorb
    mc.conv_tol = float(args.casscf_conv_tol)
    mc.max_cycle = int(args.casscf_max_cycle)
    energy = float(mc.kernel(mo0)[0] if mo0 is not None else mc.kernel()[0])
    return mc, energy, natorb


def _run_case(case: dict[str, Any], args: argparse.Namespace) -> str:
    case_id = case["case_id"]
    case_dir = args.outdir / case_id
    summary_path = case_dir / "summary.json"
    if _completed(summary_path) and not args.force:
        print(f"SKIP completed ROHF/CASSCF {case_id}", flush=True)
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    pyscf_log_path = case_dir / "pyscf.log"
    active_orbital_labels = _active_orbital_labels(case_id, not args.no_avas_labels)
    start = time.perf_counter()

    input_snapshot = {
        **case,
        "active_orbital_labels": active_orbital_labels,
        "active_orbital_label_source": "local benchmark AVAS mapping; MolVQE JSON does not store orbital labels",
        "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
        "geometry_source": "MolVQE JSON meta.molecule atoms xyz, used directly with PySCF unit='Bohr'",
        "scf_method_requested": "ROHF",
    }
    _write_json(case_dir / "input_snapshot.json", input_snapshot)

    run_config = {
        "allow_unconverged_casscf": bool(args.allow_unconverged_casscf),
        "casscf_conv_tol": float(args.casscf_conv_tol),
        "casscf_max_cycle": int(args.casscf_max_cycle),
        "force_natorb": bool(args.force_natorb),
        "honor_dataset_casscf_natorb": not bool(args.force_natorb),
        "newton_if_not_converged": bool(args.newton_if_not_converged),
        "pyscf_verbose": int(args.pyscf_verbose),
        "scf_conv_tol": float(args.scf_conv_tol),
        "scf_diis_space": int(args.scf_diis_space),
        "scf_level_shift": float(args.scf_level_shift),
        "scf_max_cycle": int(args.scf_max_cycle),
        "use_avas_labels": not bool(args.no_avas_labels),
    }
    _write_json(case_dir / "run_config.json", run_config)

    try:
        mol = _build_mol(case, pyscf_log_path, int(args.pyscf_verbose))
        mf = _run_rohf(mol, args)
        mo0, avas_info = _initial_active_orbitals(mf, mol, case, active_orbital_labels)
        mc, casscf_energy, casscf_natorb_used = _run_casscf(mf, mo0, case, args)

        if not bool(mc.converged) and not args.allow_unconverged_casscf:
            raise RuntimeError("CASSCF did not converge")

        local_scf_energy = float(mf.e_tot)
        summary = {
            "status": "completed",
            "case_id": case_id,
            "source_json_path": case["source_json_path"],
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "symmetry": bool(case["symmetry"]),
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "active_orbital_labels": active_orbital_labels,
            "active_orbital_label_source": "local benchmark AVAS mapping; MolVQE JSON does not store orbital labels",
            "active_space_source": "MolVQE JSON meta.casscf ncas/nelec",
            "avas_info": avas_info,
            "geometry_source": "MolVQE JSON meta.molecule atoms xyz, used directly with PySCF unit='Bohr'",
            "local_scf_method": "ROHF",
            "local_scf_energy_hartree": local_scf_energy,
            "local_scf_converged": bool(mf.converged),
            "dataset_scf_method": case["dataset_scf_method"],
            "dataset_scf_energy_hartree": case["dataset_scf_energy_hartree"],
            "dataset_scf_converged": case["dataset_scf_converged"],
            "local_minus_dataset_scf_hartree": local_scf_energy - case["dataset_scf_energy_hartree"],
            "local_casscf_energy_hartree": casscf_energy,
            "local_casscf_converged": bool(mc.converged),
            "local_casscf_natorb": bool(casscf_natorb_used),
            "local_casscf_nelecas": tuple(int(value) for value in mc.nelecas),
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "dataset_casscf_converged": case["dataset_casscf_converged"],
            "dataset_casscf_natorb": case["dataset_casscf_natorb"],
            "local_minus_dataset_casscf_hartree": casscf_energy - case["dataset_casscf_energy_hartree"],
            "pyscf_log_path": str(pyscf_log_path),
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        _write_json(summary_path, summary)
        print(
            f"DONE ROHF/CASSCF {case_id} "
            f"scf={local_scf_energy:.12f} casscf={casscf_energy:.12f}",
            flush=True,
        )
        return "completed"
    except Exception as exc:  # noqa: BLE001
        failure = {
            "status": "failed",
            "case_id": case_id,
            "source_json_path": case["source_json_path"],
            "active_electrons": int(case["active_electrons"]),
            "active_orbitals": int(case["active_orbitals"]),
            "basis": case["basis"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
            "dataset_scf_energy_hartree": case["dataset_scf_energy_hartree"],
            "dataset_casscf_energy_hartree": case["dataset_casscf_energy_hartree"],
            "elapsed_seconds": float(time.perf_counter() - start),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, failure)
        print(f"FAIL ROHF/CASSCF {case_id}: {exc!r}", flush=True)
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--scf-diis-space", type=int, default=12)
    parser.add_argument("--scf-level-shift", type=float, default=0.0)
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--casscf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--casscf-max-cycle", type=int, default=100)
    parser.add_argument("--pyscf-verbose", type=int, default=4)
    parser.add_argument("--newton-if-not-converged", action="store_true")
    parser.add_argument("--allow-unconverged-casscf", action="store_true")
    parser.add_argument("--force-natorb", action="store_true")
    parser.add_argument("--no-avas-labels", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    _set_thread_limits()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    selected_case_ids: Sequence[str] = tuple(args.case_id or DEFAULT_CASE_JSONS)
    missing = [case_id for case_id in selected_case_ids if case_id not in DEFAULT_CASE_JSONS]
    if missing:
        raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")

    cases = [_load_case(case_id, DEFAULT_CASE_JSONS[case_id]) for case_id in selected_case_ids]
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    for case in cases:
        status = _run_case(case, args)
        counts[status] = counts.get(status, 0) + 1

    print(json.dumps(counts, sort_keys=True), flush=True)
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
