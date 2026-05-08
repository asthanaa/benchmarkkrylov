"""Run one live-updating all-trans polyene QP thick study from an `input.json`.

This mirrors the QP-backed live trace workflow used for N2 and Cr2, but builds
an idealized planar bond-alternated polyene directly in PySCF and injects a
custom MO ordering into EZFIO:

- full molecule in the requested basis
- active space = localized 24-orbital pi window
- active orbitals ordered along the carbon chain
- DMRG reference from the localized active-space FCIDUMP
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import debug_selector_parity as dsp  # noqa: E402
import cr2_qp_fixed_budget_sweep_plots as cr2_fixed  # noqa: E402
import n2_thick_iteration_study as n2_study  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    _set_mo_class_direct,
    ensure_ezfio_defaults,
    validate_rebuildable_qp_root,
    write_ezfio_value,
)


@dataclass(frozen=True)
class PolyeneCase:
    name: str
    title: str
    n_carbons: int
    basis: str
    ncas: int
    nelecas: int
    ncore: int
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"
    geometry_mode: str = "bond_alternated_planar"
    cc_single_ang: float = 1.45
    cc_double_ang: float = 1.35
    ch_ang: float = 1.08
    localizer: str = "pipek"


def _require_qp_root(spec: dict[str, Any]) -> Path:
    study_cfg = spec.get("study", {})
    raw = study_cfg.get("qp_root")
    if raw is None:
        raise RuntimeError("Polyene QP study requires study.qp_root")
    qp_root = Path(str(raw)).expanduser().resolve()
    validation = validate_rebuildable_qp_root(qp_root)
    if not validation.get("is_rebuildable"):
        raise RuntimeError(
            "Polyene QP study requires a rebuildable Quantum Package root. "
            f"validation={json.dumps(validation, sort_keys=True)}"
        )
    return qp_root


def _case_from_spec(spec: dict[str, Any]) -> PolyeneCase:
    system_cfg = spec.get("system", {})
    molecule = spec["molecule"]
    cas_cfg = spec["cas"]
    selection_cfg = spec.get("active_space", {})
    return PolyeneCase(
        name=str(spec.get("name", "polyene_case")).strip() or "polyene_case",
        title=str(spec["label"]),
        n_carbons=int(system_cfg.get("n_carbons", 24)),
        basis=str(molecule["basis"]),
        ncas=int(cas_cfg["ncas"]),
        nelecas=int(cas_cfg["nelecas"]),
        ncore=int(cas_cfg["ncore"]),
        charge=int(molecule.get("charge", 0)),
        spin=int(molecule.get("spin", 0)),
        unit=str(molecule.get("unit", "Angstrom")),
        geometry_mode=str(system_cfg.get("geometry_mode", "bond_alternated_planar")),
        cc_single_ang=float(system_cfg.get("cc_single_ang", 1.45)),
        cc_double_ang=float(system_cfg.get("cc_double_ang", 1.35)),
        ch_ang=float(system_cfg.get("ch_ang", 1.08)),
        localizer=str(selection_cfg.get("localizer", "pipek")).strip().lower() or "pipek",
    )


def _rotate2d(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], 0.0], dtype=float)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        raise RuntimeError("Encountered zero-length vector while building polyene geometry")
    return np.asarray(vec, dtype=float) / norm


def _carbon_backbone(case: PolyeneCase) -> list[np.ndarray]:
    if case.n_carbons < 2:
        raise ValueError("Polyene requires at least two carbons")
    coords: list[np.ndarray] = [np.zeros(3, dtype=float)]
    for bond_idx in range(1, int(case.n_carbons)):
        length = float(case.cc_double_ang if bond_idx % 2 == 1 else case.cc_single_ang)
        if bond_idx == 1:
            angle_deg = 0.0
        else:
            angle_deg = 60.0 if bond_idx % 2 == 0 else -60.0
        step = length * np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg)), 0.0], dtype=float)
        coords.append(coords[-1] + step)
    center = np.mean(np.vstack(coords), axis=0)
    return [coord - center for coord in coords]


def _hydrogen_sites(carbon_coords: list[np.ndarray], *, ch_ang: float) -> list[np.ndarray]:
    hydrogens: list[np.ndarray] = []
    n = len(carbon_coords)
    for idx, carbon in enumerate(carbon_coords):
        if idx == 0:
            toward_chain = _unit(carbon_coords[1] - carbon)
            dirs = [_unit(_rotate2d(toward_chain, 120.0)), _unit(_rotate2d(toward_chain, -120.0))]
            hydrogens.extend([carbon + float(ch_ang) * direction for direction in dirs])
            continue
        if idx == n - 1:
            toward_chain = _unit(carbon_coords[-2] - carbon)
            dirs = [_unit(_rotate2d(toward_chain, 120.0)), _unit(_rotate2d(toward_chain, -120.0))]
            hydrogens.extend([carbon + float(ch_ang) * direction for direction in dirs])
            continue
        prev_dir = _unit(carbon_coords[idx - 1] - carbon)
        next_dir = _unit(carbon_coords[idx + 1] - carbon)
        outward = _unit(-(prev_dir + next_dir))
        hydrogens.append(carbon + float(ch_ang) * outward)
    return hydrogens


def _polyene_atom(case: PolyeneCase) -> tuple[str, list[np.ndarray]]:
    if case.geometry_mode != "bond_alternated_planar":
        raise ValueError(f"Unsupported polyene geometry_mode: {case.geometry_mode!r}")
    carbons = _carbon_backbone(case)
    hydrogens = _hydrogen_sites(carbons, ch_ang=float(case.ch_ang))
    lines = [
        *(f"C {coord[0]: .12f} {coord[1]: .12f} {coord[2]: .12f}" for coord in carbons),
        *(f"H {coord[0]: .12f} {coord[1]: .12f} {coord[2]: .12f}" for coord in hydrogens),
    ]
    return "\n".join(lines), carbons


def _build_polyene_mf(case: PolyeneCase):
    from pyscf import gto, scf

    atom, carbon_coords = _polyene_atom(case)
    mol = gto.M(
        atom=atom,
        basis=case.basis,
        unit=case.unit,
        charge=case.charge,
        spin=case.spin,
        verbose=0,
        cart=True,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF RHF did not converge for {case.title}")
    return mol, mf, carbon_coords


def _carbon_pz_ao_groups(mol, n_carbons: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(int(n_carbons))]
    for ao_idx, label in enumerate(mol.ao_labels(fmt=False)):
        atom_idx, symbol, nl, component = label
        if int(atom_idx) >= int(n_carbons):
            continue
        if str(symbol).strip().upper() != "C":
            continue
        if str(component).strip().lower() != "z":
            continue
        nl_text = str(nl).strip().lower()
        if "p" not in nl_text:
            continue
        groups[int(atom_idx)].append(int(ao_idx))
    if not all(groups):
        raise RuntimeError("Failed to identify carbon p_z AO groups for the polyene active-space selection")
    return groups


def _localize_active_window(mol, coeff: np.ndarray, *, localizer: str) -> tuple[np.ndarray, str]:
    from pyscf import lo

    mode = str(localizer).strip().lower()
    if mode in {"pipek", "pipek_mezey", "pm"}:
        try:
            return np.asarray(lo.PM(mol, coeff).kernel(), dtype=float), "pipek_mezey"
        except Exception:
            return np.asarray(lo.Boys(mol, coeff).kernel(), dtype=float), "boys_fallback"
    if mode == "boys":
        return np.asarray(lo.Boys(mol, coeff).kernel(), dtype=float), "boys"
    raise ValueError(f"Unsupported localizer: {localizer!r}")


def _orient_and_order_active(
    active_coeff: np.ndarray,
    *,
    carbon_coords: list[np.ndarray],
    pz_groups: list[list[int]],
) -> tuple[np.ndarray, list[float], list[int], list[float]]:
    carbon_x = [float(coord[0]) for coord in carbon_coords]
    coeff = np.asarray(active_coeff, dtype=float).copy()
    centers: list[float] = []
    dominant_carbons: list[int] = []
    dominant_weights: list[float] = []
    for orb_idx in range(coeff.shape[1]):
        per_carbon = np.array([float(np.sum(np.square(coeff[idxs, orb_idx]))) for idxs in pz_groups], dtype=float)
        dominant = int(np.argmax(per_carbon))
        dominant_carbons.append(dominant)
        dominant_weights.append(float(per_carbon[dominant]))
        sign_probe = float(np.sum(coeff[pz_groups[dominant], orb_idx]))
        if sign_probe < 0.0:
            coeff[:, orb_idx] *= -1.0
        total = float(np.sum(per_carbon))
        center = float(np.dot(per_carbon, np.asarray(carbon_x, dtype=float)) / total) if total > 1.0e-12 else float(orb_idx)
        centers.append(center)
    order = list(np.argsort(np.asarray(centers), kind="stable"))
    ordered = coeff[:, order]
    ordered_centers = [float(centers[idx]) for idx in order]
    ordered_dominants = [int(dominant_carbons[idx]) for idx in order]
    ordered_weights = [float(dominant_weights[idx]) for idx in order]
    return ordered, ordered_centers, ordered_dominants, ordered_weights


def _build_active_orbital_model(case: PolyeneCase) -> dict[str, Any]:
    from pyscf import ao2mo, mcscf
    from pyscf.tools import fcidump as pyscf_fcidump

    mol, mf, carbon_coords = _build_polyene_mf(case)
    nmo_total = int(mf.mo_coeff.shape[1])
    nocc = int(round(float(np.sum(np.asarray(mf.mo_occ) > 1.0))))
    occ_active = int(case.nelecas // 2)
    virt_active = int(case.ncas - occ_active)
    active_start = int(nocc - occ_active)
    active_stop = int(nocc + virt_active)
    if active_stop - active_start != int(case.ncas):
        raise RuntimeError("Polyene active-space window size does not match CAS size")
    if active_start != int(case.ncore):
        raise RuntimeError(
            "Polyene input ncore is inconsistent with the requested HOMO-centered pi window. "
            f"expected ncore={active_start}, got {case.ncore}"
        )
    active_idx = np.arange(active_start, active_stop, dtype=int)
    core_idx = np.arange(0, active_start, dtype=int)
    deleted_idx = np.arange(active_stop, nmo_total, dtype=int)

    pz_groups = _carbon_pz_ao_groups(mol, int(case.n_carbons))
    localized, localizer_used = _localize_active_window(mol, mf.mo_coeff[:, active_idx], localizer=case.localizer)
    ordered_active, active_centers, dominant_carbons, dominant_weights = _orient_and_order_active(
        localized,
        carbon_coords=carbon_coords,
        pz_groups=pz_groups,
    )

    # The QP thick-restart path only needs the inactive core plus the active
    # pi window. Keeping the trailing deleted virtuals forces a much larger
    # AO->MO transformation without changing the CAS Hamiltonian.
    qp_mo = np.column_stack([mf.mo_coeff[:, core_idx], ordered_active])
    active_occ_value = float(case.nelecas) / float(case.ncas)
    qp_occ = np.concatenate(
        [
            np.full(len(core_idx), 2.0, dtype=float),
            np.full(int(case.ncas), active_occ_value, dtype=float),
        ]
    )

    mc = mcscf.CASCI(mf, int(case.ncas), int(case.nelecas))
    mc.ncore = int(case.ncore)
    mc.mo_coeff = np.column_stack([mf.mo_coeff[:, core_idx], ordered_active, mf.mo_coeff[:, deleted_idx]])
    h1eff, ecore = mc.get_h1eff()
    eri_packed = mc.get_h2eff()
    eri8 = ao2mo.restore(8, eri_packed, int(case.ncas))
    nelec_cas = (int(case.nelecas // 2), int(case.nelecas - case.nelecas // 2))

    payload = _build_generic_shell_payload(
        mol=mol,
        basis_name=case.basis,
        mo_coeff=np.asarray(qp_mo, dtype=float),
        mo_occ=np.asarray(qp_occ, dtype=float),
        mo_label=f"PySCF_RHF_{case.basis.upper().replace('-', '')}_POLYENE_PI",
        total_energy_eh=float(mf.e_tot),
    )
    return {
        "mol": mol,
        "mf": mf,
        "carbon_coords": carbon_coords,
        "payload": payload,
        "nmo_total": nmo_total,
        "full_nmo_total": nmo_total,
        "qp_mo": np.asarray(qp_mo, dtype=float),
        "qp_occ": np.asarray(qp_occ, dtype=float),
        "qp_mo_num": int(qp_mo.shape[1]),
        "active_window_start": active_start,
        "active_window_stop": active_stop,
        "active_centers": active_centers,
        "active_dominant_carbons": dominant_carbons,
        "active_dominant_weights": dominant_weights,
        "localizer_used": localizer_used,
        "ecore": float(ecore),
        "norb": int(case.ncas),
        "nelec_cas": nelec_cas,
        "h1": np.asarray(h1eff),
        "eri8": np.asarray(eri8),
    }


def _build_generic_shell_payload(
    *,
    mol,
    basis_name: str,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    mo_label: str,
    total_energy_eh: float,
) -> dict[str, object]:
    ao_labels = list(mol.ao_labels(fmt=False))
    ao_cursor = 0
    shell_prim_num: list[int] = []
    shell_ang_mom: list[int] = []
    shell_index: list[int] = []
    basis_nucleus_index: list[int] = []
    prim_coef: list[float] = []
    prim_expo: list[float] = []
    ao_prim_num: list[int] = []
    ao_nucl: list[int] = []
    ao_power: list[list[int]] = []
    ao_coef: list[list[float]] = []
    ao_expo: list[list[float]] = []
    shell_counter = 0

    for ib in range(mol.nbas):
        atom_index = int(mol.bas_atom(ib)) + 1
        nprim = int(mol.bas_nprim(ib))
        nctr = int(mol.bas_nctr(ib))
        l = int(mol.bas_angular(ib))
        expos = [float(x) for x in mol.bas_exp(ib).tolist()]
        ctr_coeff = mol.bas_ctr_coeff(ib)
        if ctr_coeff.shape[1] != nctr:
            raise RuntimeError(
                "PySCF contraction matrix shape mismatch while building the polyene shell: "
                f"expected {nctr} contraction columns, got {ctr_coeff.shape[1]}"
            )
        n_cart = int(mol.bas_len_cart(ib))
        for ictr in range(nctr):
            coeffs = [float(x) for x in ctr_coeff[:, ictr].tolist()]
            shell_counter += 1
            shell_prim_num.append(nprim)
            shell_ang_mom.append(l)
            basis_nucleus_index.append(atom_index)
            prim_coef.extend(coeffs)
            prim_expo.extend(expos)
            shell_index.extend([shell_counter] * nprim)
            for _ in range(n_cart):
                if ao_cursor >= len(ao_labels):
                    raise RuntimeError("AO label cursor overflow while building the polyene shell")
                component = str(ao_labels[ao_cursor][3]).strip()
                power = [component.count("x"), component.count("y"), component.count("z")]
                ao_prim_num.append(nprim)
                ao_nucl.append(atom_index)
                ao_power.append(power)
                ao_coef.append(coeffs)
                ao_expo.append(expos)
                ao_cursor += 1
    if ao_cursor != len(ao_labels):
        raise RuntimeError("AO label cursor mismatch while building the polyene shell")

    ao_prim_num_max = max(ao_prim_num) if ao_prim_num else 0
    padded_ao_coef = [row + [0.0] * (ao_prim_num_max - len(row)) for row in ao_coef]
    padded_ao_expo = [row + [0.0] * (ao_prim_num_max - len(row)) for row in ao_expo]
    nucleus_shell_num = [basis_nucleus_index.count(atom_idx + 1) for atom_idx in range(mol.natm)]
    return {
        "mo_num": int(mo_coeff.shape[1]),
        "mo_label": str(mo_label),
        "mo_occ": [float(x) for x in np.asarray(mo_occ, dtype=float).tolist()],
        "mo_coef": [[float(x) for x in row] for row in np.asarray(mo_coeff, dtype=float).tolist()],
        "nucl_num": int(mol.natm),
        "nucl_charge": [float(x) for x in mol.atom_charges().tolist()],
        "nucl_label": [str(mol.atom_pure_symbol(i)).capitalize() for i in range(mol.natm)],
        "nucl_coord_bohr": [[float(x) for x in row] for row in mol.atom_coords(unit="Bohr").T.tolist()],
        "hf_total_energy_eh": float(total_energy_eh),
        "nuclear_repulsion_eh": float(mol.energy_nuc()),
        "elec_alpha_num": int(mol.nelec[0]),
        "elec_beta_num": int(mol.nelec[1]),
        "nmo_total": int(mo_coeff.shape[1]),
        "ao_num": int(len(ao_labels)),
        "ao_basis_name": str(basis_name),
        "ao_cartesian": True,
        "ao_prim_num": [int(x) for x in ao_prim_num],
        "ao_nucl": [int(x) for x in ao_nucl],
        "ao_power": [[int(v) for v in row] for row in ao_power],
        "ao_coef": padded_ao_coef,
        "ao_expo": padded_ao_expo,
        "shell_num": int(shell_counter),
        "prim_num": int(len(prim_coef)),
        "shell_prim_num": [int(x) for x in shell_prim_num],
        "shell_ang_mom": [int(x) for x in shell_ang_mom],
        "shell_index": [int(x) for x in shell_index],
        "basis_nucleus_index": [int(x) for x in basis_nucleus_index],
        "nucleus_shell_num": [int(x) for x in nucleus_shell_num],
        "prim_coef": prim_coef,
        "prim_expo": prim_expo,
    }


def _load_or_create_system_setup(
    *,
    case: PolyeneCase,
    qp_root: Path,
    outdir: Path,
    force: bool,
) -> dict[str, object]:
    setup_dir = outdir / "system_setup"
    xyz_path = setup_dir / "polyene.xyz"
    ezfio_path = setup_dir / "polyene.ezfio"
    fcidump_path = setup_dir / "qp_active.FCIDUMP"
    metadata_path = setup_dir / "system_setup.json"
    if (
        not force
        and metadata_path.exists()
        and ezfio_path.exists()
        and fcidump_path.exists()
    ):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    if setup_dir.exists() and force:
        shutil.rmtree(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)

    qp = cr2_fixed.discover_qp_paths_from_root(qp_root)
    atom_text, _ = _polyene_atom(case)
    n2_study._create_xyz(xyz_path, atom_text, comment=case.title)
    if ezfio_path.exists():
        shutil.rmtree(ezfio_path)
    ezfio_path.mkdir(parents=True, exist_ok=True)
    ensure_ezfio_defaults(ezfio_path, qp_root=qp.qp_root)

    model = _build_active_orbital_model(case)
    payload = dict(model["payload"])
    for rel in ("ao_basis/ao_md5", "mo_basis/ao_md5"):
        target = ezfio_path / rel
        if target.exists():
            target.unlink()
    cr2_fixed._inject_pyscf_template_shell(qp, ezfio_path, payload, setup_dir / "pyscf_shell_payload.json")
    _set_mo_class_direct(qp, ezfio_path, mo_num=int(payload["mo_num"]), ncore=int(case.ncore), ncas=int(case.ncas))
    write_ezfio_value(ezfio_path, "ao_two_e_ints/do_ao_cholesky", True)
    write_ezfio_value(ezfio_path, "ao_two_e_ints/io_ao_cholesky", "Write")
    write_ezfio_value(ezfio_path, "mo_two_e_ints/do_mo_cholesky", False)
    write_ezfio_value(ezfio_path, "mo_two_e_ints/io_mo_cholesky", "None")
    dsp._force_integral_rebuild(qp, ezfio_path)

    from pyscf.tools import fcidump as pyscf_fcidump

    pyscf_fcidump.from_integrals(
        str(fcidump_path),
        model["h1"],
        model["eri8"],
        int(case.ncas),
        int(case.nelecas),
        nuc=float(model["ecore"]),
        ms=int(model["nelec_cas"][0] - model["nelec_cas"][1]),
    )

    metadata: dict[str, object] = {
        "case": case.name,
        "title": case.title,
        "basis": case.basis,
        "n_carbons": int(case.n_carbons),
        "ncas": int(case.ncas),
        "nelecas": int(case.nelecas),
        "ncore": int(case.ncore),
        "charge": int(case.charge),
        "spin": int(case.spin),
        "geometry_mode": case.geometry_mode,
        "cc_single_ang": float(case.cc_single_ang),
        "cc_double_ang": float(case.cc_double_ang),
        "ch_ang": float(case.ch_ang),
        "localizer_used": str(model["localizer_used"]),
        "active_window_start": int(model["active_window_start"]),
        "active_window_stop": int(model["active_window_stop"]),
        "active_centers": [float(x) for x in model["active_centers"]],
        "active_dominant_carbons": [int(x) for x in model["active_dominant_carbons"]],
        "active_dominant_weights": [float(x) for x in model["active_dominant_weights"]],
        "nmo_total": int(model["qp_mo_num"]),
        "qp_mo_num": int(model["qp_mo_num"]),
        "full_nmo_total": int(model["full_nmo_total"]),
        "ezfio_path": str(ezfio_path),
        "fcidump_path": str(fcidump_path),
        "xyz_path": str(xyz_path),
        "setup_kind": "pyscf_polyene_localized_pi_shell",
        "hf_total_energy_eh": float(model["mf"].e_tot),
        "ecore_eh": float(model["ecore"]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _prepare_case_qp(spec: dict[str, Any], *, case_dir: Path, force: bool = False) -> n2_study.PreparedQpCase:
    study_cfg = spec.get("study", {})
    omp_threads = int(study_cfg.get("omp_threads", 8))
    qp_root = _require_qp_root(spec)
    case = _case_from_spec(spec)
    setup = _load_or_create_system_setup(case=case, qp_root=qp_root, outdir=case_dir, force=force)
    setup_dir = case_dir / "system_setup"
    ezfio_path = Path(str(setup["ezfio_path"]))
    fcidump_path = Path(str(setup["fcidump_path"]))
    h1, eri8, norb, nelec, ecore = dsp._load_qp_fcidump_hamiltonian(fcidump_path)
    reference_summary, reference_choice = n2_study._resolve_qp_reference(
        spec=spec,
        qp_root=qp_root,
        ezfio_path=ezfio_path,
        fcidump_path=fcidump_path,
        setup_dir=setup_dir,
        h1=h1,
        eri8=eri8,
        norb=int(norb),
        nelec=tuple(nelec),
        ecore=float(ecore),
        omp_threads=int(omp_threads),
        force=force,
    )
    return n2_study.PreparedQpCase(
        qp_root=qp_root,
        setup_dir=setup_dir,
        ezfio_path=ezfio_path,
        fcidump_path=fcidump_path,
        ecore=float(ecore),
        norb=int(norb),
        nelec=(int(nelec[0]), int(nelec[1])),
        reference_total_eh=float(reference_summary["total_energy_eh"]),
        reference_summary=reference_summary,
        reference_mode=str(reference_choice["effective_reference_mode"]),
        omp_threads=int(omp_threads),
    )


def run_case(input_path: Path, *, force: bool = False) -> dict[str, Any]:
    input_path = input_path.resolve()
    spec = n2_study.load_study_input(input_path)
    case_dir = input_path.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    study_summary_path = case_dir / "study_summary.json"
    if study_summary_path.exists() and not force:
        return json.loads(study_summary_path.read_text(encoding="utf-8"))

    case_label = str(spec["label"])
    max_cumulative_iterations = int(spec["study"]["max_cumulative_iterations"])
    plot_max_cumulative_iterations = int(spec.get("study", {}).get("plot_max_cumulative_iterations", max_cumulative_iterations))
    stop_abs_delta_mha = (
        None if spec.get("study", {}).get("stop_abs_delta_mHa") is None else float(spec["study"]["stop_abs_delta_mHa"])
    )
    run_specs = spec["runs"]
    run_labels = {str(run["name"]): str(run["label"]) for run in run_specs}

    if force:
        n2_study._clean_case_outputs(
            case_dir,
            [str(run["name"]) for run in run_specs],
            remove_system_setup=True,
        )

    prepared = _prepare_case_qp(spec, case_dir=case_dir, force=force)
    run_summaries: dict[str, dict[str, Any]] = {}
    for run_spec in run_specs:
        run_name = str(run_spec["name"])
        run_label = str(run_spec["label"])
        summary = n2_study._run_qp_thick_trace(
            case_dir=case_dir,
            case_label=case_label,
            run_name=run_name,
            run_label=run_label,
            prepared=prepared,
            run_params=dict(run_spec["solver"]["params"]),
            max_cumulative_iterations=max_cumulative_iterations,
            plot_max_cumulative_iterations=plot_max_cumulative_iterations,
            stop_abs_delta_mha=stop_abs_delta_mha,
            run_labels=run_labels,
        )
        run_summaries[run_name] = summary

    comparison = n2_study.plot_case_comparison(
        case_dir=case_dir,
        case_label=case_label,
        run_labels=run_labels,
        max_cumulative_iterations=plot_max_cumulative_iterations,
    )
    payload = {
        "input_path": str(input_path),
        "case_dir": str(case_dir),
        "label": case_label,
        "backend": "qp",
        "reference_total_energy_eh": float(prepared.reference_total_eh),
        "reference_kind": prepared.reference_summary.get("reference_kind"),
        "reference_mode": prepared.reference_mode,
        "max_cumulative_iterations": max_cumulative_iterations,
        "plot_max_cumulative_iterations": plot_max_cumulative_iterations,
        "stop_abs_delta_mHa": stop_abs_delta_mha,
        "runs": run_summaries,
        "comparison_plot_png": None if comparison is None else str(comparison[0]),
        "comparison_plot_pdf": None if comparison is None else str(comparison[1]),
    }
    study_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a live-updating all-trans polyene QP thick study from input.json")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    payload = run_case(args.input, force=bool(args.force))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
