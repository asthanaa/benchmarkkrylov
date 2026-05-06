"""Run one live-updating Mo2 QP thick/baseline study from an `input.json`.

This mirrors the QP-backed live trace workflow used for Cr2, but uses a
Mo2-specific molecule builder and a QP-exported active-space FCIDUMP to drive
the DMRG reference in the same style as the Cr2 campaign.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from ccik import build_cas_hamiltonian_pyscf  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    discover_qp_paths_from_root,
    ensure_ezfio_defaults,
    run_qp_setup,
    validate_rebuildable_qp_root,
)


@dataclass(frozen=True)
class Mo2FixedBudgetCase:
    name: str
    title: str
    basis: str
    distance_ang: float
    ncas: int
    nelecas: int
    ncore: int


MO2_TEMPLATE_SHELL_CANDIDATES = (
    ROOT / "inputs" / "n2" / "2.0" / "n2_cas14e18_R2p00_dmrg1000_baseline_m031_resume_t48" / "system_setup" / "n2.ezfio",
    ROOT / "inputs" / "n2" / "2.0" / "n2_cas14e18_R2p00_dmrg1000_exactized_m031_k1000_s001000_t4_dynamic" / "system_setup" / "n2.ezfio",
)

REQUIRED_TEMPLATE_SHELL_FILES = (
    "ao_basis/ao_md5",
    "ao_basis/ao_num",
    "ao_basis/ao_nucl.gz",
    "ao_basis/ao_prim_num.gz",
    "ao_basis/ao_coef.gz",
    "ao_basis/ao_expo.gz",
    "ao_basis/ao_power.gz",
    "basis/basis",
    "nuclei/nucl_num",
)


def make_mo2_atom(r_ang: float) -> str:
    z = float(r_ang) / 2.0
    return f"""
    Mo 0.0 0.0 {-z}
    Mo 0.0 0.0 {+z}
    """


def create_xyz(path: Path, atom: str, *, comment: str) -> None:
    lines = [line.strip() for line in atom.splitlines() if line.strip()]
    path.write_text(
        f"{len(lines)}\n{comment}\n" + "\n".join(lines) + "\n",
        encoding="ascii",
    )


def mo2_spatial_orbitals(*, basis: str, r_ang: float) -> int:
    from pyscf import gto

    mol = gto.M(atom=make_mo2_atom(r_ang), basis=basis, unit="Angstrom", charge=0, spin=0, verbose=0)
    return int(mol.nao_nr())


def _require_qp_root(spec: dict[str, Any]) -> Path:
    study_cfg = spec.get("study", {})
    raw = study_cfg.get("qp_root")
    if raw is None:
        raise RuntimeError("Mo2 QP study requires study.qp_root")
    qp_root = Path(str(raw)).expanduser().resolve()
    validation = validate_rebuildable_qp_root(qp_root)
    if not validation.get("is_rebuildable"):
        raise RuntimeError(
            "Mo2 QP study requires a rebuildable Quantum Package root. "
            f"validation={json.dumps(validation, sort_keys=True)}"
        )
    return qp_root


def _case_from_spec(spec: dict[str, Any]) -> Mo2FixedBudgetCase:
    molecule = spec["molecule"]
    cas_cfg = spec["cas"]
    return Mo2FixedBudgetCase(
        name=str(spec.get("name", "mo2_case")).strip() or "mo2_case",
        title=str(spec["label"]),
        basis=str(molecule["basis"]),
        distance_ang=float(molecule["distance_ang"]),
        ncas=int(cas_cfg["ncas"]),
        nelecas=int(cas_cfg["nelecas"]),
        ncore=int(cas_cfg.get("ncore", 0)),
    )


def _system_setup_paths(outdir: Path) -> dict[str, Path]:
    setup_dir = outdir / "system_setup"
    return {
        "dir": setup_dir,
        "xyz": setup_dir / "mo2.xyz",
        "ezfio": setup_dir / "mo2.ezfio",
        "fcidump": setup_dir / "qp_active.FCIDUMP",
        "pyscf_fcidump": setup_dir / "pyscf_active.FCIDUMP",
        "metadata": setup_dir / "system_setup.json",
        "reference_correction": setup_dir / "reference_correction.json",
    }


def _build_reference_fcidump_from_pyscf(
    *,
    case: Mo2FixedBudgetCase,
    outdir: Path,
    filename: str = "qp_active.FCIDUMP",
) -> Path:
    from pyscf import gto
    from pyscf.tools import fcidump as pyscf_fcidump

    mol = gto.M(
        atom=make_mo2_atom(float(case.distance_ang)),
        basis=str(case.basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    cas = CASSpec(ncas=int(case.ncas), nelecas=int(case.nelecas), ncore=int(case.ncore))
    h1, eri8, ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
    fcidump_path = outdir / str(filename)
    pyscf_fcidump.from_integrals(
        str(fcidump_path),
        h1,
        eri8,
        int(norb),
        int(nelec[0] + nelec[1]),
        nuc=float(ecore),
        ms=int(nelec[0] - nelec[1]),
    )
    return fcidump_path


def _missing_template_shell_files(path: Path) -> list[str]:
    return [rel for rel in REQUIRED_TEMPLATE_SHELL_FILES if not (path / rel).exists()]


def _materialize_template_shell(target: Path) -> Path:
    for donor in MO2_TEMPLATE_SHELL_CANDIDATES:
        if not donor.exists():
            continue
        missing = _missing_template_shell_files(donor)
        if missing:
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(donor, target)
        return target
    raise FileNotFoundError(
        "No complete EZFIO template shell available for Mo2 fallback. "
        f"checked={[str(path) for path in MO2_TEMPLATE_SHELL_CANDIDATES]}"
    )


def _build_rhf_shell_payload(*, atom: str, basis: str, title: str) -> dict[str, object]:
    from pyscf import gto, scf

    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
        cart=True,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF RHF did not converge for {title}")
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
                f"PySCF contraction matrix shape mismatch for {title}: "
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
                    raise RuntimeError(f"AO label cursor overflow while building shell for {title}")
                component = str(ao_labels[ao_cursor][3]).strip()
                power = [component.count("x"), component.count("y"), component.count("z")]
                ao_prim_num.append(nprim)
                ao_nucl.append(atom_index)
                ao_power.append(power)
                ao_coef.append(coeffs)
                ao_expo.append(expos)
                ao_cursor += 1
    if ao_cursor != len(ao_labels):
        raise RuntimeError(f"AO label cursor mismatch while building shell for {title}")

    ao_prim_num_max = max(ao_prim_num) if ao_prim_num else 0
    padded_ao_coef = [row + [0.0] * (ao_prim_num_max - len(row)) for row in ao_coef]
    padded_ao_expo = [row + [0.0] * (ao_prim_num_max - len(row)) for row in ao_expo]
    nucleus_shell_num = [basis_nucleus_index.count(atom_idx + 1) for atom_idx in range(mol.natm)]
    return {
        "mo_num": int(mf.mo_coeff.shape[1]),
        "mo_label": f"PySCF_RHF_{basis.upper().replace('-', '')}",
        "mo_occ": [float(x) for x in mf.mo_occ.tolist()],
        "mo_coef": [[float(x) for x in row] for row in mf.mo_coeff.tolist()],
        "nucl_num": int(mol.natm),
        "nucl_charge": [float(x) for x in mol.atom_charges().tolist()],
        "nucl_label": [str(mol.atom_pure_symbol(i)).capitalize() for i in range(mol.natm)],
        "nucl_coord_bohr": [[float(x) for x in row] for row in mol.atom_coords(unit="Bohr").T.tolist()],
        "hf_total_energy_eh": float(mf.e_tot),
        "nuclear_repulsion_eh": float(mol.energy_nuc()),
        "elec_alpha_num": int(mol.nelec[0]),
        "elec_beta_num": int(mol.nelec[1]),
        "nmo_total": int(mf.mo_coeff.shape[1]),
        "ao_num": int(len(ao_labels)),
        "ao_basis_name": str(basis),
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


def _create_system_setup_via_template_shell(
    *,
    case: Mo2FixedBudgetCase,
    qp,
    setup_dir: Path,
    ezfio: Path,
) -> dict[str, object]:
    _materialize_template_shell(ezfio)
    ensure_ezfio_defaults(ezfio, qp_root=qp.qp_root)
    payload = _build_rhf_shell_payload(
        atom=make_mo2_atom(float(case.distance_ang)),
        basis=str(case.basis),
        title=str(case.title),
    )
    for rel in ("ao_basis/ao_md5", "mo_basis/ao_md5"):
        target = ezfio / rel
        if target.exists():
            target.unlink()
    cr2_fixed._inject_pyscf_template_shell(qp, ezfio, payload, setup_dir / "pyscf_shell_payload.json")
    cr2_fixed._set_mo_class_absolute(qp, ezfio, int(payload["mo_num"]), int(case.ncore), int(case.ncas))
    dsp._force_integral_rebuild(qp, ezfio)
    return {
        "setup_kind": "pyscf_constructed_shell",
        "hf_total_energy_eh": float(payload["hf_total_energy_eh"]),
        "nmo_total": int(payload["nmo_total"]),
    }


def _load_or_create_system_setup(
    *,
    case: Mo2FixedBudgetCase,
    qp_root: Path,
    outdir: Path,
    force: bool,
) -> dict[str, Any]:
    paths = _system_setup_paths(outdir)
    metadata_path = paths["metadata"]
    if (
        not force
        and metadata_path.exists()
        and paths["ezfio"].exists()
        and paths["fcidump"].exists()
    ):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    setup_dir = paths["dir"]
    if setup_dir.exists() and force:
        shutil.rmtree(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)

    qp = discover_qp_paths_from_root(qp_root)
    nmo_total = int(mo2_spatial_orbitals(basis=case.basis, r_ang=case.distance_ang))
    create_xyz(paths["xyz"], make_mo2_atom(case.distance_ang), comment=case.title)
    if paths["ezfio"].exists():
        shutil.rmtree(paths["ezfio"])

    setup_backend = "fresh_qp_setup"
    setup_fallback_reason: str | None = None
    setup_details: dict[str, object] = {"setup_kind": "qp_create_ezfio", "nmo_total": nmo_total}
    try:
        run_qp_setup(
            qp,
            basis=case.basis,
            ezfio=paths["ezfio"],
            xyz=paths["xyz"],
            ncore=case.ncore,
            ncas=case.ncas,
            nmo_total=nmo_total,
        )
    except Exception as exc:
        if paths["ezfio"].exists():
            shutil.rmtree(paths["ezfio"])
        setup_details = _create_system_setup_via_template_shell(
            case=case,
            qp=qp,
            setup_dir=setup_dir,
            ezfio=paths["ezfio"],
        )
        setup_backend = "pyscf_constructed_shell_fallback"
        setup_fallback_reason = f"{type(exc).__name__}: {exc}"
    ensure_ezfio_defaults(paths["ezfio"], qp_root=qp.qp_root)
    pyscf_fcidump_path = _build_reference_fcidump_from_pyscf(
        case=case,
        outdir=setup_dir,
        filename=paths["pyscf_fcidump"].name,
    )
    fcidump_path = dsp._generate_qp_fcidump(qp, paths["ezfio"], setup_dir, case_spec=None)
    payload: dict[str, Any] = {
        "case": case.name,
        "title": case.title,
        "basis": case.basis,
        "distance_ang": case.distance_ang,
        "ncas": case.ncas,
        "nelecas": case.nelecas,
        "ncore": case.ncore,
        "nmo_total": nmo_total,
        "ezfio_path": str(paths["ezfio"]),
        "fcidump_path": str(fcidump_path),
        "pyscf_reference_fcidump_path": str(pyscf_fcidump_path),
        "xyz_path": str(paths["xyz"]),
        "setup_backend": setup_backend,
        "setup_fallback_reason": setup_fallback_reason,
        "fcidump_backend": "qp_export_active_space",
        "pyscf_reference_fcidump_backend": "pyscf_cas_hamiltonian",
        "reference_correction_path": str(paths["reference_correction"]),
    }
    payload.update(setup_details)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _prepare_case_qp(spec: dict[str, Any], *, case_dir: Path, force: bool = False) -> n2_study.PreparedQpCase:
    study_cfg = spec.get("study", {})
    omp_threads = int(study_cfg.get("omp_threads", 8))
    qp_root = _require_qp_root(spec)
    case = _case_from_spec(spec)
    setup = _load_or_create_system_setup(
        case=case,
        qp_root=qp_root,
        outdir=case_dir,
        force=force,
    )
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
    reference_total_eh = float(reference_summary["total_energy_eh"])
    correction_path = setup_dir / "reference_correction.json"
    if correction_path.exists():
        correction_payload = json.loads(correction_path.read_text(encoding="utf-8"))
        corrected_total = correction_payload.get("corrected_reference_total_energy_eh")
        if corrected_total is not None:
            reference_total_eh = float(corrected_total)
            reference_summary = dict(reference_summary)
            reference_summary["raw_reference_total_energy_eh"] = float(reference_summary["total_energy_eh"])
            reference_summary["total_energy_eh"] = float(reference_total_eh)
            reference_summary["reference_correction_path"] = str(correction_path)
            reference_summary["corrected_reference_kind"] = correction_payload.get("corrected_reference_kind")
    return n2_study.PreparedQpCase(
        qp_root=qp_root,
        setup_dir=setup_dir,
        ezfio_path=ezfio_path,
        fcidump_path=fcidump_path,
        ecore=float(ecore),
        norb=int(norb),
        nelec=(int(nelec[0]), int(nelec[1])),
        reference_total_eh=float(reference_total_eh),
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
    p = argparse.ArgumentParser(description="Run a live-updating Mo2 QP thick study from input.json")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    payload = run_case(args.input, force=bool(args.force))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
