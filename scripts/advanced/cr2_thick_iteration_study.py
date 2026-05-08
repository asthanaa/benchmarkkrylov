"""Run one live-updating Cr2 QP thick/baseline study from an `input.json`.

This mirrors the QP-backed live trace workflow used for N2, but uses the
existing Cr2 setup helper to build the molecule, FCIDUMP, and DMRG reference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from ccik import build_cas_hamiltonian_pyscf, make_mol_pyscf  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from qp_ccik_common import validate_rebuildable_qp_root  # noqa: E402


def _require_qp_root(spec: dict[str, Any]) -> Path:
    study_cfg = spec.get("study", {})
    raw = study_cfg.get("qp_root")
    if raw is None:
        raise RuntimeError("Cr2 QP study requires study.qp_root")
    qp_root = Path(str(raw)).expanduser().resolve()
    validation = validate_rebuildable_qp_root(qp_root)
    if not validation.get("is_rebuildable"):
        raise RuntimeError(
            "Cr2 QP study requires a rebuildable Quantum Package root. "
            f"validation={json.dumps(validation, sort_keys=True)}"
        )
    return qp_root


def _case_from_spec(spec: dict[str, Any]) -> cr2_fixed.Cr2FixedBudgetCase:
    molecule = spec["molecule"]
    cas_cfg = spec["cas"]
    return cr2_fixed.Cr2FixedBudgetCase(
        name=str(spec.get("name", "cr2_case")).strip() or "cr2_case",
        title=str(spec["label"]),
        basis=str(molecule["basis"]),
        distance_ang=float(molecule["distance_ang"]),
        ncas=int(cas_cfg["ncas"]),
        nelecas=int(cas_cfg["nelecas"]),
        ncore=int(cas_cfg.get("ncore", 0)),
        fixed_config=cr2_fixed.CASE.fixed_config,
    )


def _build_reference_fcidump_from_pyscf(
    *,
    case: cr2_fixed.Cr2FixedBudgetCase,
    outdir: Path,
) -> Path:
    from pyscf import gto
    from pyscf.tools import fcidump as pyscf_fcidump

    mol = gto.M(
        atom=cr2_fixed.make_cr2_atom(float(case.distance_ang)),
        basis=str(case.basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
        cart=True,
    )
    cas = CASSpec(ncas=int(case.ncas), nelecas=int(case.nelecas), ncore=int(case.ncore))
    h1, eri8, ecore, norb, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
    fcidump_path = outdir / "qp_active.FCIDUMP"
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


def _load_or_create_system_setup(
    *,
    case: cr2_fixed.Cr2FixedBudgetCase,
    qp_root: Path,
    outdir: Path,
    force: bool,
) -> dict[str, Any]:
    paths = cr2_fixed._system_setup_paths(outdir)
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

    qp = cr2_fixed.discover_qp_paths_from_root(qp_root)
    nmo_total = int(cr2_fixed.cr2_spatial_orbitals(basis=case.basis, r_ang=case.distance_ang))
    cr2_fixed.create_xyz(paths["xyz"], cr2_fixed.make_cr2_atom(case.distance_ang), comment=case.title)
    if paths["ezfio"].exists():
        shutil.rmtree(paths["ezfio"])

    setup_details: dict[str, object]
    setup_backend = "qp_create_ezfio"
    setup_fallback_reason: str | None = None
    if cr2_fixed._qp_frontend_available(qp):
        try:
            cr2_fixed.run_qp_setup(
                qp,
                basis=case.basis,
                ezfio=paths["ezfio"],
                xyz=paths["xyz"],
                ncore=case.ncore,
                ncas=case.ncas,
                nmo_total=nmo_total,
            )
            setup_details = {"setup_kind": "qp_create_ezfio", "nmo_total": nmo_total}
        except Exception as exc:
            if paths["ezfio"].exists():
                shutil.rmtree(paths["ezfio"])
            setup_details = cr2_fixed._create_system_setup_via_template_shell(
                case=case,
                qp=qp,
                setup_dir=setup_dir,
                ezfio=paths["ezfio"],
            )
            setup_backend = "pyscf_constructed_shell_fallback"
            setup_fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        setup_details = cr2_fixed._create_system_setup_via_template_shell(
            case=case,
            qp=qp,
            setup_dir=setup_dir,
            ezfio=paths["ezfio"],
        )
        setup_backend = "pyscf_constructed_shell"

    cr2_fixed.ensure_ezfio_defaults(paths["ezfio"], qp_root=qp.qp_root)
    fcidump_path = _build_reference_fcidump_from_pyscf(case=case, outdir=setup_dir)
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
        "xyz_path": str(paths["xyz"]),
        "setup_backend": setup_backend,
        "setup_fallback_reason": setup_fallback_reason,
        "fcidump_backend": "pyscf_cas_hamiltonian",
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
    p = argparse.ArgumentParser(description="Run a live-updating Cr2 QP thick study from input.json")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    payload = run_case(args.input, force=bool(args.force))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
