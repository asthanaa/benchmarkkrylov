from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import n2_qp_fixed_budget_sweep_plots as n2_fixed  # noqa: E402
import n2_qp_threshold_sweep_plots as threshold  # noqa: E402
import debug_selector_parity as dsp  # noqa: E402
from ccik import make_mol_pyscf  # noqa: E402
from cr2_qp_ccik_tune import create_xyz, cr2_spatial_orbitals, make_cr2_atom  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    CCIKParams as QPCCIKParams,
    discover_qp_paths_from_root,
    ensure_ezfio_defaults,
    qp_parallel_env,
    rebuild_ccik_plugin,
    resolve_omp_threads,
    run_qp_setup,
    set_ccik_params,
    validate_fresh_qp_root,
    write_ezfio_value,
)


THRESHOLD_TEST_ROOT = threshold.THRESHOLD_TEST_ROOT
DEFAULT_DMRG_PYTHON = THRESHOLD_TEST_ROOT / "block2_cr2_sto3g_R1p68_cas28_20" / ".venv" / "bin" / "python"
DEFAULT_STORE_KEEP_GRID = (32_768, 65_536, 131_072, 262_144, 524_288)
DEFAULT_PROPAGATE_KEEP_GRID = (1_024, 2_048, 4_096, 8_192, 16_384)
INTERNAL_EXTERNAL_KEEP = 1_000_000
CR2_TEMPLATE_GIT_PATH = "experiments/cr2_qp_campaign_20260314/repro_w0_f4_rel/cr2.ezfio"
CR2_TEMPLATE_FALLBACKS = (
    Path("/private/tmp/cr2_qprun_direct.ixCz9d/cr2.ezfio"),
    Path("/private/tmp/cr2_qprun_direct.fC1MNw/cr2.ezfio"),
    Path("/private/tmp/cr2_cipsi_direct.RPfzyZ/cr2.ezfio"),
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


@dataclass(frozen=True)
class Cr2FixedBudgetCase:
    name: str
    title: str
    basis: str
    distance_ang: float
    ncas: int
    nelecas: int
    ncore: int
    fixed_config: threshold.StructuralConfig
    fcidump_path: Path | None = None


CASE = Cr2FixedBudgetCase(
    name="cr2_sto3g_R1p68_cas28_20",
    title="Cr2 / STO-3G / R = 1.68 A / CAS(28e,20o)",
    basis="sto-3g",
    distance_ang=1.68,
    ncas=20,
    nelecas=28,
    ncore=10,
    fixed_config=threshold.StructuralConfig(
        method="thick",
        tag="restart_50_60_12",
        basis_size=24,
        max_cycle=60,
        restart_m_cycle=50,
        restart_nroot=12,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="Fresh-setup Cr2 fixed-budget thick sweep with DMRG reference")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--qp-root", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--omp-threads", default="auto")
    p.add_argument("--dmrg-memory-gb", type=int, default=96)
    p.add_argument("--dmrg-threads", type=int, default=16)
    p.add_argument("--dmrg-conv-tol", type=float, default=1e-8)
    p.add_argument("--dmrg-python", type=Path, default=DEFAULT_DMRG_PYTHON)
    p.add_argument("--fixed-store-keep", type=int, default=131_072)
    p.add_argument("--fixed-propagate-keep", type=int, default=4_096)
    p.add_argument(
        "--store-grid",
        default=None,
        help="Comma-separated fixed_store_keep sweep values. Default: 32768,65536,131072,262144,524288",
    )
    p.add_argument(
        "--propagate-grid",
        default=None,
        help="Comma-separated fixed_propagate_keep sweep values. Default: 1024,2048,4096,8192,16384",
    )
    args = p.parse_args(argv)
    if args.outdir is None:
        args.outdir = THRESHOLD_TEST_ROOT / f"cr2_qp_{CASE.name}_fixed_budget_centered_{stamp}"
    args.fixed_store_keep = int(args.fixed_store_keep)
    args.fixed_propagate_keep = int(args.fixed_propagate_keep)
    if args.fixed_store_keep < 1 or args.fixed_propagate_keep < 1:
        raise ValueError("fixed-budget anchor values must be positive")
    args.store_grid = n2_fixed._parse_int_grid(args.store_grid, DEFAULT_STORE_KEEP_GRID)
    args.propagate_grid = n2_fixed._parse_int_grid(args.propagate_grid, DEFAULT_PROPAGATE_KEEP_GRID)
    return args


def _system_setup_paths(outdir: Path) -> dict[str, Path]:
    setup_dir = outdir / "system_setup"
    return {
        "dir": setup_dir,
        "xyz": setup_dir / "cr2.xyz",
        "ezfio": setup_dir / "cr2.ezfio",
        "fcidump": setup_dir / "qp_active.FCIDUMP",
        "metadata": setup_dir / "system_setup.json",
        "fcidump_stdout": setup_dir / "fcidump_stdout.log",
        "fcidump_stderr": setup_dir / "fcidump_stderr.log",
    }


def _qp_frontend_available(qp) -> bool:
    qp_create_ezfio = Path(str(qp.qp_create_ezfio))
    create_ok = qp_create_ezfio.is_file() and os.access(qp_create_ezfio, os.X_OK)
    set_class_ok = qp.qp_set_mo_class.is_file() and os.access(qp.qp_set_mo_class, os.X_OK)
    if create_ok and set_class_ok:
        return True
    run = subprocess.run(
        [
            "/bin/bash",
            "-lc",
            (
                f"source '{qp.quantum_package_rc}' >/dev/null 2>&1 && "
                "command -v qp_create_ezfio >/dev/null 2>&1 && "
                "command -v qp_set_mo_class >/dev/null 2>&1"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
    )
    return run.returncode == 0


def _extract_git_tree(pathspec: str, destination: Path) -> None:
    run = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD", pathspec],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(run.stdout), mode="r:") as tf:
        tf.extractall(path=destination)


def _missing_template_shell_files(path: Path) -> list[str]:
    return [rel for rel in REQUIRED_TEMPLATE_SHELL_FILES if not (path / rel).exists()]


def _ensure_complete_template_shell(path: Path) -> None:
    missing = _missing_template_shell_files(path)
    if missing:
        raise RuntimeError(
            "Incomplete Cr2 template EZFIO shell. "
            f"path={path}; missing={missing}"
        )


def _materialize_template_shell(target: Path) -> Path:
    worktree_template = ROOT / CR2_TEMPLATE_GIT_PATH
    if worktree_template.exists():
        _ensure_complete_template_shell(worktree_template)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(worktree_template, target)
        return target

    for fallback in CR2_TEMPLATE_FALLBACKS:
        if fallback.exists():
            if _missing_template_shell_files(fallback):
                continue
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(fallback, target)
            return target

    extract_root = target.parent / "_template_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    _extract_git_tree(CR2_TEMPLATE_GIT_PATH, extract_root)
    extracted = extract_root / CR2_TEMPLATE_GIT_PATH
    if not extracted.exists():
        raise RuntimeError(
            "Failed to materialize template EZFIO. "
            f"git path missing: {CR2_TEMPLATE_GIT_PATH}; "
            f"fallbacks checked: {[str(path) for path in CR2_TEMPLATE_FALLBACKS]}"
        )
    _ensure_complete_template_shell(extracted)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(extracted), str(target))
    shutil.rmtree(extract_root, ignore_errors=True)
    return target


def _build_cr2_rhf_payload(case: Cr2FixedBudgetCase, template_ezfio: Path) -> dict[str, object]:
    from pyscf import gto, scf

    mol = gto.M(
        atom=make_cr2_atom(float(case.distance_ang)),
        basis=case.basis,
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
        raise RuntimeError(f"PySCF RHF did not converge for {case.title}")
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

    for ib in range(mol.nbas):
        atom_index = int(mol.bas_atom(ib)) + 1
        nprim = int(mol.bas_nprim(ib))
        nctr = int(mol.bas_nctr(ib))
        if nctr != 1:
            raise RuntimeError(f"Unsupported PySCF shell with nctr={nctr}; expected 1 for {case.title}")
        l = int(mol.bas_angular(ib))
        expos = [float(x) for x in mol.bas_exp(ib).tolist()]
        ctr_coeff = mol.bas_ctr_coeff(ib)
        coeffs = [float(x) for x in ctr_coeff[:, 0].tolist()]
        shell_prim_num.append(nprim)
        shell_ang_mom.append(l)
        basis_nucleus_index.append(atom_index)
        prim_coef.extend(coeffs)
        prim_expo.extend(expos)
        shell_index.extend([ib + 1] * nprim)
        for _ in range(int(mol.bas_len_cart(ib))):
            if ao_cursor >= len(ao_labels):
                raise RuntimeError(f"AO label cursor overflow while building Cr2 shell for {case.title}")
            component = str(ao_labels[ao_cursor][3]).strip()
            power = [component.count("x"), component.count("y"), component.count("z")]
            ao_prim_num.append(nprim)
            ao_nucl.append(atom_index)
            ao_power.append(power)
            ao_coef.append(coeffs)
            ao_expo.append(expos)
            ao_cursor += 1
    if ao_cursor != len(ao_labels):
        raise RuntimeError(f"AO label cursor mismatch while building Cr2 shell for {case.title}")

    ao_prim_num_max = max(ao_prim_num) if ao_prim_num else 0
    padded_ao_coef = [row + [0.0] * (ao_prim_num_max - len(row)) for row in ao_coef]
    padded_ao_expo = [row + [0.0] * (ao_prim_num_max - len(row)) for row in ao_expo]
    nucleus_shell_num = [basis_nucleus_index.count(atom_idx + 1) for atom_idx in range(mol.natm)]
    return {
        "mo_num": int(mf.mo_coeff.shape[1]),
        "mo_label": f"PySCF_RHF_{case.basis.upper().replace('-', '')}",
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
        "ao_basis_name": str(case.basis),
        "ao_cartesian": True,
        "ao_prim_num": [int(x) for x in ao_prim_num],
        "ao_nucl": [int(x) for x in ao_nucl],
        "ao_power": [[int(v) for v in row] for row in ao_power],
        "ao_coef": padded_ao_coef,
        "ao_expo": padded_ao_expo,
        "shell_num": int(mol.nbas),
        "prim_num": int(len(prim_coef)),
        "shell_prim_num": [int(x) for x in shell_prim_num],
        "shell_ang_mom": [int(x) for x in shell_ang_mom],
        "shell_index": [int(x) for x in shell_index],
        "basis_nucleus_index": [int(x) for x in basis_nucleus_index],
        "nucleus_shell_num": [int(x) for x in nucleus_shell_num],
        "prim_coef": prim_coef,
        "prim_expo": prim_expo,
    }


def _inject_pyscf_template_shell(qp, ezfio: Path, payload: dict[str, object], payload_path: Path) -> None:
    payload_path = payload_path.resolve()
    ezfio = ezfio.resolve()
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    script = f"""
import json
from pathlib import Path
from ezfio import ezfio

payload = json.loads(Path(r"{payload_path}").read_text())
ezfio.set_file(r"{ezfio}")
ezfio.set_nuclei_nucl_num(int(payload["nucl_num"]))
ezfio.set_nuclei_nucl_charge([float(x) for x in payload["nucl_charge"]])
ezfio.set_nuclei_nucl_label([str(x) for x in payload["nucl_label"]])
ezfio.set_nuclei_nucl_coord(payload["nucl_coord_bohr"])
ezfio.set_nuclei_nuclear_repulsion(float(payload["nuclear_repulsion_eh"]))
ezfio.set_nuclei_io_nuclear_repulsion("None")
ezfio.set_nuclei_is_periodic(False)
ezfio.set_nuclei_point_charges(False)
ezfio.set_basis_typ("Gaussian")
ezfio.set_basis_basis(str(payload["ao_basis_name"]))
ezfio.set_basis_ao_normalized(True)
ezfio.set_basis_primitives_normalized(True)
ezfio.set_basis_shell_num(int(payload["shell_num"]))
ezfio.set_basis_prim_num(int(payload["prim_num"]))
ezfio.set_basis_shell_prim_num([int(x) for x in payload["shell_prim_num"]])
ezfio.set_basis_shell_ang_mom([int(x) for x in payload["shell_ang_mom"]])
ezfio.set_basis_shell_index([int(x) for x in payload["shell_index"]])
ezfio.set_basis_basis_nucleus_index([int(x) for x in payload["basis_nucleus_index"]])
ezfio.set_basis_nucleus_shell_num([int(x) for x in payload["nucleus_shell_num"]])
ezfio.set_basis_prim_coef([float(x) for x in payload["prim_coef"]])
ezfio.set_basis_prim_expo([float(x) for x in payload["prim_expo"]])
ezfio.set_ao_basis_ao_num(int(payload["ao_num"]))
ezfio.set_ao_basis_ao_basis(str(payload["ao_basis_name"]))
ezfio.set_ao_basis_ao_prim_num([int(x) for x in payload["ao_prim_num"]])
ezfio.set_ao_basis_ao_nucl([int(x) for x in payload["ao_nucl"]])
ezfio.set_ao_basis_ao_power([[int(v) for v in row] for row in payload["ao_power"]])
ezfio.set_ao_basis_ao_coef([[float(v) for v in row] for row in payload["ao_coef"]])
ezfio.set_ao_basis_ao_expo([[float(v) for v in row] for row in payload["ao_expo"]])
ezfio.set_ao_basis_ao_cartesian(bool(payload["ao_cartesian"]))
ezfio.set_ao_basis_use_cgtos(False)
ezfio.set_electrons_elec_alpha_num(int(payload["elec_alpha_num"]))
ezfio.set_electrons_elec_beta_num(int(payload["elec_beta_num"]))
ezfio.set_mo_basis_mo_num(int(payload["mo_num"]))
ezfio.set_mo_basis_mo_label(str(payload["mo_label"]))
ezfio.set_mo_basis_mo_occ([float(x) for x in payload["mo_occ"]])
ezfio.set_mo_basis_mo_coef(payload["mo_coef"])
ezfio.set_mo_basis_mo_symmetry([1] * int(payload["mo_num"]))
"""
    subprocess.run(
        ["/bin/bash", "-lc", f"source '{qp.quantum_package_rc}' >/dev/null 2>&1 && python3 - <<'PY'\n{script}\nPY"],
        check=True,
        text=True,
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
    )


def _set_mo_class_absolute(qp, ezfio: Path, mo_num: int, ncore: int, ncas: int) -> None:
    ezfio = ezfio.resolve()
    write_ezfio_value(ezfio, "mo_basis/mo_num", int(mo_num))
    script = f"""
from ezfio import ezfio
ezfio.set_file(r"{ezfio}")
mo_num = int(ezfio.get_mo_basis_mo_num())
mo_class = []
for i in range(1, mo_num + 1):
    if i <= {int(ncore)}:
        mo_class.append("Core")
    elif i <= {int(ncore + ncas)}:
        mo_class.append("Active")
    else:
        mo_class.append("Deleted")
ezfio.set_mo_basis_mo_class(mo_class)
ezfio.set_mo_basis_mo_symmetry([1] * mo_num)
"""
    subprocess.run(
        ["/bin/bash", "-lc", f"source '{qp.quantum_package_rc}' >/dev/null 2>&1 && python3 - <<'PY'\n{script}\nPY"],
        check=True,
        text=True,
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
    )


def _create_system_setup_via_template_shell(
    *,
    case: Cr2FixedBudgetCase,
    qp,
    setup_dir: Path,
    ezfio: Path,
) -> dict[str, object]:
    ezfio.mkdir(parents=True, exist_ok=True)
    ensure_ezfio_defaults(ezfio, qp_root=qp.qp_root)
    payload = _build_cr2_rhf_payload(case, ezfio)
    for rel in ("ao_basis/ao_md5", "mo_basis/ao_md5"):
        target = ezfio / rel
        if target.exists():
            target.unlink()
    _inject_pyscf_template_shell(qp, ezfio, payload, setup_dir / "pyscf_shell_payload.json")
    _set_mo_class_absolute(qp, ezfio, int(payload["mo_num"]), int(case.ncore), int(case.ncas))
    dsp._force_integral_rebuild(qp, ezfio)
    return {
        "setup_kind": "pyscf_constructed_shell",
        "hf_total_energy_eh": float(payload["hf_total_energy_eh"]),
        "nmo_total": int(payload["nmo_total"]),
    }


def _export_fcidump(qp, ezfio: Path, outdir: Path) -> Path:
    env = {**os.environ, "QP_ROOT": str(qp.qp_root), **qp_parallel_env(1)}
    quoted_rc = str(qp.quantum_package_rc.resolve())
    quoted_binary = str((qp.qp_root / "src" / "tools" / "fcidump").resolve())
    run = subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"source '{quoted_rc}' >/dev/null 2>&1 && '{quoted_binary}' '{ezfio.name}'",
        ],
        cwd=str(outdir),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    (outdir / "fcidump_stdout.log").write_text(run.stdout, encoding="utf-8")
    (outdir / "fcidump_stderr.log").write_text(run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(
            "FCIDUMP export failed.\n"
            f"stdout_path={outdir / 'fcidump_stdout.log'}\n"
            f"stderr_path={outdir / 'fcidump_stderr.log'}"
        )
    raw_candidates = [
        outdir / "FCIDUMP",
        Path(str(ezfio) + ".FCIDUMP"),
        outdir / f"{ezfio.name}.FCIDUMP",
    ]
    raw = next((candidate for candidate in raw_candidates if candidate.exists()), None)
    if raw is None:
        raise RuntimeError(
            "FCIDUMP export completed but no output file was found.\n"
            f"checked={','.join(str(path) for path in raw_candidates)}\n"
            f"stdout_path={outdir / 'fcidump_stdout.log'}\n"
            f"stderr_path={outdir / 'fcidump_stderr.log'}"
        )
    target = outdir / "qp_active.FCIDUMP"
    if raw != target:
        raw.replace(target)
    return target


def _load_or_create_system_setup(
    *,
    case: Cr2FixedBudgetCase,
    qp_root: Path,
    outdir: Path,
    force: bool,
) -> dict[str, object]:
    paths = _system_setup_paths(outdir)
    if (
        not force
        and paths["metadata"].exists()
        and paths["ezfio"].exists()
        and paths["fcidump"].exists()
    ):
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    setup_dir = paths["dir"]
    if setup_dir.exists() and force:
        shutil.rmtree(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)

    qp = discover_qp_paths_from_root(qp_root)
    nmo_total = int(cr2_spatial_orbitals(basis=case.basis, r_ang=case.distance_ang))
    create_xyz(paths["xyz"], make_cr2_atom(case.distance_ang), comment=case.title)
    if paths["ezfio"].exists():
        shutil.rmtree(paths["ezfio"])
    setup_details: dict[str, object]
    if _qp_frontend_available(qp):
        run_qp_setup(
            qp,
            basis=case.basis,
            ezfio=paths["ezfio"],
            xyz=paths["xyz"],
            ncore=case.ncore,
            ncas=case.ncas,
            nmo_total=nmo_total,
        )
        setup_details = {"setup_kind": "qp_create_ezfio", "nmo_total": nmo_total}
    else:
        setup_details = _create_system_setup_via_template_shell(
            case=case,
            qp=qp,
            setup_dir=setup_dir,
            ezfio=paths["ezfio"],
        )
    ensure_ezfio_defaults(paths["ezfio"], qp_root=qp.qp_root)
    fcidump_path = _export_fcidump(qp, paths["ezfio"], setup_dir)
    payload = {
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
    }
    payload.update(setup_details)
    paths["metadata"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _normalize_reference_summary(summary: dict[str, object]) -> dict[str, object]:
    return threshold._normalize_reference_summary(summary)


def _normalized_c1_isym(value: object) -> int:
    isym = int(value or 0)
    return 1 if isym <= 0 else isym


def _load_or_compute_dmrg_reference(
    *,
    fcidump_path: Path,
    outdir: Path,
    force: bool,
    memory_gb: int,
    threads: int,
    conv_tol: float,
    dmrg_python: Path,
) -> dict[str, object]:
    path = outdir / "reference_dmrg_summary.json"
    if path.exists() and not force:
        return _normalize_reference_summary(json.loads(path.read_text(encoding="utf-8")))
    if not dmrg_python.exists():
        raise FileNotFoundError(f"DMRG Python interpreter not found: {dmrg_python}")

    from pyscf.tools import fcidump as pyscf_fcidump

    runtime_dir = outdir / "reference_dmrg_runtime"
    scratch_dir = runtime_dir / "scratch"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "dmrg.log"
    stdout_path = runtime_dir / "subprocess.stdout"
    stderr_path = runtime_dir / "subprocess.stderr"
    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data["MS2"])
    ecore = float(data.get("ECORE", 0.0))
    orb_sym = [int(x) for x in data.get("ORBSYM", [1] * norb)]
    isym = _normalized_c1_isym(data.get("ISYM", 0))
    hf_occs = threshold._hf_occ_vector(norb, nelec)

    bond_dims = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6000]
    noises = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 0.0, 0.0, 0.0]
    dav_thrds = [1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9, 1e-10, 1e-10]
    script = r"""
import contextlib
import json
import os
import time
from pathlib import Path
from pyblock2.gsdmrg import GSDMRG
from pyscf.tools import fcidump as pyscf_fcidump

fcidump_path = Path(os.environ["CKA_FCIDUMP"])
scratch_dir = Path(os.environ["CKA_SCRATCH"])
log_path = Path(os.environ["CKA_LOG"])
memory_gb = int(os.environ["CKA_MEMORY_GB"])
threads = int(os.environ["CKA_THREADS"])
conv_tol = float(os.environ["CKA_CONV_TOL"])
bond_dims = json.loads(os.environ["CKA_BOND_DIMS"])
noises = json.loads(os.environ["CKA_NOISES"])
dav_thrds = json.loads(os.environ["CKA_DAV_THRDS"])
n_steps = int(os.environ["CKA_N_STEPS"])
hf_occs = json.loads(os.environ["CKA_OCCS_JSON"])

data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
norb = int(data["NORB"])
nelec = int(data["NELEC"])
ms2 = int(data["MS2"])
orb_sym = [int(x) for x in data.get("ORBSYM", [1] * norb)]
isym = int(os.environ["CKA_ISYM"])
ecore = float(data.get("ECORE", 0.0))
t0 = time.perf_counter()
init_mode = "fci_init"

def run_dmrg(occs, bias, label):
    local_scratch = scratch_dir / label
    local_scratch.mkdir(parents=True, exist_ok=True)
    dmrg = GSDMRG(
        scratch=str(local_scratch),
        memory=float(memory_gb) * 1e9,
        omp_threads=int(threads),
        verbose=3,
        print_statistics=True,
    )
    dmrg.init_hamiltonian(
        "c1",
        n_sites=norb,
        n_elec=nelec,
        twos=ms2,
        isym=isym,
        orb_sym=orb_sym,
        e_core=ecore,
        h1e=data["H1"],
        g2e=data["H2"],
    )
    return float(
        dmrg.dmrg(
            bond_dims=bond_dims,
            noises=noises,
            dav_thrds=dav_thrds,
            occs=occs,
            bias=bias,
            n_steps=n_steps,
            conv_tol=conv_tol,
            cutoff=1e-14,
        )
    )

old_tmpdir = os.environ.get("TMPDIR")
os.environ["TMPDIR"] = str(scratch_dir)
try:
    with log_path.open("w", encoding="utf-8") as fh:
        with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            try:
                e_total = run_dmrg(None, 1.0, "fci_init")
            except RuntimeError as exc:
                if "Cannot generate initial guess 0" not in str(exc):
                    raise
                init_mode = "hf_occ_fallback"
                print("Retrying DMRG with HF occupation-number initialized MPS")
                e_total = run_dmrg(hf_occs, 0.0, "hf_occ_init")
finally:
    if old_tmpdir is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = old_tmpdir
summary = {
    "reference_kind": "dmrg",
    "reference_label": "DMRG",
    "total_energy_eh": float(e_total),
    "electronic_energy_eh": float(e_total - ecore),
    "ecore_eh": ecore,
    "bond_dims": bond_dims,
    "noises": noises,
    "dav_thrds": dav_thrds,
    "n_steps": n_steps,
    "init_mode": init_mode,
    "isym": isym,
    "threads": threads,
    "memory_gb": memory_gb,
    "conv_tol": conv_tol,
    "wall_s": float(time.perf_counter() - t0),
    "log_path": str(log_path),
    "fcidump_path": str(fcidump_path),
}
print(json.dumps(summary))
"""
    env = os.environ.copy()
    env.update(
        {
            "CKA_FCIDUMP": str(fcidump_path),
            "CKA_SCRATCH": str(scratch_dir),
            "CKA_LOG": str(log_path),
            "CKA_MEMORY_GB": str(int(memory_gb)),
            "CKA_THREADS": str(int(threads)),
            "CKA_CONV_TOL": str(float(conv_tol)),
            "CKA_BOND_DIMS": json.dumps(bond_dims),
            "CKA_NOISES": json.dumps(noises),
            "CKA_DAV_THRDS": json.dumps(dav_thrds),
            "CKA_N_STEPS": "40",
            "CKA_OCCS_JSON": json.dumps(hf_occs),
            "CKA_ISYM": str(isym),
        }
    )
    run = subprocess.run([str(dmrg_python), "-c", script], text=True, capture_output=True, env=env)
    stdout_path.write_text(run.stdout, encoding="utf-8")
    stderr_path.write_text(run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(
            "DMRG reference subprocess failed with rc="
            f"{run.returncode}; see {log_path}, {stdout_path}, and {stderr_path}"
        )
    summary = _normalize_reference_summary(json.loads(run.stdout.strip()))
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _make_fixed_budget_params(
    config: threshold.StructuralConfig,
    *,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
) -> QPCCIKParams:
    return QPCCIKParams(
        ccik_basis_size=int(config.basis_size),
        ccik_generator_keep=1,
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_max_parents_for_expansion=0,
        ccik_include_current_support=True,
        ccik_selector_keep=1,
        ccik_external_keep=int(INTERNAL_EXTERNAL_KEEP),
        ccik_external_candidate_factor=1,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_compression_policy="fixed_budget",
        ccik_fixed_store_keep=int(fixed_store_keep),
        ccik_fixed_propagate_keep=int(fixed_propagate_keep),
        ccik_store_abs_tol=0.0,
        ccik_propagate_abs_tol=0.0,
        ccik_nkeep=1,
        ccik_tmp_keep_factor=1,
        ccik_preserve_prev_support=False,
        ccik_preserve_prev_support_cap=0,
        ccik_total_det_budget=0,
        ccik_conv_tol=1e-12,
        ccik_ritz_overlap_tol=1e-8,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=False,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=False,
        ccik_max_cycle=int(config.max_cycle),
        ccik_restart_m_cycle=int(config.restart_m_cycle),
        ccik_restart_nroot=int(config.restart_nroot),
        ccik_verbose=False,
    )


def _extract_optional(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)} =\s*([-+0-9.eE]+)", text)
    return None if match is None else float(match.group(1))


def _run_single_qp_job(
    *,
    case: Cr2FixedBudgetCase,
    reference: dict[str, object],
    qp_root: Path,
    base_ezfio: Path,
    config: threshold.StructuralConfig,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    run_dir: Path,
    omp_threads: int,
    force: bool,
) -> dict[str, object]:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    qp = discover_qp_paths_from_root(qp_root)
    params = _make_fixed_budget_params(
        config,
        fixed_store_keep=int(fixed_store_keep),
        fixed_propagate_keep=int(fixed_propagate_keep),
    )
    runtime_parent = Path(tempfile.mkdtemp(prefix=f"{case.name}_{config.tag}_", dir="/tmp"))
    runtime_ezfio = runtime_parent / base_ezfio.name
    shutil.copytree(base_ezfio, runtime_ezfio)
    ensure_ezfio_defaults(runtime_ezfio, qp_root=qp.qp_root)
    set_ccik_params(qp, runtime_ezfio, params)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_m", int(params.ccik_basis_size))
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace", True)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace_topn", 32)

    env = {**os.environ, "QP_ROOT": str(qp.qp_root), **qp_parallel_env(int(omp_threads))}
    quoted_rc = str(qp.quantum_package_rc.resolve())
    quoted_ccik = str(qp.ccik_binary.resolve())
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"source '{quoted_rc}' >/dev/null 2>&1 && qp set_file '{runtime_ezfio.name}' >/dev/null && '{quoted_ccik}' '{runtime_ezfio.name}'",
        ],
        cwd=str(runtime_parent),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_s = float(time.perf_counter() - t0)

    stdout_path = run_dir / "ccik_stdout.log"
    stderr_path = run_dir / "ccik_stderr.log"
    trace_src = runtime_parent / "ccik_debug_trace.log"
    trace_dst = run_dir / "ccik_debug_trace.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if trace_src.exists():
        shutil.copy2(trace_src, trace_dst)

    e_total = _extract_optional(proc.stdout, "CCIK_FINAL_TOTAL_ENERGY")
    e_elec = _extract_optional(proc.stdout, "CCIK_FINAL_ELECTRONIC_ENERGY")
    basis = _extract_optional(proc.stdout, "CCIK_FINAL_BASIS")
    delta = None if e_total is None else float((float(e_total) - float(reference["total_energy_eh"])) * 1000.0)
    failure = None
    if proc.returncode != 0 or e_total is None:
        failure = (
            f"QP CCIK run failed.\nreturncode={proc.returncode}\n"
            f"stdout_path={stdout_path}\nstderr_path={stderr_path}\ntrace_path={trace_dst}"
        )

    qp_summary = {
        "case": case.name,
        "returncode": int(proc.returncode),
        "final_total_energy": None if e_total is None else float(e_total),
        "final_electronic_energy": None if e_elec is None else float(e_elec),
        "final_basis": None if basis is None else int(basis),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "trace_path": str(trace_dst),
        "trace_exists": trace_dst.exists(),
    }
    (run_dir / "qp_summary.json").write_text(json.dumps(qp_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "case": case.name,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "method": config.method,
        "config_tag": config.tag,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "e_total_eh": None if e_total is None else float(e_total),
        "delta_vs_ref_mEh": delta,
        "abs_delta_vs_ref_mEh": None if delta is None else abs(float(delta)),
        "wall_s": wall_s,
        "omp_threads": int(omp_threads),
        "compression_policy": params.ccik_compression_policy,
        "basis_size": int(config.basis_size),
        "max_cycle": int(config.max_cycle),
        "restart_m_cycle": int(config.restart_m_cycle),
        "restart_nroot": int(config.restart_nroot),
        "failure": failure,
        "artifacts": {
            "qp_summary": str(run_dir / "qp_summary.json"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "trace": str(trace_dst),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(runtime_parent, ignore_errors=True)
    return payload


def _calibration_payload() -> dict[str, object]:
    config = CASE.fixed_config
    return {
        "calibration_mode": "fixed_structural_config",
        "dense": {"chosen": None, "runs": []},
        "thick": {
            "chosen": {
                "tag": config.tag,
                "basis_size": int(config.basis_size),
                "max_cycle": int(config.max_cycle),
                "restart_m_cycle": int(config.restart_m_cycle),
                "restart_nroot": int(config.restart_nroot),
            },
            "runs": [],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    qp_root = threshold._require_qp_root(args.qp_root)
    omp_threads = resolve_omp_threads(args.omp_threads)
    qp = discover_qp_paths_from_root(qp_root)
    rebuild_summary = rebuild_ccik_plugin(qp, clean=False, sync_workspace=True)
    (outdir / "qp_plugin_rebuild.json").write_text(
        json.dumps(
            {
                "plugin_dir": rebuild_summary["plugin_dir"],
                "binary_path": rebuild_summary["binary_path"],
                "binary_exists": rebuild_summary["binary_exists"],
                "command": rebuild_summary["command"],
                "sync_summary": rebuild_summary["sync_summary"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    qp_root_validation = validate_fresh_qp_root(qp_root, outdir=outdir / "qp_root_validation")
    (outdir / "qp_root_validation.json").write_text(
        json.dumps(qp_root_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not qp_root_validation["passed"]:
        raise RuntimeError(
            f"Fresh-root validation failed for {qp_root}. See {outdir / 'qp_root_validation.json'} for details."
        )

    setup = _load_or_create_system_setup(case=CASE, qp_root=qp_root, outdir=outdir, force=args.force)
    reference = _load_or_compute_dmrg_reference(
        fcidump_path=Path(setup["fcidump_path"]),
        outdir=outdir,
        force=args.force,
        memory_gb=args.dmrg_memory_gb,
        threads=args.dmrg_threads,
        conv_tol=args.dmrg_conv_tol,
        dmrg_python=args.dmrg_python,
    )
    calibration = _calibration_payload()
    (outdir / "calibration.json").write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dummy_dense = threshold.StructuralConfig(
        method="dense",
        tag="unused_dense",
        basis_size=1,
        max_cycle=1,
        restart_m_cycle=0,
        restart_nroot=2,
    )
    job_specs = n2_fixed._build_case_matrix(
        CASE,
        dummy_dense,
        CASE.fixed_config,
        methods=["thick"],
        store_grid=args.store_grid,
        propagate_grid=args.propagate_grid,
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
    )
    ordered_jobs = n2_fixed._execution_order(
        job_specs,
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
    )

    csv_path = outdir / "fixed_budget_sweep.csv"
    progress_path = outdir / "progress.json"
    rows_by_key = {} if args.force else n2_fixed._load_cached_rows(csv_path)
    n2_fixed._refresh_outputs(
        outdir=outdir,
        case=CASE,
        reference=reference,
        calibration=calibration,
        rows_by_key=rows_by_key,
        total_rows=len(job_specs),
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
        store_grid=args.store_grid,
        propagate_grid=args.propagate_grid,
        methods=["thick"],
        csv_path=csv_path,
        progress_path=progress_path,
    )

    base_ezfio = Path(setup["ezfio_path"])
    for job in ordered_jobs:
        key = (str(job["sweep_kind"]), int(job["budget_value"]), str(job["method"]))
        if key in rows_by_key and not args.force:
            continue
        run_dir = outdir / n2_fixed._run_dir_for(
            sweep_kind=str(job["sweep_kind"]),
            method=str(job["method"]),
            budget=int(job["budget_value"]),
        )
        result = _run_single_qp_job(
            case=CASE,
            reference=reference,
            qp_root=qp_root,
            base_ezfio=base_ezfio,
            config=CASE.fixed_config,
            fixed_store_keep=int(job["fixed_store_keep"]),
            fixed_propagate_keep=int(job["fixed_propagate_keep"]),
            run_dir=run_dir,
            omp_threads=omp_threads,
            force=args.force,
        )
        rows_by_key[key] = {
            "case": CASE.name,
            "reference_kind": reference["reference_kind"],
            "reference_label": reference["reference_label"],
            "sweep_kind": str(job["sweep_kind"]),
            "budget_name": str(job["budget_name"]),
            "budget_value": int(job["budget_value"]),
            "method": str(job["method"]),
            "config_tag": str(CASE.fixed_config.tag),
            "fixed_store_keep": int(job["fixed_store_keep"]),
            "fixed_propagate_keep": int(job["fixed_propagate_keep"]),
            "e_total_eh": result["e_total_eh"],
            "delta_vs_ref_mEh": result["delta_vs_ref_mEh"],
            "abs_delta_vs_ref_mEh": result["abs_delta_vs_ref_mEh"],
            "wall_s": result["wall_s"],
            "basis_size": int(CASE.fixed_config.basis_size),
            "max_cycle": int(CASE.fixed_config.max_cycle),
            "restart_m_cycle": int(CASE.fixed_config.restart_m_cycle),
            "restart_nroot": int(CASE.fixed_config.restart_nroot),
            "failure": result["failure"],
        }
        n2_fixed._refresh_outputs(
            outdir=outdir,
            case=CASE,
            reference=reference,
            calibration=calibration,
            rows_by_key=rows_by_key,
            total_rows=len(job_specs),
            fixed_store_keep=args.fixed_store_keep,
            fixed_propagate_keep=args.fixed_propagate_keep,
            store_grid=args.store_grid,
            propagate_grid=args.propagate_grid,
            methods=["thick"],
            csv_path=csv_path,
            progress_path=progress_path,
        )

    n2_fixed._refresh_outputs(
        outdir=outdir,
        case=CASE,
        reference=reference,
        calibration=calibration,
        rows_by_key=rows_by_key,
        total_rows=len(job_specs),
        fixed_store_keep=args.fixed_store_keep,
        fixed_propagate_keep=args.fixed_propagate_keep,
        store_grid=args.store_grid,
        propagate_grid=args.propagate_grid,
        methods=["thick"],
        csv_path=csv_path,
        progress_path=progress_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
