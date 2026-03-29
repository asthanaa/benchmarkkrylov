from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cr2_qp_fixed_budget_sweep_plots as cr2_fixed  # noqa: E402
import debug_selector_parity as dsp  # noqa: E402
import n2_qp_fixed_budget_sweep_plots as n2_fixed  # noqa: E402
import n2_qp_threshold_sweep_plots as n2_threshold  # noqa: E402
from _shared.n2_utils import make_n2_atom  # noqa: E402
from ccik import CCIKStochasticParams, ccik_ground_energy_stochastic, exact_cas_fci_energy_pyscf  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    CCIKParams as QPCCIKParams,
    detect_physical_cpu_count,
    discover_qp_paths_from_root,
    ensure_ezfio_defaults,
    qp_parallel_env,
    resolve_omp_threads,
    run_qp_setup,
    set_ccik_params,
    validate_rebuildable_qp_root,
    write_ezfio_value,
)


PUB_ROOT = ROOT / "docs" / "assets" / "pub"
DEFAULT_STAMP = datetime.now().strftime("%Y%m%d")
DEFAULT_OUTDIR = PUB_ROOT / f"fixed_budget_transfer_{DEFAULT_STAMP}_live"
KPROP_GRID = (1024, 512, 2048)
KSTORE_GRID = (65_536, 49_152, 32_768, 88_474, 98_304, 131_072)
CSV_FIELDS = [
    "system",
    "title",
    "reference_kind",
    "reference_label",
    "method",
    "engine",
    "phase",
    "budget_name",
    "budget_value",
    "fixed_store_keep",
    "fixed_propagate_keep",
    "config_tag",
    "reference_total_energy_eh",
    "e_total_eh",
    "delta_vs_ref_mEh",
    "abs_delta_vs_ref_mEh",
    "wall_s",
    "source_kind",
    "stochastic_walkers",
    "failure",
]
PLOT_FLOOR_MEH = 1.0e-6
N2_NMO_TOTAL = 18


@dataclass(frozen=True)
class PublicationCase:
    name: str
    title: str
    kind: str
    basis: str
    distance_ang: float
    ncas: int
    nelecas: int
    ncore: int
    nmo_total: int | None
    thick_config: n2_threshold.StructuralConfig
    external_keep: int
    reference_kind: str
    stochastic_m: int
    stochastic_walkers: int
    dmrg_python: Path | None = None
    dmrg_memory_gb: int = 0
    dmrg_threads: int = 0
    seed_kprop_csv: Path | None = None
    seed_ezfio: Path | None = None
    seed_fcidump: Path | None = None


CASES: tuple[PublicationCase, ...] = (
    PublicationCase(
        name="n2_cas12_12_R2p50",
        title="N2 / 6-31G / R = 2.50 A / CAS(12e,12o)",
        kind="n2",
        basis="6-31g",
        distance_ang=2.50,
        ncas=12,
        nelecas=12,
        ncore=1,
        nmo_total=N2_NMO_TOTAL,
        thick_config=n2_threshold.StructuralConfig(
            method="thick",
            tag="restart_30_30_8",
            basis_size=30,
            max_cycle=30,
            restart_m_cycle=30,
            restart_nroot=8,
        ),
        external_keep=n2_threshold.CASES["n2_cas12_12_R2p50"].internal_external_keep,
        reference_kind="exact_fci",
        stochastic_m=30,
        stochastic_walkers=200_000,
        seed_kprop_csv=ROOT
        / "experiments"
        / "threshold_test"
        / "n2_qp_n2_cas12_12_R2p50_fixed_budget_centered_20260326_live"
        / "fixed_budget_sweep.csv",
        seed_ezfio=Path("/private/tmp/n2_cas12_12_R2p50_thick_restart_30_30_8_ucsp0mjr/n2_cas12_12_R2p50.ezfio"),
        seed_fcidump=Path("/private/tmp/n2_cas12_12_R2p50_ats48m55/n2_cas12_12_R2p50.ezfio.FCIDUMP"),
    ),
    PublicationCase(
        name="n2_cas14_14_R2p50",
        title="N2 / 6-31G / R = 2.50 A / CAS(14e,14o)",
        kind="n2",
        basis="6-31g",
        distance_ang=2.50,
        ncas=14,
        nelecas=14,
        ncore=0,
        nmo_total=N2_NMO_TOTAL,
        thick_config=n2_threshold.StructuralConfig(
            method="thick",
            tag="restart_16_16_5",
            basis_size=16,
            max_cycle=16,
            restart_m_cycle=16,
            restart_nroot=5,
        ),
        external_keep=n2_threshold.CASES["n2_cas14_14_R2p50"].internal_external_keep,
        reference_kind="exact_fci",
        stochastic_m=16,
        stochastic_walkers=250_000,
        seed_kprop_csv=ROOT
        / "experiments"
        / "threshold_test"
        / "n2_qp_n2_cas14_14_R2p50_fixed_budget_centered_20260326_thick_prop2048_live"
        / "fixed_budget_sweep.csv",
        seed_ezfio=Path("/private/tmp/n2_cas14_14_R2p50_thick_restart_16_16_5_aj495usi/n2_cas14_14_R2p50.ezfio"),
        seed_fcidump=Path("/private/tmp/n2_cas14_14_R2p50_d93zd7_1/n2_cas14_14_R2p50.ezfio.FCIDUMP"),
    ),
    PublicationCase(
        name="cr2_sto3g_R1p68_cas28_20",
        title="Cr2 / STO-3G / R = 1.68 A / CAS(28e,20o)",
        kind="cr2",
        basis="sto-3g",
        distance_ang=1.68,
        ncas=20,
        nelecas=28,
        ncore=10,
        nmo_total=None,
        thick_config=cr2_fixed.CASE.fixed_config,
        external_keep=1_000_000,
        reference_kind="dmrg",
        stochastic_m=24,
        stochastic_walkers=300_000,
        dmrg_python=ROOT
        / "experiments"
        / "threshold_test"
        / "block2_cr2_sto3g_R1p68_cas28_20"
        / ".venv"
        / "bin"
        / "python",
        dmrg_memory_gb=128,
        dmrg_threads=8,
    ),
)


def _candidate_qp_roots() -> list[Path]:
    return [
        Path("/Users/ayush.asthana/quantum_package/qp2_ccik_pristine_20260325"),
        Path("/Users/ayush.asthana/quantum_package/qp2_ccik_20260325"),
        Path("/private/tmp/qp2_fresh_20260320"),
    ]


def _resolve_qp_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(_candidate_qp_roots())
    env_root = os.environ.get("QP_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if validate_rebuildable_qp_root(resolved)["is_rebuildable"]:
            return resolved
    checked = {str(path): validate_rebuildable_qp_root(path) for path in seen}
    raise RuntimeError(f"No rebuildable QP root available.\n{json.dumps(checked, indent=2, sort_keys=True)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication fixed-budget transfer benchmark for thick QP and stochastic CCIK")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--qp-root", type=Path, default=None)
    p.add_argument("--omp-threads", default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--phase", choices=["all", "kprop", "kstore"], default="all")
    p.add_argument("--no-seed", action="store_true", help="Do not seed N2 thick Kprop rows from finished QP sweeps")
    p.add_argument("--systems", default=None, help="Comma-separated subset of systems to run")
    args = p.parse_args(argv)
    if args.outdir is None:
        requested = {part.strip() for part in str(args.systems or "").split(",") if part.strip()}
        if requested == {"cr2_sto3g_R1p68_cas28_20"}:
            args.outdir = PUB_ROOT / f"cr2_fixed_budget_pub_{DEFAULT_STAMP}_live"
        else:
            args.outdir = DEFAULT_OUTDIR
    return args


def _selected_cases(args: argparse.Namespace) -> list[PublicationCase]:
    if args.systems is None:
        return list(CASES)
    requested = {part.strip() for part in str(args.systems).split(",") if part.strip()}
    unknown = requested - {case.name for case in CASES}
    if unknown:
        raise ValueError(f"Unknown systems requested: {sorted(unknown)}")
    return [case for case in CASES if case.name in requested]


def _normalize_csv_float(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped)


def _load_rows(csv_path: Path) -> dict[tuple[str, str, str, int], dict[str, object]]:
    if not csv_path.exists():
        return {}
    rows: dict[tuple[str, str, str, int], dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = {
                "system": raw["system"],
                "title": raw["title"],
                "reference_kind": raw["reference_kind"],
                "reference_label": raw["reference_label"],
                "method": raw["method"],
                "engine": raw["engine"],
                "phase": raw["phase"],
                "budget_name": raw["budget_name"],
                "budget_value": int(raw["budget_value"]),
                "fixed_store_keep": int(raw["fixed_store_keep"]),
                "fixed_propagate_keep": int(raw["fixed_propagate_keep"]),
                "config_tag": raw["config_tag"],
                "reference_total_energy_eh": _normalize_csv_float(raw["reference_total_energy_eh"]),
                "e_total_eh": _normalize_csv_float(raw["e_total_eh"]),
                "delta_vs_ref_mEh": _normalize_csv_float(raw["delta_vs_ref_mEh"]),
                "abs_delta_vs_ref_mEh": _normalize_csv_float(raw["abs_delta_vs_ref_mEh"]),
                "wall_s": _normalize_csv_float(raw["wall_s"]),
                "source_kind": raw["source_kind"],
                "stochastic_walkers": int(raw["stochastic_walkers"]) if raw["stochastic_walkers"].strip() else None,
                "failure": raw["failure"] or None,
            }
            rows[_row_key(row)] = row
    return rows


def _row_key(row: dict[str, object]) -> tuple[str, str, str, int]:
    return (str(row["system"]), str(row["method"]), str(row["phase"]), int(row["budget_value"]))


def _sorted_rows(rows_by_key: dict[tuple[str, str, str, int], dict[str, object]]) -> list[dict[str, object]]:
    system_order = {case.name: idx for idx, case in enumerate(CASES)}
    method_order = {"ccik_thick_qp": 0, "ccik_stochastic": 1}
    phase_order = {"kprop": 0, "kstore": 1}
    return [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda item: (
                system_order.get(item[0], 99),
                phase_order.get(item[2], 99),
                method_order.get(item[1], 99),
                int(item[3]),
            ),
        )
    ]


def _write_rows_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)


def _create_xyz(path: Path, atom: str, *, comment: str) -> None:
    lines = [line.strip() for line in atom.strip().splitlines() if line.strip()]
    path.write_text(f"{len(lines)}\n{comment}\n" + "\n".join(lines) + "\n", encoding="ascii")


def _n2_setup_paths(outdir: Path) -> dict[str, Path]:
    setup_dir = outdir / "system_setup"
    return {
        "dir": setup_dir,
        "xyz": setup_dir / "n2.xyz",
        "ezfio": setup_dir / "n2.ezfio",
        "fcidump": setup_dir / "qp_active.FCIDUMP",
        "metadata": setup_dir / "system_setup.json",
    }


def _load_or_create_n2_setup(*, case: PublicationCase, qp_root: Path, outdir: Path, force: bool) -> dict[str, object]:
    paths = _n2_setup_paths(outdir)
    if paths["metadata"].exists() and not force:
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if Path(payload["fcidump_path"]).exists() and Path(payload["ezfio_path"]).exists():
            return payload

    if paths["dir"].exists():
        shutil.rmtree(paths["dir"])
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if case.seed_ezfio is not None and case.seed_fcidump is not None and case.seed_ezfio.exists() and case.seed_fcidump.exists():
        shutil.copytree(case.seed_ezfio, paths["ezfio"])
        shutil.copy2(case.seed_fcidump, paths["fcidump"])
        ensure_ezfio_defaults(paths["ezfio"], qp_root=qp_root)
        payload = {
            "case": case.name,
            "title": case.title,
            "basis": case.basis,
            "distance_ang": float(case.distance_ang),
            "ncas": int(case.ncas),
            "nelecas": int(case.nelecas),
            "ncore": int(case.ncore),
            "ezfio_path": str(paths["ezfio"]),
            "fcidump_path": str(paths["fcidump"]),
            "source_kind": "seeded_tmp_ezfio_and_fcidump",
            "seed_ezfio": str(case.seed_ezfio),
            "seed_fcidump": str(case.seed_fcidump),
        }
        paths["metadata"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    _create_xyz(paths["xyz"], make_n2_atom(float(case.distance_ang)), comment=case.title)
    qp = discover_qp_paths_from_root(qp_root)
    run_qp_setup(
        qp,
        basis=case.basis,
        ezfio=paths["ezfio"],
        xyz=paths["xyz"],
        ncore=int(case.ncore),
        ncas=int(case.ncas),
        nmo_total=int(case.nmo_total or 0),
    )
    fcidump_path = dsp._generate_qp_fcidump(qp, paths["ezfio"], paths["dir"], case_spec=None)
    payload = {
        "case": case.name,
        "title": case.title,
        "basis": case.basis,
        "distance_ang": float(case.distance_ang),
        "ncas": int(case.ncas),
        "nelecas": int(case.nelecas),
        "ncore": int(case.ncore),
        "ezfio_path": str(paths["ezfio"]),
        "fcidump_path": str(fcidump_path),
        "qp_fcidump_summary": str(paths["dir"] / "qp_fcidump_summary.json"),
    }
    paths["metadata"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_or_prepare_case(*, case: PublicationCase, qp_root: Path, outdir: Path, force: bool) -> dict[str, object]:
    if case.kind == "n2":
        return _load_or_create_n2_setup(case=case, qp_root=qp_root, outdir=outdir, force=force)
    if case.kind == "cr2":
        return cr2_fixed._load_or_create_system_setup(case=cr2_fixed.CASE, qp_root=qp_root, outdir=outdir, force=force)
    raise ValueError(f"Unsupported case kind {case.kind!r}")


def _load_or_compute_exact_reference(*, fcidump_path: Path, outdir: Path, force: bool) -> dict[str, object]:
    path = outdir / "reference_exact_fci_summary.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    h1, eri8, norb, nelec, ecore = dsp._load_qp_fcidump_hamiltonian(fcidump_path)
    t0 = time.perf_counter()
    total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, norb, nelec))
    payload = {
        "reference_kind": "exact_fci",
        "reference_label": "Exact FCI",
        "total_energy_eh": total,
        "ecore_eh": float(ecore),
        "norb": int(norb),
        "nelec": [int(nelec[0]), int(nelec[1])],
        "wall_s": float(time.perf_counter() - t0),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _normalized_c1_isym(value: object) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 1
    return 1 if parsed == 0 else parsed


def _load_or_compute_cr2_dmrg_reference(
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
        return json.loads(path.read_text(encoding="utf-8"))
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
    hf_occs = n2_threshold._hf_occ_vector(norb, nelec)

    bond_dims = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5000]
    noises = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 0.0, 0.0, 0.0, 0.0]
    dav_thrds = [1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9, 1e-10, 1e-10, 1e-10]
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
            "Cr2 DMRG reference subprocess failed with rc="
            f"{run.returncode}; see {log_path}, {stdout_path}, and {stderr_path}"
        )
    payload = json.loads(run.stdout.strip())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_or_compute_reference(
    *,
    case: PublicationCase,
    setup: dict[str, object],
    outdir: Path,
    force: bool,
) -> dict[str, object]:
    fcidump_path = Path(str(setup["fcidump_path"]))
    if case.reference_kind == "exact_fci":
        return _load_or_compute_exact_reference(fcidump_path=fcidump_path, outdir=outdir, force=force)
    if case.reference_kind == "dmrg":
        return _load_or_compute_cr2_dmrg_reference(
            fcidump_path=fcidump_path,
            outdir=outdir,
            force=force,
            memory_gb=int(case.dmrg_memory_gb),
            threads=int(case.dmrg_threads),
            conv_tol=1e-8,
            dmrg_python=Path(case.dmrg_python) if case.dmrg_python is not None else Path(),
        )
    raise ValueError(f"Unsupported reference kind {case.reference_kind!r}")


def _make_qp_fixed_budget_params(case: PublicationCase, *, fixed_store_keep: int, fixed_propagate_keep: int) -> QPCCIKParams:
    config = case.thick_config
    return QPCCIKParams(
        ccik_basis_size=int(config.basis_size),
        ccik_generator_keep=1,
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_max_parents_for_expansion=0,
        ccik_include_current_support=True,
        ccik_selector_keep=1,
        ccik_external_keep=int(case.external_keep),
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


def _run_qp_thick_job(
    *,
    case: PublicationCase,
    reference: dict[str, object],
    qp_root: Path,
    base_ezfio: Path,
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
    params = _make_qp_fixed_budget_params(
        case,
        fixed_store_keep=int(fixed_store_keep),
        fixed_propagate_keep=int(fixed_propagate_keep),
    )
    runtime_parent = Path(tempfile.mkdtemp(prefix=f"{case.name}_{case.thick_config.tag}_", dir="/tmp"))
    runtime_ezfio = runtime_parent / base_ezfio.name
    shutil.copytree(base_ezfio, runtime_ezfio)
    ensure_ezfio_defaults(runtime_ezfio, qp_root=qp.qp_root)
    set_ccik_params(qp, runtime_ezfio, params)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_m", int(params.ccik_basis_size))

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
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    e_total = _extract_optional(proc.stdout, "CCIK_FINAL_TOTAL_ENERGY")
    delta = None if e_total is None else float((float(e_total) - float(reference["total_energy_eh"])) * 1000.0)
    failure = None
    if proc.returncode != 0 or e_total is None:
        failure = f"QP thick run failed (rc={proc.returncode}); see {stdout_path} and {stderr_path}"

    payload = {
        "system": case.name,
        "title": case.title,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "method": "ccik_thick_qp",
        "engine": "quantum_package",
        "config_tag": case.thick_config.tag,
        "reference_total_energy_eh": float(reference["total_energy_eh"]),
        "e_total_eh": None if e_total is None else float(e_total),
        "delta_vs_ref_mEh": delta,
        "abs_delta_vs_ref_mEh": None if delta is None else abs(float(delta)),
        "wall_s": wall_s,
        "source_kind": "fresh_qp",
        "stochastic_walkers": None,
        "failure": failure,
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(runtime_parent, ignore_errors=True)
    return payload


def _patch_n2_donor_case(case: PublicationCase) -> n2_threshold.SweepCaseSpec:
    if case.seed_ezfio is None or case.seed_fcidump is None:
        raise RuntimeError(f"N2 publication case {case.name} is missing preserved donor shell paths")
    if not case.seed_ezfio.exists() or not case.seed_fcidump.exists():
        raise FileNotFoundError(
            f"Missing preserved donor shell for {case.name}: "
            f"ezfio={case.seed_ezfio}, fcidump={case.seed_fcidump}"
        )
    donor = dsp.DONOR_N2_CASES[case.name]
    dsp.DONOR_N2_CASES[case.name] = replace(
        donor,
        donor_ezfio=case.seed_ezfio,
        cached_qp_fcidump=case.seed_fcidump,
    )
    return n2_threshold.CASES[case.name]


def _run_n2_qp_thick_job(
    *,
    case: PublicationCase,
    reference: dict[str, object],
    qp_root: Path,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    run_dir: Path,
    omp_threads: int,
    force: bool,
) -> dict[str, object]:
    case_spec = _patch_n2_donor_case(case)
    result = n2_fixed._run_single_qp_job(
        case=case_spec,
        reference=reference,
        qp_root=qp_root,
        config=case.thick_config,
        fixed_store_keep=int(fixed_store_keep),
        fixed_propagate_keep=int(fixed_propagate_keep),
        run_dir=run_dir,
        omp_threads=int(omp_threads),
        timeout_s=case_spec.exact_timeout_s,
        force=force,
    )
    return {
        "system": case.name,
        "title": case.title,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "method": "ccik_thick_qp",
        "engine": "quantum_package",
        "config_tag": case.thick_config.tag,
        "reference_total_energy_eh": float(reference["total_energy_eh"]),
        "e_total_eh": result["e_total_eh"],
        "delta_vs_ref_mEh": result["delta_vs_ref_mEh"],
        "abs_delta_vs_ref_mEh": result["abs_delta_vs_ref_mEh"],
        "wall_s": result["wall_s"],
        "source_kind": "fresh_qp_donor_seeded",
        "stochastic_walkers": None,
        "failure": result["failure"],
    }


def _run_stochastic_job(
    *,
    case: PublicationCase,
    reference: dict[str, object],
    fcidump_path: Path,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    run_dir: Path,
    force: bool,
) -> dict[str, object]:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    h1, eri8, norb, nelec, ecore = dsp._load_qp_fcidump_hamiltonian(fcidump_path)
    params = CCIKStochasticParams(
        m=int(case.stochastic_m),
        compression_policy="fixed_budget",
        fixed_store_keep=int(fixed_store_keep),
        fixed_propagate_keep=int(fixed_propagate_keep),
        selector_mode="abs_residual",
        candidate_mode="stochastic",
        selector_depth=1,
        generator_keep=1,
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=0,
        include_current_support=True,
        tmp_keep_factor=1,
        preserve_prev_support=False,
        preserve_prev_support_cap=0,
        n_walkers=int(case.stochastic_walkers),
        seed=0,
        verbose=False,
        orth_tol=1e-12,
    )
    stats: dict[str, object] = {}
    t0 = time.perf_counter()
    e_cas = ccik_ground_energy_stochastic(h1, eri8, norb, nelec, params=params, stats=stats)
    wall_s = float(time.perf_counter() - t0)
    e_total = float(ecore + e_cas)
    delta = float((e_total - float(reference["total_energy_eh"])) * 1000.0)
    payload = {
        "system": case.name,
        "title": case.title,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "method": "ccik_stochastic",
        "engine": "python",
        "config_tag": f"m{case.stochastic_m}_walkers{case.stochastic_walkers}",
        "reference_total_energy_eh": float(reference["total_energy_eh"]),
        "e_total_eh": e_total,
        "delta_vs_ref_mEh": delta,
        "abs_delta_vs_ref_mEh": abs(delta),
        "wall_s": wall_s,
        "source_kind": "fresh_python",
        "stochastic_walkers": int(case.stochastic_walkers),
        "failure": None,
        "stats": stats,
        "params": params.__dict__,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _best_kprop(rows_by_key: dict[tuple[str, str, str, int], dict[str, object]], case: PublicationCase, method: str) -> int | None:
    candidates = [
        row
        for row in rows_by_key.values()
        if row["system"] == case.name
        and row["method"] == method
        and row["phase"] == "kprop"
        and not row.get("failure")
        and row.get("abs_delta_vs_ref_mEh") is not None
    ]
    completed_budgets = {int(row["budget_value"]) for row in candidates}
    if any(int(budget) not in completed_budgets for budget in KPROP_GRID):
        return None
    if not candidates:
        return None
    best = min(candidates, key=lambda row: (float(row["abs_delta_vs_ref_mEh"]), int(row["budget_value"])))
    return int(best["budget_value"])


def _summary_rows(rows_by_key: dict[tuple[str, str, str, int], dict[str, object]]) -> list[dict[str, object]]:
    rows = _sorted_rows(rows_by_key)
    best_by_combo: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        if row.get("failure") or row.get("abs_delta_vs_ref_mEh") is None:
            continue
        key = (str(row["system"]), str(row["method"]), str(row["phase"]))
        current = best_by_combo.get(key)
        if current is None or (
            float(row["abs_delta_vs_ref_mEh"]),
            int(row["budget_value"]),
        ) < (
            float(current["abs_delta_vs_ref_mEh"]),
            int(current["budget_value"]),
        ):
            best_by_combo[key] = row
    return [best_by_combo[key] for key in sorted(best_by_combo)]


def _write_summary_markdown(outdir: Path, rows_by_key: dict[tuple[str, str, str, int], dict[str, object]], cases: list[PublicationCase]) -> None:
    title = "# Cr2 Fixed-Budget Publication Benchmark" if len(cases) == 1 and cases[0].kind == "cr2" else "# Fixed-Budget Transfer Benchmark"
    lines = [
        title,
        "",
        "Settled deterministic settings:",
        "- engine: Quantum Package",
        "- selector: abs_residual",
        "- deterministic solver: thick CCIK",
        "- compression: fixed_budget",
        "- anchor store keep: 65536",
        "- Kprop scan: 512, 1024, 2048",
        "- Kstore scan: 32768, 49152, 65536, 88474, 98304, 131072",
        "",
        "Method note:",
        "- `ccik_thick_qp` is the settled production path.",
        "- `ccik_stochastic` is the maintained Python stochastic implementation run on the same FCIDUMP Hamiltonian and reference.",
        "",
        "## Best Completed Rows",
        "",
    ]
    if len(cases) == 1 and cases[0].kind == "cr2":
        case = cases[0]
        thick_best_kprop = _best_kprop(rows_by_key, case, "ccik_thick_qp")
        if thick_best_kprop is not None:
            lines.extend(
                [
                    "## Cr2 Publication Summary",
                    "",
                    f"- system: {case.title}",
                    f"- best thick-QP Kprop: {thick_best_kprop}",
                ]
            )
            thick_kprop_row = rows_by_key.get((case.name, "ccik_thick_qp", "kprop", int(thick_best_kprop)))
            if thick_kprop_row is not None and thick_kprop_row.get("abs_delta_vs_ref_mEh") is not None:
                lines.append(f"- thick-QP error at best Kprop: {float(thick_kprop_row['abs_delta_vs_ref_mEh']):.6f} mEh")
            thick_kstore_rows = [
                row
                for row in rows_by_key.values()
                if row["system"] == case.name
                and row["method"] == "ccik_thick_qp"
                and row["phase"] == "kstore"
                and not row.get("failure")
                and row.get("abs_delta_vs_ref_mEh") is not None
            ]
            if thick_kstore_rows:
                best_store = min(thick_kstore_rows, key=lambda row: (float(row["abs_delta_vs_ref_mEh"]), int(row["budget_value"])))
                lines.extend(
                    [
                        f"- best thick-QP Kstore: {int(best_store['budget_value'])}",
                        f"- best thick-QP error: {float(best_store['abs_delta_vs_ref_mEh']):.6f} mEh",
                    ]
                )
                stochastic_match = rows_by_key.get((case.name, "ccik_stochastic", "kstore", int(best_store["budget_value"])))
                if stochastic_match is not None and stochastic_match.get("abs_delta_vs_ref_mEh") is not None and not stochastic_match.get("failure"):
                    lines.append(
                        f"- stochastic error at same Kstore/Kprop: {float(stochastic_match['abs_delta_vs_ref_mEh']):.6f} mEh"
                    )
            lines.append("")
    for row in _summary_rows(rows_by_key):
        lines.extend(
            [
                f"### {row['system']} / {row['method']} / {row['phase']}",
                f"- budget value: {int(row['budget_value'])}",
                f"- fixed_store_keep: {int(row['fixed_store_keep'])}",
                f"- fixed_propagate_keep: {int(row['fixed_propagate_keep'])}",
                f"- abs error: {float(row['abs_delta_vs_ref_mEh']):.6f} mEh",
                f"- wall: {float(row['wall_s']):.2f} s",
                "",
            ]
        )
    lines.extend(["## Progress", ""])
    for case in cases:
        completed = sum(1 for row in rows_by_key.values() if row["system"] == case.name and not row.get("failure"))
        total = sum(1 for row in rows_by_key.values() if row["system"] == case.name)
        lines.append(f"- {case.title}: {completed} completed rows written ({total} cached or fresh rows recorded)")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _publication_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_scan(
    *,
    outdir: Path,
    rows_by_key: dict[tuple[str, str, str, int], dict[str, object]],
    cases: list[PublicationCase],
    phase: str,
) -> None:
    _publication_rcparams()
    fig, axes = plt.subplots(len(cases), 1, figsize=(7.0, 2.2 * len(cases)), sharex=False, constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    method_styles = {
        "ccik_thick_qp": {"label": "CCIK-thick (QP)", "color": "black", "marker": "o", "linestyle": "-"},
        "ccik_stochastic": {"label": "CCIK-stochastic", "color": "#d95f02", "marker": "s", "linestyle": "--"},
    }
    for ax, case in zip(axes, cases):
        for method, style in method_styles.items():
            method_rows = [
                row
                for row in rows_by_key.values()
                if row["system"] == case.name
                and row["method"] == method
                and row["phase"] == phase
                and row.get("failure") is None
                and row.get("abs_delta_vs_ref_mEh") is not None
            ]
            method_rows.sort(key=lambda row: int(row["budget_value"]))
            if not method_rows:
                continue
            xvals = [int(row["budget_value"]) for row in method_rows]
            yvals = [max(PLOT_FLOOR_MEH, float(row["abs_delta_vs_ref_mEh"])) for row in method_rows]
            ax.plot(xvals, yvals, label=style["label"], color=style["color"], marker=style["marker"], linestyle=style["linestyle"])
        ax.set_yscale("log")
        ax.set_ylabel(r"$|E-E_{\mathrm{ref}}|$ (mEh)")
        ax.set_title(case.title)
        if phase == "kprop":
            ax.set_xlabel(r"$K_{\mathrm{prop}}$")
        else:
            ax.set_xlabel(r"$K_{\mathrm{store}}$")
        ax.legend(loc="best")
    stem = "kprop_scan" if phase == "kprop" else "kstore_scan"
    fig.savefig(outdir / f"{stem}.png")
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)


def _refresh_outputs(
    *,
    outdir: Path,
    rows_by_key: dict[tuple[str, str, str, int], dict[str, object]],
    cases: list[PublicationCase],
    total_jobs: int,
) -> None:
    rows = _sorted_rows(rows_by_key)
    _write_rows_csv(outdir / "results.csv", rows)
    _plot_scan(outdir=outdir, rows_by_key=rows_by_key, cases=cases, phase="kprop")
    _plot_scan(outdir=outdir, rows_by_key=rows_by_key, cases=cases, phase="kstore")
    progress = {
        "completed_rows": sum(1 for row in rows if row.get("failure") is None and row.get("e_total_eh") is not None),
        "failed_rows": sum(1 for row in rows if row.get("failure")),
        "cached_or_written_rows": len(rows),
        "planned_jobs": int(total_jobs),
    }
    (outdir / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_markdown(outdir, rows_by_key, cases)


def _seed_n2_kprop_rows(
    *,
    case: PublicationCase,
    reference: dict[str, object],
    rows_by_key: dict[tuple[str, str, str, int], dict[str, object]],
) -> None:
    if case.seed_kprop_csv is None or not case.seed_kprop_csv.exists():
        return
    with case.seed_kprop_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if raw["method"] != "thick":
                continue
            if raw["sweep_kind"] != "fixed_propagate_keep":
                continue
            if int(raw["fixed_store_keep"]) != 65_536:
                continue
            budget = int(raw["budget_value"])
            if budget not in KPROP_GRID:
                continue
            key = (case.name, "ccik_thick_qp", "kprop", budget)
            existing = rows_by_key.get(key)
            if existing is not None and existing.get("failure") is None and existing.get("e_total_eh") is not None:
                continue
            e_total = _normalize_csv_float(raw["e_total_eh"])
            delta = None if e_total is None else float((float(e_total) - float(reference["total_energy_eh"])) * 1000.0)
            rows_by_key[key] = {
                "system": case.name,
                "title": case.title,
                "reference_kind": reference["reference_kind"],
                "reference_label": reference["reference_label"],
                "method": "ccik_thick_qp",
                "engine": "quantum_package",
                "phase": "kprop",
                "budget_name": "fixed_propagate_keep",
                "budget_value": budget,
                "fixed_store_keep": 65_536,
                "fixed_propagate_keep": budget,
                "config_tag": case.thick_config.tag,
                "reference_total_energy_eh": float(reference["total_energy_eh"]),
                "e_total_eh": e_total,
                "delta_vs_ref_mEh": delta,
                "abs_delta_vs_ref_mEh": None if delta is None else abs(delta),
                "wall_s": _normalize_csv_float(raw["wall_s"]),
                "source_kind": "seeded_previous_qp",
                "stochastic_walkers": None,
                "failure": raw["failure"] or None,
            }


def _kprop_jobs(cases: list[PublicationCase]) -> list[tuple[PublicationCase, str, int]]:
    jobs: list[tuple[PublicationCase, str, int]] = []
    for case in cases:
        for budget in KPROP_GRID:
            jobs.append((case, "ccik_thick_qp", int(budget)))
            jobs.append((case, "ccik_stochastic", int(budget)))
    return jobs


def _kstore_jobs(cases: list[PublicationCase], rows_by_key: dict[tuple[str, str, str, int], dict[str, object]]) -> list[tuple[PublicationCase, str, int, int]]:
    jobs: list[tuple[PublicationCase, str, int, int]] = []
    for case in cases:
        for method in ("ccik_thick_qp", "ccik_stochastic"):
            best = _best_kprop(rows_by_key, case, method)
            if best is None:
                continue
            for budget in KSTORE_GRID:
                jobs.append((case, method, int(best), int(budget)))
    return jobs


def _seed_kstore_anchor_rows(
    *,
    cases: list[PublicationCase],
    rows_by_key: dict[tuple[str, str, str, int], dict[str, object]],
) -> None:
    for case in cases:
        for method in ("ccik_thick_qp", "ccik_stochastic"):
            best = _best_kprop(rows_by_key, case, method)
            if best is None:
                continue
            kprop_key = (case.name, method, "kprop", int(best))
            kprop_row = rows_by_key.get(kprop_key)
            if kprop_row is None or kprop_row.get("failure") or kprop_row.get("e_total_eh") is None:
                continue
            kstore_key = (case.name, method, "kstore", 65_536)
            existing = rows_by_key.get(kstore_key)
            if existing is not None and existing.get("failure") is None and existing.get("e_total_eh") is not None:
                continue
            rows_by_key[kstore_key] = {
                **kprop_row,
                "phase": "kstore",
                "budget_name": "fixed_store_keep",
                "budget_value": 65_536,
                "fixed_store_keep": 65_536,
                "fixed_propagate_keep": int(best),
                "source_kind": "seeded_from_kprop_anchor",
            }


def _run_phase_kprop(
    *,
    cases: list[PublicationCase],
    qp_root: Path,
    omp_threads: int,
    ensure_case_ready: Any,
    rows_by_key: dict[tuple[str, str, str, int], dict[str, object]],
    outdir: Path,
    force: bool,
    seed_n2: bool,
    total_jobs: int,
) -> None:
    if seed_n2:
        for case in cases:
            if case.kind == "n2":
                _, reference = ensure_case_ready(case)
                _seed_n2_kprop_rows(case=case, reference=reference, rows_by_key=rows_by_key)
        _refresh_outputs(outdir=outdir, rows_by_key=rows_by_key, cases=cases, total_jobs=total_jobs)
    for case, method, budget in _kprop_jobs(cases):
        key = (case.name, method, "kprop", int(budget))
        if key in rows_by_key and not force:
            continue
        setup, reference = ensure_case_ready(case)
        run_dir = outdir / "runs" / case.name / method / "kprop" / str(int(budget))
        if method == "ccik_thick_qp":
            if case.kind == "n2":
                payload = _run_n2_qp_thick_job(
                    case=case,
                    reference=reference,
                    qp_root=qp_root,
                    fixed_store_keep=65_536,
                    fixed_propagate_keep=int(budget),
                    run_dir=run_dir,
                    omp_threads=omp_threads,
                    force=force,
                )
            else:
                payload = _run_qp_thick_job(
                    case=case,
                    reference=reference,
                    qp_root=qp_root,
                    base_ezfio=Path(str(setup["ezfio_path"])),
                    fixed_store_keep=65_536,
                    fixed_propagate_keep=int(budget),
                    run_dir=run_dir,
                    omp_threads=omp_threads,
                    force=force,
                )
            row = {
                **payload,
                "phase": "kprop",
                "budget_name": "fixed_propagate_keep",
                "budget_value": int(budget),
                "fixed_store_keep": 65_536,
                "fixed_propagate_keep": int(budget),
            }
        else:
            payload = _run_stochastic_job(
                case=case,
                reference=reference,
                fcidump_path=Path(str(setup["fcidump_path"])),
                fixed_store_keep=65_536,
                fixed_propagate_keep=int(budget),
                run_dir=run_dir,
                force=force,
            )
            row = {
                **payload,
                "phase": "kprop",
                "budget_name": "fixed_propagate_keep",
                "budget_value": int(budget),
                "fixed_store_keep": 65_536,
                "fixed_propagate_keep": int(budget),
            }
        rows_by_key[key] = row
        _refresh_outputs(outdir=outdir, rows_by_key=rows_by_key, cases=cases, total_jobs=total_jobs)


def _run_phase_kstore(
    *,
    cases: list[PublicationCase],
    qp_root: Path,
    omp_threads: int,
    ensure_case_ready: Any,
    rows_by_key: dict[tuple[str, str, str, int], dict[str, object]],
    outdir: Path,
    force: bool,
    total_jobs: int,
) -> None:
    _seed_kstore_anchor_rows(cases=cases, rows_by_key=rows_by_key)
    _refresh_outputs(outdir=outdir, rows_by_key=rows_by_key, cases=cases, total_jobs=total_jobs)
    for case, method, best_kprop, budget in _kstore_jobs(cases, rows_by_key):
        key = (case.name, method, "kstore", int(budget))
        if key in rows_by_key and not force:
            continue
        setup, reference = ensure_case_ready(case)
        run_dir = outdir / "runs" / case.name / method / "kstore" / str(int(budget))
        if method == "ccik_thick_qp":
            if case.kind == "n2":
                payload = _run_n2_qp_thick_job(
                    case=case,
                    reference=reference,
                    qp_root=qp_root,
                    fixed_store_keep=int(budget),
                    fixed_propagate_keep=int(best_kprop),
                    run_dir=run_dir,
                    omp_threads=omp_threads,
                    force=force,
                )
            else:
                payload = _run_qp_thick_job(
                    case=case,
                    reference=reference,
                    qp_root=qp_root,
                    base_ezfio=Path(str(setup["ezfio_path"])),
                    fixed_store_keep=int(budget),
                    fixed_propagate_keep=int(best_kprop),
                    run_dir=run_dir,
                    omp_threads=omp_threads,
                    force=force,
                )
            row = {
                **payload,
                "phase": "kstore",
                "budget_name": "fixed_store_keep",
                "budget_value": int(budget),
                "fixed_store_keep": int(budget),
                "fixed_propagate_keep": int(best_kprop),
            }
        else:
            payload = _run_stochastic_job(
                case=case,
                reference=reference,
                fcidump_path=Path(str(setup["fcidump_path"])),
                fixed_store_keep=int(budget),
                fixed_propagate_keep=int(best_kprop),
                run_dir=run_dir,
                force=force,
            )
            row = {
                **payload,
                "phase": "kstore",
                "budget_name": "fixed_store_keep",
                "budget_value": int(budget),
                "fixed_store_keep": int(budget),
                "fixed_propagate_keep": int(best_kprop),
            }
        rows_by_key[key] = row
        _refresh_outputs(outdir=outdir, rows_by_key=rows_by_key, cases=cases, total_jobs=total_jobs)


def _write_run_metadata(outdir: Path, qp_root: Path, omp_threads: int, cases: list[PublicationCase]) -> None:
    payload = {
        "qp_root": str(qp_root),
        "omp_threads": int(omp_threads),
        "kprop_grid": list(KPROP_GRID),
        "kstore_grid": list(KSTORE_GRID),
        "cases": [
            {
                "name": case.name,
                "title": case.title,
                "basis": case.basis,
                "distance_ang": float(case.distance_ang),
                "ncas": int(case.ncas),
                "nelecas": int(case.nelecas),
                "ncore": int(case.ncore),
                "thick_config": {
                    "tag": case.thick_config.tag,
                    "basis_size": int(case.thick_config.basis_size),
                    "max_cycle": int(case.thick_config.max_cycle),
                    "restart_m_cycle": int(case.thick_config.restart_m_cycle),
                    "restart_nroot": int(case.thick_config.restart_nroot),
                },
                "stochastic_m": int(case.stochastic_m),
                "stochastic_walkers": int(case.stochastic_walkers),
                "reference_kind": case.reference_kind,
            }
            for case in cases
        ],
    }
    (outdir / "run_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = _selected_cases(args)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    qp_root = _resolve_qp_root(args.qp_root)
    omp_threads = resolve_omp_threads(args.omp_threads)
    qp = discover_qp_paths_from_root(qp_root)
    if not qp.ccik_binary.exists():
        raise FileNotFoundError(f"CCIK binary not found in QP root: {qp.ccik_binary}")
    (outdir / "qp_runtime.json").write_text(
        json.dumps(
            {
                "qp_root": str(qp_root),
                "quantum_package_rc": str(qp.quantum_package_rc),
                "ccik_binary": str(qp.ccik_binary),
                "ccik_binary_exists": qp.ccik_binary.exists(),
                "plugin_dir": str(qp.ccik_binary.parent),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_run_metadata(outdir, qp_root, omp_threads, cases)

    rows_by_key = {} if args.force else _load_rows(outdir / "results.csv")
    case_setups: dict[str, dict[str, object]] = {}
    references: dict[str, dict[str, object]] = {}

    def ensure_case_ready(case: PublicationCase) -> tuple[dict[str, object], dict[str, object]]:
        setup = case_setups.get(case.name)
        reference = references.get(case.name)
        if setup is None or reference is None:
            case_dir = outdir / "systems" / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            setup = _load_or_prepare_case(case=case, qp_root=qp_root, outdir=case_dir, force=args.force)
            reference = _load_or_compute_reference(case=case, setup=setup, outdir=case_dir, force=args.force)
            case_setups[case.name] = setup
            references[case.name] = reference
        return setup, reference

    total_jobs = len(_kprop_jobs(cases)) + 2 * len(KSTORE_GRID) * len(cases)
    _refresh_outputs(outdir=outdir, rows_by_key=rows_by_key, cases=cases, total_jobs=total_jobs)

    if args.phase in {"all", "kprop"}:
        _run_phase_kprop(
            cases=cases,
            qp_root=qp_root,
            omp_threads=omp_threads,
            ensure_case_ready=ensure_case_ready,
            rows_by_key=rows_by_key,
            outdir=outdir,
            force=args.force,
            seed_n2=not args.no_seed,
            total_jobs=total_jobs,
        )
    if args.phase in {"all", "kstore"}:
        _run_phase_kstore(
            cases=cases,
            qp_root=qp_root,
            omp_threads=omp_threads,
            ensure_case_ready=ensure_case_ready,
            rows_by_key=rows_by_key,
            outdir=outdir,
            force=args.force,
            total_jobs=total_jobs,
        )
    _refresh_outputs(outdir=outdir, rows_by_key=rows_by_key, cases=cases, total_jobs=total_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
