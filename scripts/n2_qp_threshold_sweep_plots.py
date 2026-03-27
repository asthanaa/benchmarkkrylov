from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccik import exact_cas_fci_energy_pyscf  # noqa: E402
import debug_selector_parity as dsp  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    DEFAULT_QP_REBUILD_MANIFEST,
    CCIKParams as QPCCIKParams,
    detect_physical_cpu_count,
    discover_qp_paths_from_root,
    ensure_ezfio_defaults,
    infer_qp_root_from_rebuild_manifest,
    qp_parallel_env,
    rebuild_ccik_plugin,
    resolve_rebuildable_qp_root,
    resolve_omp_threads,
    set_ccik_params,
    validate_fresh_qp_root,
    validate_rebuildable_qp_root,
    write_ezfio_value,
)


THRESHOLD_TEST_ROOT = ROOT / "experiments" / "threshold_test"
THRESHOLD_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
DEFAULT_DMRG_PYTHON = ROOT / ".venv" / "bin" / "python"
EXACT_FCI_TIMEOUT_S = 7200
EXACT_DELTA_TOL_MEH = 0.1
PLOT_FLOOR_MEH = 1.0e-9
DEFAULT_ROW_TIMEOUT_S: int | None = None
CALIBRATION_KEEP = 65_536
CALIBRATION_GENERATOR_KEEP = 512
CALIBRATION_TMP_KEEP_FACTOR = 2
CALIBRATION_PRESERVE_CAP = 8192
FAILURE_INFERENCE_MARKERS = (
    "threshold compression policy exceeded internal candidate limit",
    "QP CCIK run timed out",
)


@dataclass(frozen=True)
class StructuralConfig:
    method: str
    tag: str
    basis_size: int
    max_cycle: int
    restart_m_cycle: int
    restart_nroot: int


@dataclass(frozen=True)
class SweepCaseSpec:
    name: str
    donor_case_name: str
    title: str
    fcidump_path: Path
    dense_basis_sizes: tuple[int, ...]
    thick_ladder: tuple[tuple[int, int, int], ...]
    internal_external_keep: int
    exact_timeout_s: int = EXACT_FCI_TIMEOUT_S
    cached_dmrg_reference_path: Path | None = None


CASES: dict[str, SweepCaseSpec] = {
    "n2_cas12_12_R2p50": SweepCaseSpec(
        name="n2_cas12_12_R2p50",
        donor_case_name="n2_cas12_12_R2p50",
        title="N2 / 6-31G / R = 2.50 A / CAS(12e,12o)",
        fcidump_path=THRESHOLD_TEST_ROOT
        / "donor_n2_parity_sweep_20260320_rerun"
        / "n2_cas12_12_R2p50"
        / "connected_sd_fullq"
        / "qp_active.FCIDUMP",
        dense_basis_sizes=(20, 30, 40),
        thick_ladder=((12, 12, 4), (20, 20, 6), (30, 30, 8)),
        internal_external_keep=200_000,
        cached_dmrg_reference_path=THRESHOLD_TEST_ROOT / "n2_cas12_method_compare_20260324" / "dmrg" / "dmrg_summary.json",
    ),
    "n2_cas14_14_R2p50": SweepCaseSpec(
        name="n2_cas14_14_R2p50",
        donor_case_name="n2_cas14_14_R2p50",
        title="N2 / 6-31G / R = 2.50 A / CAS(14e,14o)",
        fcidump_path=THRESHOLD_TEST_ROOT
        / "donor_n2_parity_sweep_20260320_rerun"
        / "n2_cas14_14_R2p50"
        / "connected_sd_fullq"
        / "qp_active.FCIDUMP",
        dense_basis_sizes=(12, 18, 24),
        thick_ladder=((8, 8, 3), (12, 12, 4), (16, 16, 5)),
        internal_external_keep=500_000,
    ),
}


CSV_FIELDS = [
    "case",
    "reference_kind",
    "reference_label",
    "sweep_kind",
    "threshold_name",
    "threshold_value",
    "method",
    "config_tag",
    "store_abs_tol",
    "propagate_abs_tol",
    "e_total_eh",
    "delta_vs_ref_mEh",
    "abs_delta_vs_ref_mEh",
    "wall_s",
    "basis_size",
    "max_cycle",
    "restart_m_cycle",
    "restart_nroot",
    "failure",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="QP threshold sweeps for large N2 donor-backed active spaces")
    p.add_argument("--case", choices=sorted(CASES), required=True)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--qp-root", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument(
        "--omp-threads",
        default="auto",
        help=f'OpenMP thread count for QP runs ("auto" uses physical cores, currently {detect_physical_cpu_count()})',
    )
    p.add_argument("--reference-kind", choices=["dmrg", "auto"], default="dmrg")
    p.add_argument("--row-timeout-s", type=int, default=DEFAULT_ROW_TIMEOUT_S, help="Optional per-row timeout in seconds; default disables the row timeout")
    p.add_argument("--dmrg-memory-gb", type=int, default=64)
    p.add_argument("--dmrg-threads", type=int, default=detect_physical_cpu_count())
    p.add_argument("--dmrg-conv-tol", type=float, default=1e-8)
    p.add_argument("--dmrg-python", type=Path, default=DEFAULT_DMRG_PYTHON)
    args = p.parse_args(argv)
    if args.outdir is None:
        args.outdir = THRESHOLD_TEST_ROOT / f"n2_qp_{args.case}_threshold_sweep_{stamp}"
    if args.row_timeout_s is not None and int(args.row_timeout_s) <= 0:
        args.row_timeout_s = None
    return args


def _row_key(row: dict[str, object]) -> tuple[str, float, str]:
    return (str(row["sweep_kind"]), float(row["threshold_value"]), str(row["method"]))


def _parse_optional_float(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped)


def _parse_optional_int(text: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return None
    return int(float(stripped))


def _load_cached_rows(csv_path: Path) -> dict[tuple[str, float, str], dict[str, object]]:
    if not csv_path.exists():
        return {}
    rows: dict[tuple[str, float, str], dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = {
                "case": raw["case"],
                "reference_kind": raw["reference_kind"],
                "reference_label": raw["reference_label"],
                "sweep_kind": raw["sweep_kind"],
                "threshold_name": raw["threshold_name"],
                "threshold_value": float(raw["threshold_value"]),
                "method": raw["method"],
                "config_tag": raw["config_tag"],
                "store_abs_tol": float(raw["store_abs_tol"]),
                "propagate_abs_tol": float(raw["propagate_abs_tol"]),
                "e_total_eh": _parse_optional_float(raw["e_total_eh"]),
                "delta_vs_ref_mEh": _parse_optional_float(raw["delta_vs_ref_mEh"]),
                "abs_delta_vs_ref_mEh": _parse_optional_float(raw["abs_delta_vs_ref_mEh"]),
                "wall_s": _parse_optional_float(raw["wall_s"]),
                "basis_size": _parse_optional_int(raw["basis_size"]),
                "max_cycle": _parse_optional_int(raw["max_cycle"]),
                "restart_m_cycle": _parse_optional_int(raw["restart_m_cycle"]),
                "restart_nroot": _parse_optional_int(raw["restart_nroot"]),
                "failure": raw["failure"] or None,
            }
            rows[_row_key(row)] = row
    return rows


def _sorted_rows(rows: dict[tuple[str, float, str], dict[str, object]]) -> list[dict[str, object]]:
    return [
        rows[key]
        for key in sorted(
            rows,
            key=lambda item: (
                0 if item[0] == "store_abs_tol" else 1,
                float(item[1]),
                0 if item[2] == "dense" else 1,
            ),
        )
    ]


def _write_rows_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _threshold_tag(value: float) -> str:
    return f"{float(value):.0e}"


def _dense_ladder(case: SweepCaseSpec) -> list[StructuralConfig]:
    return [
        StructuralConfig(
            method="dense",
            tag=f"basis_{basis_size}",
            basis_size=int(basis_size),
            max_cycle=1,
            restart_m_cycle=0,
            restart_nroot=2,
        )
        for basis_size in case.dense_basis_sizes
    ]


def _thick_ladder(case: SweepCaseSpec) -> list[StructuralConfig]:
    return [
        StructuralConfig(
            method="thick",
            tag=f"restart_{m_cycle}_{max_cycle}_{nroot}",
            basis_size=max(2, int(m_cycle)),
            max_cycle=int(max_cycle),
            restart_m_cycle=int(m_cycle),
            restart_nroot=int(nroot),
        )
        for m_cycle, max_cycle, nroot in case.thick_ladder
    ]


def _build_case_matrix(case: SweepCaseSpec, dense_config: StructuralConfig, thick_config: StructuralConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config_map = {"dense": dense_config, "thick": thick_config}
    for sweep_kind, held_store, held_prop in (
        ("store_abs_tol", None, 1.0e-8),
        ("propagate_abs_tol", 1.0e-8, None),
    ):
        for threshold in THRESHOLD_GRID:
            if sweep_kind == "store_abs_tol":
                store_abs_tol = float(threshold)
                propagate_abs_tol = float(held_prop)
            else:
                store_abs_tol = float(held_store)
                propagate_abs_tol = float(threshold)
            for method in ("dense", "thick"):
                config = config_map[method]
                rows.append(
                    {
                        "sweep_kind": sweep_kind,
                        "threshold_value": float(threshold),
                        "method": method,
                        "config": config,
                        "store_abs_tol": store_abs_tol,
                        "propagate_abs_tol": propagate_abs_tol,
                    }
                )
    return rows


def _execution_order(job_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        job_specs,
        key=lambda job: (
            0 if str(job["sweep_kind"]) == "store_abs_tol" else 1,
            -float(job["threshold_value"]),
            0 if str(job["method"]) == "dense" else 1,
        ),
    )


def _validate_case_files(case: SweepCaseSpec) -> None:
    donor = dsp.DONOR_N2_CASES[case.donor_case_name]
    if not donor.donor_ezfio.exists():
        raise FileNotFoundError(f"Missing donor EZFIO for {case.name}: {donor.donor_ezfio}")
    if not case.fcidump_path.exists():
        raise FileNotFoundError(f"Missing donor FCIDUMP for {case.name}: {case.fcidump_path}")


def _load_fcidump_hamiltonian(fcidump_path: Path) -> tuple[object, object, float, int, tuple[int, int]]:
    from pyscf import ao2mo
    from pyscf.tools import fcidump as pyscf_fcidump

    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec_tot = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec_tot + ms2) // 2
    nelecb = nelec_tot - neleca
    eri8 = ao2mo.restore(8, data["H2"], norb)
    return data["H1"], eri8, float(data.get("ECORE", 0.0)), norb, (int(neleca), int(nelecb))


def _run_exact_fci_subprocess(
    fcidump_path: Path,
    *,
    timeout_s: int | None,
) -> dict[str, object]:
    script = r"""
import json
import sys
import time
from pathlib import Path
from pyscf import ao2mo, fci
from pyscf.tools import fcidump as pyscf_fcidump

fcidump_path = Path(sys.argv[1])
t0 = time.perf_counter()
data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
norb = int(data["NORB"])
nelec_tot = int(data["NELEC"])
ms2 = int(data.get("MS2", 0))
neleca = (nelec_tot + ms2) // 2
nelecb = nelec_tot - neleca
ecore = float(data.get("ECORE", 0.0))
h1 = data["H1"]
eri8 = ao2mo.restore(8, data["H2"], norb)
solver = fci.direct_spin1.FCI()
solver.conv_tol = 1e-12
solver.max_cycle = 200
solver.max_space = 100
e_elec, _ = solver.kernel(h1, eri8, norb, (int(neleca), int(nelecb)))
print(
    json.dumps(
        {
            "reference_kind": "exact_fci",
            "reference_label": "Exact FCI",
            "norb": norb,
            "nelec": [int(neleca), int(nelecb)],
            "ecore_eh": ecore,
            "electronic_energy_eh": float(e_elec),
            "total_energy_eh": float(e_elec + ecore),
            "wall_s": float(time.perf_counter() - t0),
        }
    )
)
"""
    run = subprocess.run(
        [sys.executable, "-c", script, str(fcidump_path)],
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if run.returncode != 0:
        raise RuntimeError(run.stderr.strip() or run.stdout.strip() or f"Exact FCI subprocess failed with rc={run.returncode}")
    return json.loads(run.stdout.strip())


def _normalize_reference_summary(summary: dict[str, object]) -> dict[str, object]:
    normalized = dict(summary)
    reference_kind = str(normalized.get("reference_kind") or normalized.get("method") or "").strip().lower()
    if reference_kind in {"", "dmrg"}:
        normalized["reference_kind"] = "dmrg"
        normalized.setdefault("reference_label", "DMRG")
    elif reference_kind in {"exact_fci", "fci"}:
        normalized["reference_kind"] = "exact_fci"
        normalized.setdefault("reference_label", "Exact FCI")
    else:
        normalized.setdefault("reference_kind", reference_kind)
        normalized.setdefault("reference_label", str(normalized["reference_kind"]))
    return normalized


def _load_or_compute_exact_reference(
    *,
    case: SweepCaseSpec,
    outdir: Path,
    force: bool,
    timeout_s: int | None = None,
) -> dict[str, object]:
    path = outdir / "reference_exact_fci_summary.json"
    if path.exists() and not force:
        return _normalize_reference_summary(json.loads(path.read_text(encoding="utf-8")))

    if timeout_s is None:
        h1, eri8, ecore, ncas, nelec = _load_fcidump_hamiltonian(case.fcidump_path)
        t0 = time.perf_counter()
        total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
        summary = {
            "reference_kind": "exact_fci",
            "reference_label": "Exact FCI",
            "total_energy_eh": total,
            "ecore_eh": ecore,
            "norb": int(ncas),
            "nelec": [int(nelec[0]), int(nelec[1])],
            "wall_s": float(time.perf_counter() - t0),
        }
    else:
        summary = _run_exact_fci_subprocess(case.fcidump_path, timeout_s=timeout_s)
    summary = _normalize_reference_summary(summary)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _hf_occ_vector(norb: int, nelec_tot: int) -> list[float]:
    if norb <= 0:
        return []
    occs = [0.0] * int(norb)
    ndoubly = max(0, int(nelec_tot) // 2)
    for i in range(min(int(norb), ndoubly)):
        occs[i] = 2.0
    if int(nelec_tot) % 2 == 1 and ndoubly < int(norb):
        occs[ndoubly] = 1.0
    return occs


def _attempt_exact_reference_with_fallback(case: SweepCaseSpec, outdir: Path, *, force: bool) -> dict[str, object]:
    attempt_path = outdir / "reference_exact_fci_attempt.json"
    exact_path = outdir / "reference_exact_fci_summary.json"
    if exact_path.exists() and not force:
        return json.loads(exact_path.read_text(encoding="utf-8"))
    if attempt_path.exists() and not force:
        return json.loads(attempt_path.read_text(encoding="utf-8"))
    try:
        summary = _run_exact_fci_subprocess(case.fcidump_path, timeout_s=case.exact_timeout_s)
    except subprocess.TimeoutExpired:
        payload = {
            "status": "timeout",
            "timeout_s": int(case.exact_timeout_s),
            "fcidump_path": str(case.fcidump_path),
        }
        attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    except Exception as exc:
        payload = {
            "status": "failed",
            "timeout_s": int(case.exact_timeout_s),
            "fcidump_path": str(case.fcidump_path),
            "error": str(exc),
        }
        attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    exact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_or_compute_dmrg_reference(
    *,
    case: SweepCaseSpec,
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
    if case.cached_dmrg_reference_path is not None and case.cached_dmrg_reference_path.exists() and not force:
        summary = _normalize_reference_summary(json.loads(case.cached_dmrg_reference_path.read_text(encoding="utf-8")))
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
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
    data = pyscf_fcidump.read(str(case.fcidump_path), verbose=False)
    hf_occs = _hf_occ_vector(int(data["NORB"]), int(data["NELEC"]))

    bond_dims = [250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4000]
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
    dmrg.init_hamiltonian_fcidump("c1", str(fcidump_path))
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
            "CKA_FCIDUMP": str(case.fcidump_path),
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


def _load_or_compute_reference(
    *,
    case: SweepCaseSpec,
    outdir: Path,
    force: bool,
    dmrg_memory_gb: int,
    dmrg_threads: int,
    dmrg_conv_tol: float,
    dmrg_python: Path,
    reference_kind: str,
) -> dict[str, object]:
    if reference_kind == "auto" and case.name != "n2_cas12_12_R2p50":
        exact_attempt = _attempt_exact_reference_with_fallback(case, outdir, force=force)
        if exact_attempt.get("reference_kind") == "exact_fci":
            return exact_attempt
    return _load_or_compute_dmrg_reference(
        case=case,
        outdir=outdir,
        force=force,
        memory_gb=dmrg_memory_gb,
        threads=dmrg_threads,
        conv_tol=dmrg_conv_tol,
        dmrg_python=dmrg_python,
    )


def _fixed_threshold_text(sweep_kind: str) -> str:
    if sweep_kind == "store_abs_tol":
        return "fixed propagate_abs_tol = 1e-8"
    return "fixed store_abs_tol = 1e-8"


def _run_dir_for(case: SweepCaseSpec, *, sweep_kind: str, method: str, threshold: float) -> Path:
    return Path("runs") / sweep_kind / method / _threshold_tag(threshold)


def _make_threshold_params(case: SweepCaseSpec, config: StructuralConfig, *, store_abs_tol: float, propagate_abs_tol: float) -> QPCCIKParams:
    external_keep = int(case.internal_external_keep)
    return QPCCIKParams(
        ccik_basis_size=int(config.basis_size),
        ccik_generator_keep=1,
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_max_parents_for_expansion=0,
        ccik_include_current_support=True,
        ccik_selector_keep=1,
        ccik_external_keep=external_keep,
        ccik_external_candidate_factor=1,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_compression_policy="threshold",
        ccik_store_abs_tol=float(store_abs_tol),
        ccik_propagate_abs_tol=float(propagate_abs_tol),
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


def _make_calibration_params(case: SweepCaseSpec, config: StructuralConfig) -> QPCCIKParams:
    keep = int(min(case.internal_external_keep, CALIBRATION_KEEP))
    return QPCCIKParams(
        ccik_basis_size=int(config.basis_size),
        ccik_generator_keep=int(CALIBRATION_GENERATOR_KEEP),
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=0.0,
        ccik_max_parents_for_expansion=0,
        ccik_include_current_support=True,
        ccik_selector_keep=keep,
        ccik_external_keep=keep,
        ccik_external_candidate_factor=2,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_compression_policy="topk",
        ccik_store_abs_tol=0.0,
        ccik_propagate_abs_tol=0.0,
        ccik_nkeep=keep,
        ccik_tmp_keep_factor=int(CALIBRATION_TMP_KEEP_FACTOR),
        ccik_preserve_prev_support=True,
        ccik_preserve_prev_support_cap=int(CALIBRATION_PRESERVE_CAP),
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


def _prepare_run_ezfio(
    qp,
    case: SweepCaseSpec,
    run_dir: Path,
    params: QPCCIKParams,
    *,
    force: bool,
) -> Path:
    donor = dsp.DONOR_N2_CASES[case.donor_case_name]
    run_dir = run_dir.resolve()
    ezfio = run_dir / f"{case.name}.ezfio"
    if ezfio.exists() and force:
        shutil.rmtree(ezfio)
    if not ezfio.exists():
        shutil.copytree(donor.donor_ezfio, ezfio)
    ensure_ezfio_defaults(ezfio)
    dsp._set_mo_class_direct(qp, ezfio, donor.mo_num_hint, donor.ncore_hint, donor.ncas_hint)
    dsp._force_integral_rebuild(qp, ezfio)
    set_ccik_params(qp, ezfio, params)
    write_ezfio_value(ezfio, "ccik/ccik_m", int(params.ccik_basis_size))
    write_ezfio_value(ezfio, "ccik/ccik_debug_trace", True)
    write_ezfio_value(ezfio, "ccik/ccik_debug_trace_topn", 16)
    return ezfio


def _build_trace_args(qp_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        qp_root=qp_root,
        parity_basis_mode="qp_fcidump",
        matvec_backend="connected_sd_fullq",
        omp_threads="auto",
    )


def _first_step_meta(qp_summary: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    hq_steps = qp_summary.get("hq_steps")
    if not isinstance(hq_steps, dict) or not hq_steps:
        return None, None
    first_step = min(int(key) for key in hq_steps)
    payload = hq_steps[str(first_step)] if str(first_step) in hq_steps else hq_steps[first_step]
    if not isinstance(payload, dict):
        return None, None
    return payload.get("hq_meta"), payload.get("next_meta")


def _run_single_qp_job(
    *,
    case: SweepCaseSpec,
    reference: dict[str, object],
    qp_root: Path,
    config: StructuralConfig,
    store_abs_tol: float,
    propagate_abs_tol: float,
    run_dir: Path,
    omp_threads: int,
    run_mode: str,
    timeout_s: int | None,
    force: bool,
) -> dict[str, object]:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    qp = discover_qp_paths_from_root(qp_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(tempfile.mkdtemp(prefix=f"{case.name}_{config.method}_{config.tag}_", dir="/tmp"))
    if run_mode == "calibration_surrogate":
        params = _make_calibration_params(case, config)
    else:
        params = _make_threshold_params(case, config, store_abs_tol=store_abs_tol, propagate_abs_tol=propagate_abs_tol)
    ezfio = _prepare_run_ezfio(qp, case, runtime_dir, params, force=True)

    args = _build_trace_args(qp_root)
    args.omp_threads = int(omp_threads)
    t0 = time.perf_counter()
    failure = None
    energy = None
    qp_summary_path = run_dir / "qp_summary.json"
    try:
        qp_summary = dsp.run_qp_trace(args, run_dir, prepared=(runtime_dir, ezfio), timeout_s=timeout_s)
        energy = qp_summary.get("final_total_energy")
    except Exception as exc:
        qp_summary = None
        failure = str(exc)
    wall_s = float(time.perf_counter() - t0)
    shutil.rmtree(runtime_dir, ignore_errors=True)

    if qp_summary is None and qp_summary_path.exists():
        qp_summary = json.loads(qp_summary_path.read_text(encoding="utf-8"))

    first_hq, first_next = (None, None) if qp_summary is None else _first_step_meta(qp_summary)
    delta = None if energy is None else float((float(energy) - float(reference["total_energy_eh"])) * 1000.0)
    payload = {
        "case": case.name,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "method": config.method,
        "config_tag": config.tag,
        "store_abs_tol": float(store_abs_tol),
        "propagate_abs_tol": float(propagate_abs_tol),
        "e_total_eh": None if energy is None else float(energy),
        "delta_vs_ref_mEh": delta,
        "abs_delta_vs_ref_mEh": None if delta is None else abs(float(delta)),
        "wall_s": wall_s,
        "omp_threads": int(omp_threads),
        "run_mode": run_mode,
        "timeout_s": None if timeout_s is None else int(timeout_s),
        "compression_policy": params.ccik_compression_policy,
        "basis_size": int(config.basis_size),
        "max_cycle": int(config.max_cycle),
        "restart_m_cycle": int(config.restart_m_cycle),
        "restart_nroot": int(config.restart_nroot),
        "failure": failure,
        "artifacts": {
            "qp_summary": str(qp_summary_path) if qp_summary_path.exists() else None,
            "stdout": str(run_dir / "ccik_stdout.log"),
            "stderr": str(run_dir / "ccik_stderr.log"),
            "trace": str(run_dir / "ccik_debug_trace.log"),
            "runtime_dir": None,
        },
        "first_step_hq": first_hq,
        "first_step_next": first_next,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _select_config(
    *,
    case: SweepCaseSpec,
    method: str,
    ladder: list[StructuralConfig],
    reference: dict[str, object],
    qp_root: Path,
    outdir: Path,
    omp_threads: int,
    force: bool,
) -> tuple[StructuralConfig, list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    for config in ladder:
        run_dir = outdir / "calibration" / method / config.tag
        result = _run_single_qp_job(
            case=case,
            reference=reference,
            qp_root=qp_root,
            config=config,
            store_abs_tol=0.0,
            propagate_abs_tol=0.0,
            run_dir=run_dir,
            omp_threads=omp_threads,
            run_mode="calibration_surrogate",
            timeout_s=None,
            force=force,
        )
        results.append(result)

    successful = [row for row in results if row["e_total_eh"] is not None]
    if not successful:
        raise RuntimeError(f"Calibration failed for {case.name} {method}: no successful surrogate runs")

    for idx in range(len(successful) - 1):
        delta = abs(float(successful[idx]["e_total_eh"]) - float(successful[idx + 1]["e_total_eh"])) * 1000.0
        successful[idx]["next_delta_mEh"] = delta
        if delta < EXACT_DELTA_TOL_MEH:
            tag = str(successful[idx]["config_tag"])
            chosen = next(config for config in ladder if config.tag == tag)
            return chosen, results

    tag = str(successful[-1]["config_tag"])
    chosen = next(config for config in ladder if config.tag == tag)
    return chosen, results


def _load_or_run_calibration(
    *,
    case: SweepCaseSpec,
    reference: dict[str, object],
    qp_root: Path,
    outdir: Path,
    omp_threads: int,
    force: bool,
) -> dict[str, object]:
    path = outdir / "calibration.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    dense_choice, dense_results = _select_config(
        case=case,
        method="dense",
        ladder=_dense_ladder(case),
        reference=reference,
        qp_root=qp_root,
        outdir=outdir,
        omp_threads=omp_threads,
        force=force,
    )
    thick_choice, thick_results = _select_config(
        case=case,
        method="thick",
        ladder=_thick_ladder(case),
        reference=reference,
        qp_root=qp_root,
        outdir=outdir,
        omp_threads=omp_threads,
        force=force,
    )
    payload = {
        "calibration_mode": "surrogate_topk",
        "dense": {
            "chosen": {
                "tag": dense_choice.tag,
                "basis_size": dense_choice.basis_size,
                "max_cycle": dense_choice.max_cycle,
                "restart_m_cycle": dense_choice.restart_m_cycle,
                "restart_nroot": dense_choice.restart_nroot,
            },
            "runs": dense_results,
        },
        "thick": {
            "chosen": {
                "tag": thick_choice.tag,
                "basis_size": thick_choice.basis_size,
                "max_cycle": thick_choice.max_cycle,
                "restart_m_cycle": thick_choice.restart_m_cycle,
                "restart_nroot": thick_choice.restart_nroot,
            },
            "runs": thick_results,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _build_preflight_cases(case: SweepCaseSpec) -> list[tuple[str, float, float]]:
    if case.name == "n2_cas12_12_R2p50":
        return [
            ("zero", 0.0, 0.0),
            ("store_nonzero", 1.0e-5, 1.0e-8),
            ("prop_nonzero", 1.0e-8, 1.0e-5),
        ]
    return [
        ("zero", 0.0, 0.0),
        ("nonzero_sanity", 1.0e-5, 1.0e-8),
    ]


def _run_preflight_gate(case: SweepCaseSpec, *, qp_root: Path, outdir: Path, omp_threads: int, force: bool) -> dict[str, object]:
    path = outdir / "preflight_gate.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    cases_out: list[dict[str, object]] = []
    for label, store_abs_tol, propagate_abs_tol in _build_preflight_cases(case):
        run_dir = outdir / "preflight" / label
        cached = run_dir / "run_summary.json"
        if cached.exists() and not force:
            summary = json.loads(cached.read_text(encoding="utf-8"))
        else:
            args = argparse.Namespace(
                case=case.donor_case_name,
                steps=1,
                selector_mode="abs_residual",
                matvec_backend="connected_sd_fullq",
                compression_policy="threshold",
                store_abs_tol=float(store_abs_tol),
                propagate_abs_tol=float(propagate_abs_tol),
                trace_topn=16,
                parity_basis_mode="qp_fcidump",
                with_qp=True,
                qp_root=qp_root,
                omp_threads=int(omp_threads),
                keep_ezfio=False,
                R=None,
                sweep=None,
            )
            donor = dsp.DONOR_N2_CASES[case.donor_case_name]
            summary = dsp._run_single_case_backend(args, donor, "connected_sd_fullq", base_outdir=run_dir)
        qp_summary = json.loads((run_dir / "qp_summary.json").read_text(encoding="utf-8"))
        result = {
            "label": label,
            "store_abs_tol": float(store_abs_tol),
            "propagate_abs_tol": float(propagate_abs_tol),
            "energy_gap_mEh": summary.get("energy_gap_mEh"),
            "final_step_projected_l2": summary.get("final_step_projected_l2"),
            "final_step_qnext_l2": summary.get("final_step_qnext_l2"),
            "observed_markers": qp_summary.get("observed_markers"),
            "trace_layout": qp_summary.get("trace_layout"),
            "stale_binary": qp_summary.get("likely_stale_generated_binary"),
            "artifacts": {
                "run_summary": str(run_dir / "run_summary.json"),
                "qp_summary": str(run_dir / "qp_summary.json"),
                "compare": str(run_dir / "python_qp_compare.json"),
            },
        }
        cases_out.append(result)

    marker_gate = all(
        bool(all(bool(v) for v in result["observed_markers"].values()))
        and bool(all(bool(v) for v in result["trace_layout"].values()))
        and not bool(result["stale_binary"])
        for result in cases_out
    )
    zero = next(result for result in cases_out if result["label"] == "zero")
    numerical_gate = True
    if case.name == "n2_cas12_12_R2p50":
        store = next(result for result in cases_out if result["label"] == "store_nonzero")
        prop = next(result for result in cases_out if result["label"] == "prop_nonzero")
        numerical_gate = (
            float(zero["final_step_projected_l2"]) <= 1.0e-5
            and float(zero["final_step_qnext_l2"]) <= 1.0e-5
            and float(store["final_step_projected_l2"]) <= 1.0e-5
            and float(store["final_step_qnext_l2"]) <= 1.0e-4
            and float(prop["final_step_projected_l2"]) <= 1.0e-5
            and float(prop["final_step_qnext_l2"]) <= 1.0e-4
        )
    payload = {
        "case": case.name,
        "cases": cases_out,
        "gates": {
            "marker_gate": bool(marker_gate),
            "numerical_gate": bool(numerical_gate),
            "overall_passed": bool(marker_gate and numerical_gate),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _to_log_abs(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        fval = float(value)
        if not math.isfinite(fval):
            out.append(float("nan"))
        else:
            out.append(max(abs(fval), PLOT_FLOOR_MEH))
    return out


def _write_plot(
    outdir: Path,
    *,
    case: SweepCaseSpec,
    reference_label: str,
    sweep_kind: str,
    x_values: list[float],
    dense_vals: list[float],
    thick_vals: list[float],
    dense_failed: list[bool],
    thick_failed: list[bool],
) -> None:
    dense_plot = _to_log_abs(dense_vals)
    thick_plot = _to_log_abs(thick_vals)
    finite_vals = [value for value in dense_plot + thick_plot if math.isfinite(value)]
    if finite_vals:
        ymin = min(finite_vals)
        ymax = max(finite_vals)
        failure_y = 10.0 ** (math.ceil(math.log10(max(ymax, PLOT_FLOOR_MEH))) + 0.5)
        axis_ymin = 10.0 ** (math.floor(math.log10(max(ymin, PLOT_FLOOR_MEH))) - 0.5)
        axis_ymax = failure_y * 1.8
    else:
        failure_y = 1.0
        axis_ymin = PLOT_FLOOR_MEH
        axis_ymax = 10.0

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(x_values, dense_plot, marker="o", label="dense CCIK")
    ax.plot(x_values, thick_plot, marker="s", label="thick CCIK")
    dense_fail_x = [x for x, failed in zip(x_values, dense_failed, strict=True) if failed]
    thick_fail_x = [x for x, failed in zip(x_values, thick_failed, strict=True) if failed]
    if dense_fail_x:
        ax.scatter(dense_fail_x, [failure_y] * len(dense_fail_x), marker="x", s=48, linewidths=1.5, label="dense fail")
    if thick_fail_x:
        ax.scatter(thick_fail_x, [failure_y] * len(thick_fail_x), marker="x", s=48, linewidths=1.5, label="thick fail")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(sweep_kind)
    ax.set_ylabel(rf"$|\Delta E|$ vs {reference_label} (mEh)")
    ax.set_title(case.title)
    ax.text(
        0.03,
        0.97,
        _fixed_threshold_text(sweep_kind),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    if dense_fail_x or thick_fail_x:
        ax.text(
            0.03,
            0.83,
            "x = run failed / guard triggered",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(axis_ymin, axis_ymax)
    fig.tight_layout()
    stem = outdir / f"{sweep_kind}_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def _write_progress(progress_path: Path, *, case: SweepCaseSpec, reference: dict[str, object], rows: list[dict[str, object]], total_rows: int) -> None:
    progress = {
        "case": case.name,
        "reference_kind": reference["reference_kind"],
        "reference_label": reference["reference_label"],
        "completed_rows": len(rows),
        "total_rows": total_rows,
        "best_dense_abs_delta_mEh": min(
            (float(r["abs_delta_vs_ref_mEh"]) for r in rows if r["method"] == "dense" and r["abs_delta_vs_ref_mEh"] is not None),
            default=None,
        ),
        "best_thick_abs_delta_mEh": min(
            (float(r["abs_delta_vs_ref_mEh"]) for r in rows if r["method"] == "thick" and r["abs_delta_vs_ref_mEh"] is not None),
            default=None,
        ),
    }
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _should_infer_monotone_failure(failure: str | None) -> bool:
    if not failure:
        return False
    return any(marker in failure for marker in FAILURE_INFERENCE_MARKERS)


def _infer_smaller_threshold_failures(
    *,
    failed_job: dict[str, object],
    failure: str,
    job_specs: list[dict[str, object]],
    rows_by_key: dict[tuple[str, float, str], dict[str, object]],
    case: SweepCaseSpec,
    reference: dict[str, object],
) -> int:
    inferred = 0
    sweep_kind = str(failed_job["sweep_kind"])
    method = str(failed_job["method"])
    failed_threshold = float(failed_job["threshold_value"])
    config = failed_job["config"]
    for job in job_specs:
        if str(job["sweep_kind"]) != sweep_kind or str(job["method"]) != method:
            continue
        threshold = float(job["threshold_value"])
        if threshold >= failed_threshold:
            continue
        key = (sweep_kind, threshold, method)
        if key in rows_by_key:
            continue
        rows_by_key[key] = {
            "case": case.name,
            "reference_kind": reference["reference_kind"],
            "reference_label": reference["reference_label"],
            "sweep_kind": sweep_kind,
            "threshold_name": sweep_kind,
            "threshold_value": threshold,
            "method": method,
            "config_tag": str(config.tag),
            "store_abs_tol": float(job["store_abs_tol"]),
            "propagate_abs_tol": float(job["propagate_abs_tol"]),
            "e_total_eh": None,
            "delta_vs_ref_mEh": None,
            "abs_delta_vs_ref_mEh": None,
            "wall_s": 0.0,
            "basis_size": int(config.basis_size),
            "max_cycle": int(config.max_cycle),
            "restart_m_cycle": int(config.restart_m_cycle),
            "restart_nroot": int(config.restart_nroot),
            "failure": (
                "Inferred monotone failure from tighter-threshold run. "
                f"threshold={failed_threshold:.1e}, method={method}, sweep={sweep_kind}. "
                f"Original failure: {failure.splitlines()[0]}"
            ),
        }
        inferred += 1
    return inferred


def _require_qp_root(explicit: Path | None) -> Path:
    resolved = resolve_rebuildable_qp_root(explicit)
    if resolved is not None:
        return resolved
    details = {
        "explicit": None if explicit is None else validate_rebuildable_qp_root(explicit),
        "env": None if "QP_ROOT" not in os.environ else validate_rebuildable_qp_root(Path(os.environ["QP_ROOT"])),
        "manifest": None,
    }
    manifest_root = infer_qp_root_from_rebuild_manifest(DEFAULT_QP_REBUILD_MANIFEST)
    if manifest_root is not None:
        details["manifest"] = {
            "manifest_path": str(DEFAULT_QP_REBUILD_MANIFEST),
            "resolved_root": str(manifest_root),
            "validation": validate_rebuildable_qp_root(manifest_root),
        }
    raise RuntimeError(
        "No rebuildable Quantum Package root found. "
        "Expected a root with quantum_package.rc, config/build.ninja, and the installed ccik plugin.\n"
        + json.dumps(details, indent=2, sort_keys=True)
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = CASES[args.case]
    _validate_case_files(case)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    qp_root = _require_qp_root(args.qp_root)
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

    if not args.skip_preflight:
        preflight = _run_preflight_gate(case, qp_root=qp_root, outdir=outdir, omp_threads=omp_threads, force=args.force)
        if not preflight["gates"]["overall_passed"]:
            raise RuntimeError(
                f"Preflight gate failed for {case.name}. See {outdir / 'preflight_gate.json'} for details."
            )
    else:
        preflight = {"skipped": True}

    reference = _load_or_compute_reference(
        case=case,
        outdir=outdir,
        force=args.force,
        dmrg_memory_gb=args.dmrg_memory_gb,
        dmrg_threads=args.dmrg_threads,
        dmrg_conv_tol=args.dmrg_conv_tol,
        dmrg_python=args.dmrg_python,
        reference_kind=args.reference_kind,
    )
    calibration = _load_or_run_calibration(
        case=case,
        reference=reference,
        qp_root=qp_root,
        outdir=outdir,
        omp_threads=omp_threads,
        force=args.force,
    )
    dense_choice = StructuralConfig(method="dense", **calibration["dense"]["chosen"])
    thick_choice = StructuralConfig(method="thick", **calibration["thick"]["chosen"])

    csv_path = outdir / "threshold_sweep.csv"
    progress_path = outdir / "progress.json"
    rows_by_key = {} if args.force else _load_cached_rows(csv_path)
    job_specs = _build_case_matrix(case, dense_choice, thick_choice)
    ordered_jobs = _execution_order(job_specs)

    for job in ordered_jobs:
        key = (str(job["sweep_kind"]), float(job["threshold_value"]), str(job["method"]))
        if key in rows_by_key:
            continue
        config = job["config"]
        run_dir = outdir / _run_dir_for(
            case,
            sweep_kind=str(job["sweep_kind"]),
            method=str(job["method"]),
            threshold=float(job["threshold_value"]),
        )
        result = _run_single_qp_job(
            case=case,
            reference=reference,
            qp_root=qp_root,
            config=config,
            store_abs_tol=float(job["store_abs_tol"]),
            propagate_abs_tol=float(job["propagate_abs_tol"]),
            run_dir=run_dir,
            omp_threads=omp_threads,
            run_mode="threshold_sweep",
            timeout_s=args.row_timeout_s,
            force=args.force,
        )
        row = {
            "case": case.name,
            "reference_kind": reference["reference_kind"],
            "reference_label": reference["reference_label"],
            "sweep_kind": str(job["sweep_kind"]),
            "threshold_name": str(job["sweep_kind"]),
            "threshold_value": float(job["threshold_value"]),
            "method": str(job["method"]),
            "config_tag": str(config.tag),
            "store_abs_tol": float(job["store_abs_tol"]),
            "propagate_abs_tol": float(job["propagate_abs_tol"]),
            "e_total_eh": result["e_total_eh"],
            "delta_vs_ref_mEh": result["delta_vs_ref_mEh"],
            "abs_delta_vs_ref_mEh": result["abs_delta_vs_ref_mEh"],
            "wall_s": result["wall_s"],
            "basis_size": int(config.basis_size),
            "max_cycle": int(config.max_cycle),
            "restart_m_cycle": int(config.restart_m_cycle),
            "restart_nroot": int(config.restart_nroot),
            "failure": result["failure"],
        }
        rows_by_key[key] = row
        if _should_infer_monotone_failure(result["failure"]):
            _infer_smaller_threshold_failures(
                failed_job=job,
                failure=str(result["failure"]),
                job_specs=job_specs,
                rows_by_key=rows_by_key,
                case=case,
                reference=reference,
            )
        rows = _sorted_rows(rows_by_key)
        _write_rows_csv(csv_path, rows)
        _write_progress(progress_path, case=case, reference=reference, rows=rows, total_rows=len(job_specs))
        status = "failed" if result["e_total_eh"] is None else f"{float(result['delta_vs_ref_mEh']):+.6f} mEh"
        print(
            f"[{case.name}] {job['sweep_kind']} threshold={job['threshold_value']:.1e} method={job['method']} -> {status}",
            flush=True,
        )

    rows = _sorted_rows(rows_by_key)
    _write_rows_csv(csv_path, rows)
    _write_progress(progress_path, case=case, reference=reference, rows=rows, total_rows=len(job_specs))

    for sweep_kind in ("store_abs_tol", "propagate_abs_tol"):
        sweep_rows = [row for row in rows if row["sweep_kind"] == sweep_kind]
        x_values = [float(row["threshold_value"]) for row in sweep_rows if row["method"] == "dense"]
        dense_vals = [
            float("nan") if row["abs_delta_vs_ref_mEh"] is None else float(row["abs_delta_vs_ref_mEh"])
            for row in sweep_rows
            if row["method"] == "dense"
        ]
        thick_vals = [
            float("nan") if row["abs_delta_vs_ref_mEh"] is None else float(row["abs_delta_vs_ref_mEh"])
            for row in sweep_rows
            if row["method"] == "thick"
        ]
        dense_failed = [bool(row["failure"]) for row in sweep_rows if row["method"] == "dense"]
        thick_failed = [bool(row["failure"]) for row in sweep_rows if row["method"] == "thick"]
        _write_plot(
            outdir,
            case=case,
            reference_label=str(reference["reference_label"]),
            sweep_kind=sweep_kind,
            x_values=x_values,
            dense_vals=dense_vals,
            thick_vals=thick_vals,
            dense_failed=dense_failed,
            thick_failed=thick_failed,
        )

    summary = {
        "case": case.name,
        "reference": reference,
        "preflight": preflight,
        "calibration": calibration,
        "omp_threads": int(omp_threads),
        "threshold_grid": THRESHOLD_GRID,
        "fixed_thresholds": {
            "store_abs_tol_sweep": {"propagate_abs_tol": 1.0e-8},
            "propagate_abs_tol_sweep": {"store_abs_tol": 1.0e-8},
        },
        "artifacts": {
            "csv": str(csv_path),
            "progress": str(progress_path),
            "store_plot_png": str(outdir / "store_abs_tol_sweep.png"),
            "store_plot_pdf": str(outdir / "store_abs_tol_sweep.pdf"),
            "prop_plot_png": str(outdir / "propagate_abs_tol_sweep.png"),
            "prop_plot_pdf": str(outdir / "propagate_abs_tol_sweep.pdf"),
            "qp_plugin_rebuild": str(outdir / "qp_plugin_rebuild.json"),
        },
        "sweeps": {},
    }
    for sweep_kind in ("store_abs_tol", "propagate_abs_tol"):
        sweep_rows = [row for row in rows if row["sweep_kind"] == sweep_kind]
        summary["sweeps"][sweep_kind] = {}
        for method in ("dense", "thick"):
            vals = [float(row["abs_delta_vs_ref_mEh"]) for row in sweep_rows if row["method"] == method and row["abs_delta_vs_ref_mEh"] is not None]
            summary["sweeps"][sweep_kind][method] = {
                "n_rows": sum(1 for row in sweep_rows if row["method"] == method),
                "n_failures": sum(1 for row in sweep_rows if row["method"] == method and row["failure"]),
                "best_abs_delta_mEh": None if not vals else min(vals),
                "worst_abs_delta_mEh": None if not vals else max(vals),
            }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
