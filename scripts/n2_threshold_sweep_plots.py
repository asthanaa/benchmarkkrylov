from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_TEST_ROOT = ROOT / "experiments" / "threshold_test"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKParams, CCIKThickRestartParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402


THRESHOLD_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
LARGE_SPACE_THRESHOLD_GRID = [1e-6, 1e-4, 1e-2]
DEFAULT_DMRG_PYTHON = ROOT / "experiments" / "block2_cr2_d2p0_sto3g" / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class SweepCaseSpec:
    name: str
    title: str
    basis: str
    geometry_ang: float
    ncas: int
    nelecas: int
    ncore: int
    source_kind: str
    reference_kind: str
    reference_label: str
    dense_kwargs: dict[str, Any]
    thick_kwargs: dict[str, Any]
    threshold_grid: list[float]
    fixed_store_abs_tol: float
    fixed_propagate_abs_tol: float
    fcidump_path: Path | None = None
    reference_json_path: Path | None = None
    reference_json_energy_key: str | None = None


CASES: dict[str, SweepCaseSpec] = {
    "n2_cas10e8": SweepCaseSpec(
        name="n2_cas10e8",
        title="N2 / 6-31G / R = 2.50 A / CAS(10e,8o)",
        basis="6-31g",
        geometry_ang=2.50,
        ncas=8,
        nelecas=10,
        ncore=2,
        source_kind="pyscf",
        reference_kind="exact_fci",
        reference_label="FCI",
        dense_kwargs={
            "m": 24,
            "candidate_mode": "full",
            "selector_mode": "abs_residual",
        },
        thick_kwargs={
            "m_cycle": 16,
            "ncycles": 16,
            "nroot": 4,
            "tol": 1e-8,
            "candidate_mode": "full",
            "selector_mode": "abs_residual",
        },
        threshold_grid=list(reversed(THRESHOLD_GRID)),
        fixed_store_abs_tol=1e-8,
        fixed_propagate_abs_tol=1e-8,
    ),
    "n2_cas12_12": SweepCaseSpec(
        name="n2_cas12_12",
        title="N2 / 6-31G / R = 2.50 A / CAS(12e,12o)",
        basis="6-31g",
        geometry_ang=2.50,
        ncas=12,
        nelecas=12,
        ncore=1,
        source_kind="fcidump",
        reference_kind="exact_fci",
        reference_label="FCI",
        dense_kwargs={
            "m": 12,
            "candidate_mode": "connected_sd",
            "selector_mode": "abs_residual",
        },
        thick_kwargs={
            "m_cycle": 8,
            "ncycles": 8,
            "nroot": 3,
            "tol": 1e-8,
            "candidate_mode": "connected_sd",
            "selector_mode": "abs_residual",
        },
        threshold_grid=list(reversed(THRESHOLD_GRID)),
        fixed_store_abs_tol=1e-8,
        fixed_propagate_abs_tol=1e-8,
        fcidump_path=THRESHOLD_TEST_ROOT
        / "donor_n2_parity_sweep_20260320_rerun"
        / "n2_cas12_12_R2p50"
        / "connected_sd_fullq"
        / "qp_active.FCIDUMP",
    ),
    "n2_cas14_14": SweepCaseSpec(
        name="n2_cas14_14",
        title="N2 / 6-31G / R = 2.50 A / CAS(14e,14o)",
        basis="6-31g",
        geometry_ang=2.50,
        ncas=14,
        nelecas=14,
        ncore=0,
        source_kind="fcidump",
        reference_kind="external_json",
        reference_label="Python oracle",
        dense_kwargs={
            "m": 12,
            "candidate_mode": "connected_sd",
            "selector_mode": "abs_residual",
        },
        thick_kwargs={
            "m_cycle": 8,
            "ncycles": 8,
            "nroot": 3,
            "tol": 1e-8,
            "candidate_mode": "connected_sd",
            "selector_mode": "abs_residual",
        },
        threshold_grid=THRESHOLD_GRID,
        fixed_store_abs_tol=1e-8,
        fixed_propagate_abs_tol=1e-8,
        fcidump_path=THRESHOLD_TEST_ROOT
        / "donor_n2_parity_sweep_20260320_rerun"
        / "n2_cas14_14_R2p50"
        / "connected_sd_fullq"
        / "qp_active.FCIDUMP",
        reference_json_path=THRESHOLD_TEST_ROOT
        / "donor_n2_parity_sweep_20260320_rerun"
        / "n2_cas14_14_R2p50"
        / "connected_sd_fullq"
        / "python_summary.json",
        reference_json_energy_key="summary.ritz_energy",
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
    "store_abs_tol",
    "propagate_abs_tol",
    "e_total_eh",
    "delta_vs_ref_mEh",
    "wall_s",
    "m_eff",
    "ndet_union",
    "ndet_sum",
    "failure",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="Threshold sweep plots for donor-backed N2 active spaces")
    p.add_argument("--case", choices=sorted(CASES), default="n2_cas10e8")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="Ignore cached reference/CSV rows and recompute all sweep points")
    p.add_argument("--dmrg-memory-gb", type=int, default=64, help="Only used for DMRG-backed references")
    p.add_argument("--dmrg-threads", type=int, default=16, help="Only used for DMRG-backed references")
    p.add_argument("--dmrg-conv-tol", type=float, default=1e-8, help="Only used for DMRG-backed references")
    p.add_argument("--dmrg-python", type=Path, default=DEFAULT_DMRG_PYTHON, help="Python interpreter with pyblock2 installed")
    args = p.parse_args(argv)
    if args.outdir is None:
        args.outdir = THRESHOLD_TEST_ROOT / f"{args.case}_threshold_sweep_{stamp}"
    return args


def _row_key(row: dict[str, object]) -> tuple[str, float, str]:
    return (str(row["sweep_kind"]), float(row["threshold_value"]), str(row["method"]))


def _parse_optional_float(value: str) -> float | None:
    text = value.strip()
    if text == "":
        return None
    return float(text)


def _parse_optional_int(value: str) -> int | None:
    text = value.strip()
    if text == "":
        return None
    return int(float(text))


def _load_cached_rows(csv_path: Path) -> dict[tuple[str, float, str], dict[str, object]]:
    if not csv_path.exists():
        return {}
    rows: dict[tuple[str, float, str], dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row: dict[str, object] = {
                "case": raw.get("case", ""),
                "reference_kind": raw.get("reference_kind", ""),
                "reference_label": raw.get("reference_label", ""),
                "sweep_kind": raw["sweep_kind"],
                "threshold_name": raw["threshold_name"],
                "threshold_value": float(raw["threshold_value"]),
                "method": raw["method"],
                "store_abs_tol": float(raw["store_abs_tol"]),
                "propagate_abs_tol": float(raw["propagate_abs_tol"]),
                "e_total_eh": _parse_optional_float(raw["e_total_eh"]),
                "delta_vs_ref_mEh": _parse_optional_float(raw.get("delta_vs_ref_mEh", raw.get("delta_vs_fci_mEh", ""))),
                "wall_s": _parse_optional_float(raw["wall_s"]),
                "m_eff": _parse_optional_int(raw["m_eff"]),
                "ndet_union": _parse_optional_int(raw["ndet_union"]),
                "ndet_sum": _parse_optional_int(raw["ndet_sum"]),
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


def _load_fcidump_case(case: SweepCaseSpec) -> tuple[object, object, float, int, tuple[int, int]]:
    from pyscf import ao2mo
    from pyscf.tools import fcidump as pyscf_fcidump

    if case.fcidump_path is None or not case.fcidump_path.exists():
        raise FileNotFoundError(f"Missing donor-backed FCIDUMP for {case.name}: {case.fcidump_path}")
    data = pyscf_fcidump.read(str(case.fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec_tot = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec_tot + ms2) // 2
    nelecb = nelec_tot - neleca
    h1 = data["H1"]
    eri8 = ao2mo.restore(8, data["H2"], norb)
    ecore = float(data.get("ECORE", 0.0))
    return h1, eri8, ecore, norb, (int(neleca), int(nelecb))


def _build_case_hamiltonian(case: SweepCaseSpec) -> tuple[object, object, float, int, tuple[int, int]]:
    if case.source_kind == "pyscf":
        mol = make_mol_pyscf(
            atom=f"N 0 0 0; N 0 0 {case.geometry_ang}",
            basis=case.basis,
            unit="Angstrom",
            charge=0,
            spin=0,
            verbose=0,
        )
        cas = CASSpec(ncas=case.ncas, nelecas=case.nelecas, ncore=case.ncore)
        return build_cas_hamiltonian_pyscf(mol, cas=cas)
    if case.source_kind == "fcidump":
        return _load_fcidump_case(case)
    raise ValueError(f"Unsupported source_kind {case.source_kind!r}")


def _load_or_compute_exact_reference(
    *,
    outdir: Path,
    h1,
    eri8,
    ecore: float,
    ncas: int,
    nelec: tuple[int, int],
    force: bool,
) -> dict[str, object]:
    path = outdir / "reference_exact_fci.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    e_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    summary = {
        "reference_kind": "exact_fci",
        "reference_label": "FCI",
        "total_energy_eh": e_total,
        "wall_s": float(time.perf_counter() - t0),
        "norb": int(ncas),
        "nelec": [int(nelec[0]), int(nelec[1])],
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        return json.loads(path.read_text(encoding="utf-8"))

    if case.fcidump_path is None:
        raise ValueError("DMRG reference requires a donor-backed FCIDUMP path")

    runtime_dir = outdir / "reference_dmrg_runtime"
    scratch_dir = runtime_dir / "scratch"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "dmrg.log"
    stdout_path = runtime_dir / "subprocess.stdout"
    stderr_path = runtime_dir / "subprocess.stderr"

    bond_dims = [250, 500, 1000, 1500, 2000, 2500, 3000, 3000]
    noises = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0, 0.0, 0.0]
    dav_thrds = [1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9]

    from pyscf.tools import fcidump as pyscf_fcidump

    data = pyscf_fcidump.read(str(case.fcidump_path), verbose=False)
    ecore = float(data.get("ECORE", 0.0))
    dmrg_python = Path(dmrg_python)
    if not dmrg_python.exists():
        raise FileNotFoundError(f"DMRG Python interpreter not found: {dmrg_python}")

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

data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
ecore = float(data.get("ECORE", 0.0))
t0 = time.perf_counter()
with log_path.open("w", encoding="utf-8") as fh:
    old_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(scratch_dir)
    try:
        dmrg = GSDMRG(
            scratch=str(scratch_dir),
            memory=float(memory_gb) * 1e9,
            omp_threads=int(threads),
            verbose=3,
            print_statistics=True,
        )
        with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            dmrg.init_hamiltonian_fcidump("c1", str(fcidump_path))
            e_total = float(
                dmrg.dmrg(
                    bond_dims=bond_dims,
                    noises=noises,
                    dav_thrds=dav_thrds,
                    occs=None,
                    n_steps=32,
                    conv_tol=conv_tol,
                    cutoff=1e-14,
                )
            )
        del dmrg
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
        }
    )
    run = subprocess.run(
        [str(dmrg_python), "-c", script],
        text=True,
        capture_output=True,
        env=env,
    )
    stdout_path.write_text(run.stdout, encoding="utf-8")
    stderr_path.write_text(run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(
            "DMRG reference subprocess failed with rc="
            f"{run.returncode}; see {log_path}, {stdout_path}, and {stderr_path}"
        )
    summary = json.loads(run.stdout)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _extract_nested_value(payload: dict[str, object], dotted_key: str) -> object:
    value: object = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing key path {dotted_key!r}")
        value = value[part]
    return value


def _load_external_json_reference(
    *,
    case: SweepCaseSpec,
    outdir: Path,
    force: bool,
) -> dict[str, object]:
    path = outdir / "reference_external_json.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    if case.reference_json_path is None or case.reference_json_energy_key is None:
        raise ValueError("external_json reference requires reference_json_path and reference_json_energy_key")
    if not case.reference_json_path.exists():
        raise FileNotFoundError(f"Missing external reference JSON for {case.name}: {case.reference_json_path}")
    raw = json.loads(case.reference_json_path.read_text(encoding="utf-8"))
    total_energy = float(_extract_nested_value(raw, case.reference_json_energy_key))
    summary = {
        "reference_kind": "external_json",
        "reference_label": case.reference_label,
        "total_energy_eh": total_energy,
        "source_json": str(case.reference_json_path),
        "source_energy_key": case.reference_json_energy_key,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_or_compute_reference(
    *,
    case: SweepCaseSpec,
    outdir: Path,
    h1,
    eri8,
    ecore: float,
    ncas: int,
    nelec: tuple[int, int],
    force: bool,
    dmrg_memory_gb: int,
    dmrg_threads: int,
    dmrg_conv_tol: float,
    dmrg_python: Path,
) -> dict[str, object]:
    if case.reference_kind == "exact_fci":
        return _load_or_compute_exact_reference(
            outdir=outdir,
            h1=h1,
            eri8=eri8,
            ecore=ecore,
            ncas=ncas,
            nelec=nelec,
            force=force,
        )
    if case.reference_kind == "dmrg":
        return _load_or_compute_dmrg_reference(
            case=case,
            outdir=outdir,
            force=force,
            memory_gb=dmrg_memory_gb,
            threads=dmrg_threads,
            conv_tol=dmrg_conv_tol,
            dmrg_python=dmrg_python,
        )
    if case.reference_kind == "external_json":
        return _load_external_json_reference(case=case, outdir=outdir, force=force)
    raise ValueError(f"Unsupported reference_kind {case.reference_kind!r}")


def _dense_params(case: SweepCaseSpec, *, store_abs_tol: float, propagate_abs_tol: float) -> CCIKParams:
    kwargs = dict(case.dense_kwargs)
    kwargs.update(
        compression_policy="threshold",
        store_abs_tol=float(store_abs_tol),
        propagate_abs_tol=float(propagate_abs_tol),
        orth_tol=1e-12,
        verbose=False,
    )
    return CCIKParams(**kwargs)


def _thick_params(case: SweepCaseSpec, *, store_abs_tol: float, propagate_abs_tol: float) -> CCIKThickRestartParams:
    kwargs = dict(case.thick_kwargs)
    kwargs.update(
        compression_policy="threshold",
        store_abs_tol=float(store_abs_tol),
        propagate_abs_tol=float(propagate_abs_tol),
        orth_tol=1e-12,
        verbose=False,
    )
    return CCIKThickRestartParams(**kwargs)


def _run_method(
    *,
    case: SweepCaseSpec,
    method: str,
    h1,
    eri8,
    ecore: float,
    ncas: int,
    nelec: tuple[int, int],
    store_abs_tol: float,
    propagate_abs_tol: float,
) -> tuple[float | None, dict[str, object], float | None]:
    stats: dict[str, object] = {}
    t0 = time.perf_counter()
    try:
        if method == "dense":
            energy = float(
                ecore
                + ccik_ground_energy_dense(
                    h1,
                    eri8,
                    ncas,
                    nelec,
                    params=_dense_params(case, store_abs_tol=store_abs_tol, propagate_abs_tol=propagate_abs_tol),
                    stats=stats,
                )
            )
        elif method == "thick":
            energy = float(
                ecore
                + ccik_ground_energy_dense_thick_restart(
                    h1,
                    eri8,
                    ncas,
                    nelec,
                    params=_thick_params(case, store_abs_tol=store_abs_tol, propagate_abs_tol=propagate_abs_tol),
                    stats=stats,
                )
            )
        else:
            raise ValueError(f"Unsupported method {method!r}")
    except Exception as exc:  # pragma: no cover - runtime validation path
        stats["failure"] = str(exc)
        return None, stats, None
    return energy, stats, float(time.perf_counter() - t0)


def _to_log_abs(values: list[float]) -> list[float]:
    out: list[float] = []
    for val in values:
        fval = float(val)
        if not math.isfinite(fval):
            out.append(float("nan"))
            continue
        out.append(max(abs(fval), 1.0e-12))
    return out


def _write_plot(
    outdir: Path,
    *,
    case: SweepCaseSpec,
    sweep_kind: str,
    x_values: list[float],
    dense_vals: list[float],
    thick_vals: list[float],
    dense_failed: list[bool],
    thick_failed: list[bool],
    fixed_threshold_text: str,
) -> None:
    dense_plot = _to_log_abs(dense_vals)
    thick_plot = _to_log_abs(thick_vals)
    finite_vals = [v for v in dense_plot + thick_plot if math.isfinite(v)]
    if finite_vals:
        ymax = max(finite_vals)
        failure_y = 10.0 ** (math.ceil(math.log10(max(ymax, 1.0e-12))) + 0.5)
        ymin = min(finite_vals)
        ax_ymin = 10.0 ** (math.floor(math.log10(max(ymin, 1.0e-12))) - 0.5)
        ax_ymax = failure_y * 1.8
    else:
        failure_y = 1.0e6
        ax_ymin = 1.0e-6
        ax_ymax = 1.0e7

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
    ax.set_ylabel(rf"$|\Delta E|$ vs {case.reference_label} (mEh)")
    ax.set_title(case.title)
    ax.text(
        0.03,
        0.97,
        fixed_threshold_text,
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
            "x = threshold point failed\n(current support/candidate guard)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(ax_ymin, ax_ymax)
    fig.tight_layout()
    stem = outdir / f"{sweep_kind}_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def _finite_abs(values: list[float | None]) -> list[float]:
    out: list[float] = []
    for val in values:
        if val is None:
            continue
        fval = float(val)
        if math.isfinite(fval):
            out.append(abs(fval))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = CASES[args.case]
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "threshold_sweep.csv"

    h1, eri8, ecore, ncas, nelec = _build_case_hamiltonian(case)
    reference = _load_or_compute_reference(
        case=case,
        outdir=outdir,
        h1=h1,
        eri8=eri8,
        ecore=ecore,
        ncas=ncas,
        nelec=nelec,
        force=args.force,
        dmrg_memory_gb=args.dmrg_memory_gb,
        dmrg_threads=args.dmrg_threads,
        dmrg_conv_tol=args.dmrg_conv_tol,
        dmrg_python=args.dmrg_python,
    )
    e_ref = float(reference["total_energy_eh"])
    rows_by_key = {} if args.force else _load_cached_rows(csv_path)

    for sweep_kind, held_store, held_prop in (
        ("store_abs_tol", None, float(case.fixed_propagate_abs_tol)),
        ("propagate_abs_tol", float(case.fixed_store_abs_tol), None),
    ):
        for threshold in case.threshold_grid:
            if sweep_kind == "store_abs_tol":
                store_abs_tol = float(threshold)
                propagate_abs_tol = float(held_prop)
            else:
                store_abs_tol = float(held_store)
                propagate_abs_tol = float(threshold)

            for method in ("dense", "thick"):
                cache_key = (sweep_kind, float(threshold), method)
                if cache_key in rows_by_key:
                    continue
                energy, stats, wall_s = _run_method(
                    case=case,
                    method=method,
                    h1=h1,
                    eri8=eri8,
                    ecore=ecore,
                    ncas=ncas,
                    nelec=nelec,
                    store_abs_tol=store_abs_tol,
                    propagate_abs_tol=propagate_abs_tol,
                )
                row = {
                    "case": case.name,
                    "reference_kind": case.reference_kind,
                    "reference_label": case.reference_label,
                    "sweep_kind": sweep_kind,
                    "threshold_name": sweep_kind,
                    "threshold_value": float(threshold),
                    "method": method,
                    "store_abs_tol": float(store_abs_tol),
                    "propagate_abs_tol": float(propagate_abs_tol),
                    "e_total_eh": energy,
                    "delta_vs_ref_mEh": None if energy is None else float((energy - e_ref) * 1000.0),
                    "wall_s": wall_s,
                    "m_eff": stats.get("m_eff"),
                    "ndet_union": stats.get("ndet_union"),
                    "ndet_sum": stats.get("ndet_sum"),
                    "failure": stats.get("failure"),
                }
                rows_by_key[cache_key] = row
                _write_rows_csv(csv_path, _sorted_rows(rows_by_key))
                status = "failed" if energy is None else f"{float(row['delta_vs_ref_mEh']):+.6f} mEh"
                print(
                    f"[{case.name}] {sweep_kind} threshold={threshold:.1e} method={method} -> {status}",
                    flush=True,
                )

    rows = _sorted_rows(rows_by_key)
    _write_rows_csv(csv_path, rows)

    fixed_text_map = {
        "store_abs_tol": f"fixed propagate_abs_tol = {case.fixed_propagate_abs_tol:.0e}",
        "propagate_abs_tol": f"fixed store_abs_tol = {case.fixed_store_abs_tol:.0e}",
    }
    for sweep_kind in ("store_abs_tol", "propagate_abs_tol"):
        x_values = [row["threshold_value"] for row in rows if row["sweep_kind"] == sweep_kind and row["method"] == "dense"]
        dense_vals = [
            float("nan") if row["delta_vs_ref_mEh"] is None else float(row["delta_vs_ref_mEh"])
            for row in rows
            if row["sweep_kind"] == sweep_kind and row["method"] == "dense"
        ]
        dense_failed = [
            bool(row["failure"])
            for row in rows
            if row["sweep_kind"] == sweep_kind and row["method"] == "dense"
        ]
        thick_vals = [
            float("nan") if row["delta_vs_ref_mEh"] is None else float(row["delta_vs_ref_mEh"])
            for row in rows
            if row["sweep_kind"] == sweep_kind and row["method"] == "thick"
        ]
        thick_failed = [
            bool(row["failure"])
            for row in rows
            if row["sweep_kind"] == sweep_kind and row["method"] == "thick"
        ]
        _write_plot(
            outdir,
            case=case,
            sweep_kind=sweep_kind,
            x_values=x_values,
            dense_vals=dense_vals,
            thick_vals=thick_vals,
            dense_failed=dense_failed,
            thick_failed=thick_failed,
            fixed_threshold_text=fixed_text_map[sweep_kind],
        )

    summary = {
        "case": case.name,
        "system": {
            "molecule": "N2",
            "basis": "6-31G",
            "geometry_ang": case.geometry_ang,
            "ncas": case.ncas,
            "nelecas": case.nelecas,
            "ncore": case.ncore,
        },
        "reference_kind": case.reference_kind,
        "reference_label": case.reference_label,
        "reference_total_energy_eh": e_ref,
        "reference_summary": reference,
        "threshold_grid": case.threshold_grid,
        "dense_params": case.dense_kwargs,
        "thick_params": case.thick_kwargs,
        "fixed_thresholds": {
            "store_abs_tol_sweep": {"propagate_abs_tol": case.fixed_propagate_abs_tol},
            "propagate_abs_tol_sweep": {"store_abs_tol": case.fixed_store_abs_tol},
        },
        "artifacts": {
            "csv": str(csv_path),
            "store_plot_png": str((outdir / "store_abs_tol_sweep.png")),
            "store_plot_pdf": str((outdir / "store_abs_tol_sweep.pdf")),
            "prop_plot_png": str((outdir / "propagate_abs_tol_sweep.png")),
            "prop_plot_pdf": str((outdir / "propagate_abs_tol_sweep.pdf")),
        },
        "sweeps": {},
    }
    for sweep_kind in ("store_abs_tol", "propagate_abs_tol"):
        sweep_rows = [row for row in rows if row["sweep_kind"] == sweep_kind]
        for method in ("dense", "thick"):
            vals = _finite_abs([row["delta_vs_ref_mEh"] for row in sweep_rows if row["method"] == method])
            summary["sweeps"].setdefault(sweep_kind, {})[method] = {
                "n_rows": sum(1 for row in sweep_rows if row["method"] == method),
                "n_failures": sum(1 for row in sweep_rows if row["method"] == method and row["e_total_eh"] is None),
                "best_abs_delta_mEh": None if not vals else min(vals),
                "worst_abs_delta_mEh": None if not vals else max(vals),
            }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
