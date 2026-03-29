"""Exact-QP N2 CAS(10,10) validation gate for the production CCIK backend."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ccik import build_cas_hamiltonian_pyscf, exact_cas_fci_energy_pyscf, make_mol_pyscf  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from n2_utils import make_n2_atom  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    CCIKParams,
    QPPaths,
    discover_qp_paths,
    ensure_ezfio_defaults,
    run_ccik,
    set_ccik_params,
)


R_POINTS: tuple[float, ...] = (1.10, 1.80, 2.85)
CAS = CASSpec(ncas=10, nelecas=10, ncore=2)
DEFAULT_DONOR = _ROOT / "experiments" / "n2_qp_ccik_simplified_smoke" / "n2_1p10.ezfio"


@dataclass(frozen=True)
class PointData:
    R: float
    e_fci_total: float


def _build_point(R: float) -> PointData:
    mol = make_mol_pyscf(
        atom=make_n2_atom(float(R)),
        basis="6-31g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=CAS)
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    return PointData(R=float(R), e_fci_total=e_fci_total)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_n2_xyz(path: Path, R: float) -> None:
    path.write_text(f"2\nN2 {R:.2f} A\nN 0 0 0\nN 0 0 {R:.2f}\n", encoding="ascii")


def _write_centered_n2_coords(ezfio: Path, R_ang: float) -> None:
    half_z_bohr = float(R_ang) / 0.52917721092 / 2.0
    payload = (
        "  2\n"
        "                   2                    3 \n"
        "   0.000000000000000e+00\n"
        "   0.000000000000000e+00\n"
        "   0.000000000000000e+00\n"
        "   0.000000000000000e+00\n"
        f"  {-half_z_bohr: .15e}\n"
        f"   {half_z_bohr: .15e}\n"
    )
    with gzip.open(ezfio / "nuclei" / "nucl_coord.gz", "wt", encoding="ascii") as fh:
        fh.write(payload)


def _set_mo_class_direct(qp: QPPaths, ezfio: Path, *, mo_num: int, ncore: int, ncas: int) -> None:
    script = f"""
from ezfio import ezfio
ezfio.set_file(r"{ezfio}")
ezfio.set_mo_basis_mo_num({int(mo_num)})
mo_class = []
for i in range(1, {int(mo_num)} + 1):
    if i <= {int(ncore)}:
        mo_class.append("Core")
    elif i <= {int(ncore + ncas)}:
        mo_class.append("Active")
    else:
        mo_class.append("Deleted")
ezfio.set_mo_basis_mo_class(mo_class)
ezfio.set_mo_basis_mo_symmetry([1] * {int(mo_num)})
"""
    env = dict(os.environ)
    env["QP_ROOT"] = str(qp.qp_root)
    subprocess.run(
        ["/bin/bash", "-lc", f"source {qp.quantum_package_rc} && python3 - <<'PY'\n{script}\nPY"],
        check=True,
        text=True,
        env=env,
    )


def _prepare_n2_from_donor(qp: QPPaths, donor_ezfio: Path, ezfio: Path, *, R_ang: float) -> None:
    shutil.copytree(donor_ezfio, ezfio)
    _write_centered_n2_coords(ezfio, R_ang)
    ensure_ezfio_defaults(ezfio)
    bash_cmd = (
        f"source {qp.quantum_package_rc} && "
        f"qp set_file {ezfio} && "
        "qp set scf_utils thresh_scf 1.e-10 && "
        "qp set ao_two_e_ints io_ao_two_e_integrals Write && "
        "qp set mo_two_e_ints io_mo_two_e_integrals Write && "
        f"{qp.scf_binary} {ezfio}"
    )
    subprocess.run(["/bin/bash", "-lc", bash_cmd], check=True, text=True)
    _set_mo_class_direct(qp, ezfio, mo_num=18, ncore=CAS.ncore, ncas=CAS.ncas)


def _boolify(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _int_or(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _float_or(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _qp_params_from_selected(method: str, selected: dict[str, Any]) -> CCIKParams:
    generator_keep = _int_or(selected.get("generator_keep"), 256)
    nkeep_py = _int_or(selected.get("nkeep"), 12000)
    nadd_py = _int_or(selected.get("nadd"), 2000)
    kv_py = _int_or(selected.get("Kv"), 4000)
    selector_keep = max(nkeep_py, nadd_py + kv_py)
    external_keep = selector_keep
    nkeep = nkeep_py
    max_parents = selected.get("max_parents_for_expansion")
    if max_parents is None:
        max_parents = generator_keep

    common = dict(
        ccik_generator_keep=generator_keep,
        ccik_generator_fraction=0.0,
        ccik_generator_abs_coeff_cutoff=_float_or(selected.get("generator_abs_coeff_cutoff"), 0.0),
        ccik_max_parents_for_expansion=int(max_parents),
        ccik_include_current_support=_boolify(selected.get("include_current_support"), True),
        ccik_selector_keep=int(selector_keep),
        ccik_external_keep=int(external_keep),
        ccik_external_candidate_factor=4,
        ccik_pt2_rank_weight=1.0,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_nkeep=int(nkeep),
        ccik_tmp_keep_factor=_int_or(selected.get("tmp_keep_factor"), 4),
        ccik_preserve_prev_support=_boolify(selected.get("preserve_prev_support"), True),
        ccik_preserve_prev_support_cap=_int_or(selected.get("preserve_prev_support_cap"), 2048),
        ccik_total_det_budget=0,
        ccik_conv_tol=1.0e-12,
        ccik_ritz_overlap_tol=1.0e-8,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=False,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=False,
        ccik_verbose=False,
    )
    if method == "ccik":
        return CCIKParams(
            ccik_basis_size=_int_or(selected.get("m"), 32),
            ccik_max_cycle=1,
            ccik_restart_m_cycle=0,
            ccik_restart_nroot=2,
            **common,
        )
    if method == "ccik_thick":
        return CCIKParams(
            ccik_basis_size=24,
            ccik_max_cycle=_int_or(selected.get("ncycles"), 10),
            ccik_restart_m_cycle=_int_or(selected.get("m_cycle"), 5),
            ccik_restart_nroot=_int_or(selected.get("nroot"), 2),
            **common,
        )
    raise ValueError(f"Unsupported method {method!r}")


def _load_selected_params(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ccik": dict(data["ccik"]),
        "ccik_thick": dict(data["ccik_thick"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact-QP N2 CAS(10,10) validation gate")
    p.add_argument(
        "--selected-params",
        type=Path,
        default=_ROOT / "experiments" / "n2_accuracy_gate_20260320" / "selected_params.json",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=_ROOT / "experiments" / "n2_qp_exact_gate_20260320",
    )
    p.add_argument(
        "--donor-ezfio",
        type=Path,
        default=DEFAULT_DONOR,
        help="Working donor EZFIO used to avoid qp_create_ezfio during exact-QP checks",
    )
    p.add_argument("--gate-tol-eh", type=float, default=1.0e-3)
    p.add_argument("--n-det-max", type=int, default=2_000_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    qp = discover_qp_paths()
    if not args.donor_ezfio.exists():
        raise RuntimeError(f"Missing donor EZFIO: {args.donor_ezfio}")
    selected = _load_selected_params(args.selected_params)
    points = {R: _build_point(R) for R in R_POINTS}
    rows: list[dict[str, object]] = []

    for method in ("ccik", "ccik_thick"):
        params = _qp_params_from_selected(method, selected[method])
        method_dir = outdir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for R in R_POINTS:
            point = points[R]
            tag = f"n2_{R:.2f}".replace(".", "p")
            scratch = Path(tempfile.mkdtemp(prefix=f"{method}_{tag}_", dir="/tmp"))
            xyz = method_dir / f"{tag}.xyz"
            ezfio = scratch / f"{tag}.ezfio"
            log_path = method_dir / f"{tag}.ccik.log"
            _create_n2_xyz(xyz, R)
            try:
                _prepare_n2_from_donor(qp, args.donor_ezfio, ezfio, R_ang=R)
                set_ccik_params(qp, ezfio, params)
                subprocess.run(
                    [
                        "/bin/bash",
                        "-lc",
                        f"source {qp.quantum_package_rc} && qp set_file {ezfio} && qp set determinants n_det_max {args.n_det_max}",
                    ],
                    check=True,
                    text=True,
                )
                t0 = time.perf_counter()
                e_elec, e_tot, basis, out = run_ccik(qp, ezfio)
                runtime_s = float(time.perf_counter() - t0)
                log_path.write_text(out, encoding="utf-8", errors="ignore")
            finally:
                shutil.rmtree(scratch, ignore_errors=True)

            rows.append(
                {
                    "method": method,
                    "R": R,
                    "e_fci_total": point.e_fci_total,
                    "qp_ccik_electronic_eh": e_elec,
                    "qp_ccik_total_eh": e_tot,
                    "abs_err_eh": abs(e_tot - point.e_fci_total),
                    "basis_size": basis,
                    "runtime_s": runtime_s,
                    **{k: v for k, v in params.__dict__.items()},
                }
            )

    _write_csv(outdir / "results.csv", rows)
    gate = {}
    for method in ("ccik", "ccik_thick"):
        subset = [row for row in rows if row["method"] == method]
        gate[method] = {
            "max_abs_err_eh": max(float(row["abs_err_eh"]) for row in subset),
            "passed": max(float(row["abs_err_eh"]) for row in subset) <= args.gate_tol_eh,
        }
    gate["overall_passed"] = bool(gate["ccik"]["passed"] and gate["ccik_thick"]["passed"])
    (outdir / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_lines = [
        "# N2 QP Exact Gate",
        "",
        f"- Gate tol (Eh): {args.gate_tol_eh}",
        f"- Selected params file: `{args.selected_params}`",
        f"- Overall passed: {gate['overall_passed']}",
        "",
    ]
    for method in ("ccik", "ccik_thick"):
        summary_lines.append(f"- `{method}`: max |ΔE| = `{gate[method]['max_abs_err_eh']:.6e} Eh`, passed = `{gate[method]['passed']}`")
    (outdir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(gate, indent=2))
    return 0 if gate["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
