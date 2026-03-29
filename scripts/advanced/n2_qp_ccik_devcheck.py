"""Run a small N2 development check for the Quantum Package ccik backend.

This script:

1. creates per-geometry EZFIO inputs with Quantum Package
2. runs RHF in QP
3. assigns the CAS(10,10) active space for N2 / 6-31g
4. runs the direct `ccik` plugin binary
5. computes a PySCF CASCI reference
6. writes a CSV summary
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pyscf import gto, mcscf, scf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "qp_ccik_n2_devcheck"


@dataclass(frozen=True)
class QPPaths:
    qp_root: Path
    quantum_package_rc: Path
    qp_set_mo_class: Path
    qp_create_ezfio: str
    scf_binary: Path
    ccik_binary: Path


def bash(cmd: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-lc", cmd],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def discover_qp_paths() -> QPPaths:
    qp_root = os.environ.get("QP_ROOT")
    if not qp_root:
        raise RuntimeError("QP_ROOT is not set. Source quantum_package.rc first.")
    root = Path(qp_root)
    return QPPaths(
        qp_root=root,
        quantum_package_rc=root / "quantum_package.rc",
        qp_set_mo_class=root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio="qp_create_ezfio",
        scf_binary=root / "src" / "hartree_fock" / "scf",
        ccik_binary=root / "plugins" / "ccik_qp" / "stable" / "ccik" / "ccik",
    )


def create_n2_xyz(path: Path, r_ang: float) -> None:
    path.write_text(
        f"2\nN2 {r_ang:.2f} A\nN 0 0 0\nN 0 0 {r_ang:.2f}\n",
        encoding="ascii",
    )


def run_qp_setup(qp: QPPaths, ezfio: Path, xyz: Path) -> None:
    bash(
        f"source {qp.quantum_package_rc} && "
        f"{qp.qp_create_ezfio} -b 6-31g -m 1 -o {ezfio} {xyz}"
    )
    bash(
        f"source {qp.quantum_package_rc} && "
        f"qp set_file {ezfio} && "
        "qp set scf_utils thresh_scf 1.e-10 && "
        "qp set ao_two_e_ints io_ao_two_e_integrals Write && "
        "qp set mo_two_e_ints io_mo_two_e_integrals Write"
    )
    subprocess.run([str(qp.scf_binary), str(ezfio)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bash(
        f"source {qp.quantum_package_rc} && "
        f"{qp.qp_set_mo_class} --core=\"[1-2]\" --act=\"[3-12]\" --del=\"[13-18]\" {ezfio}"
    )


def set_ccik_params(qp: QPPaths, ezfio: Path, args: argparse.Namespace) -> None:
    bash(
        f"source {qp.quantum_package_rc} && "
        f"qp set_file {ezfio} && "
        f"qp set ccik ccik_m {args.ccik_m} && "
        f"qp set ccik ccik_generator_keep {args.ccik_generator_keep} && "
        f"qp set ccik ccik_selector_keep {args.ccik_selector_keep} && "
        f"qp set ccik ccik_external_keep {args.ccik_external_keep} && "
        f"qp set ccik ccik_nkeep {args.ccik_nkeep} && "
        f"qp set ccik ccik_verbose {str(args.ccik_verbose).lower()}"
    )


def run_ccik(qp: QPPaths, ezfio: Path) -> tuple[float, float, int]:
    out = subprocess.run([str(qp.ccik_binary), str(ezfio)], check=True, text=True, capture_output=True).stdout
    e_elec = _extract_value(out, "CCIK_FINAL_ELECTRONIC_ENERGY")
    e_tot = _extract_value(out, "CCIK_FINAL_TOTAL_ENERGY")
    basis = int(_extract_value(out, "CCIK_FINAL_BASIS"))
    return e_elec, e_tot, basis


def _extract_value(text: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)} =\s*([-0-9.]+)", text)
    if not m:
        raise RuntimeError(f"Could not find `{key}` in plugin output")
    return float(m.group(1))


def run_pyscf_reference(r_ang: float) -> float:
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {r_ang}", basis="6-31g", unit="Angstrom", spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10, verbose=0)
    mc = mcscf.CASCI(mf, 10, 10)
    mc.ncore = 2
    e_tot, *_ = mc.kernel()
    return float(e_tot)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="N2 development check for the QP ccik backend")
    p.add_argument("--points", nargs="+", type=float, default=[1.00, 1.10, 1.50, 2.00, 2.50])
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--keep-ezfio", action="store_true")
    p.add_argument("--ccik-m", type=int, default=16)
    p.add_argument("--ccik-generator-keep", type=int, default=512)
    p.add_argument("--ccik-selector-keep", type=int, default=8192)
    p.add_argument("--ccik-external-keep", type=int, default=8192)
    p.add_argument("--ccik-nkeep", type=int, default=8192)
    p.add_argument("--ccik-verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    qp = discover_qp_paths()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for r_ang in args.points:
        tag = f"{r_ang:.2f}".replace(".", "p")
        xyz = out_dir / f"n2_{tag}.xyz"
        ezfio = out_dir / f"n2_{tag}.ezfio"
        if ezfio.exists():
            shutil.rmtree(ezfio)
        create_n2_xyz(xyz, r_ang)
        run_qp_setup(qp, ezfio, xyz)
        set_ccik_params(qp, ezfio, args)
        e_elec, e_tot, basis = run_ccik(qp, ezfio)
        e_ref = run_pyscf_reference(r_ang)
        rows.append(
            {
                "R_ang": f"{r_ang:.2f}",
                "pyscf_total_eh": f"{e_ref:.12f}",
                "qp_ccik_electronic_eh": f"{e_elec:.12f}",
                "qp_ccik_total_eh": f"{e_tot:.12f}",
                "abs_err_eh": f"{abs(e_tot - e_ref):.6e}",
                "basis_size": basis,
            }
        )
        if not args.keep_ezfio:
            shutil.rmtree(ezfio)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "R_ang",
                "pyscf_total_eh",
                "qp_ccik_electronic_eh",
                "qp_ccik_total_eh",
                "abs_err_eh",
                "basis_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(csv_path)
    for row in rows:
        print(
            f"R={row['R_ang']}  ref={row['pyscf_total_eh']}  "
            f"qp={row['qp_ccik_total_eh']}  err={row['abs_err_eh']}  basis={row['basis_size']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
