"""Compare deterministic variational CIPSI on the N2 CAS(14,14) hard point.

This reuses the local experimental CIPSI implementation in
`experiments/krylov cipsi dense reuseRitz.py` because the local Quantum Package
`qp_run` wrapper is currently broken on this machine.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

from n2_utils import make_n2_atom


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "n2_hardpoint_cipsi_var_compare"
DEFAULT_EXACT = -108.774883260748

sys.path.insert(0, str(ROOT / "src"))
from ccik.pyscf_cas import CASSpec, build_cas_hamiltonian_pyscf


def load_local_cipsi_module():
    path = ROOT / "experiments" / "krylov cipsi dense reuseRitz.py"
    spec = importlib.util.spec_from_file_location("krylov_cipsi_dense_reuseRitz", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load CIPSI module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run deterministic variational CIPSI on the N2 hard point")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--R", type=float, default=2.5)
    p.add_argument("--ncas", type=int, default=14)
    p.add_argument("--nelecas", type=int, default=14)
    p.add_argument("--ncore", type=int, default=0)
    p.add_argument("--ndet-max", type=int, nargs="+", default=[8192, 16384])
    p.add_argument("--niter", type=int, default=20)
    p.add_argument("--nadd", type=int, default=2000)
    p.add_argument("--Kv", type=int, default=3000)
    p.add_argument("--davidson-tol", type=float, default=1e-10)
    p.add_argument("--exact-total-eh", type=float, default=DEFAULT_EXACT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cipsi_mod = load_local_cipsi_module()
    atom = make_n2_atom(float(args.R))
    from pyscf import gto

    mol = gto.M(atom=atom, basis=args.basis, unit="Angstrom", charge=0, spin=0, verbose=0)
    h1eff, eri8, ecore, ncas, nelec_cas = build_cas_hamiltonian_pyscf(
        mol,
        cas=CASSpec(ncas=int(args.ncas), nelecas=int(args.nelecas), ncore=int(args.ncore)),
        orbital_method="casci",
    )

    rows: list[dict[str, object]] = []
    for ndet_max in args.ndet_max:
        t0 = time.time()
        e_cas = cipsi_mod.cipsi_dense_variational(
            h1eff,
            eri8,
            ncas,
            nelec_cas,
            niter=int(args.niter),
            nadd=int(args.nadd),
            ndet_max=int(ndet_max),
            Kv=int(args.Kv),
            davidson_tol=float(args.davidson_tol),
            verbose=False,
        )
        wall_s = time.time() - t0
        e_tot = ecore + e_cas
        err = abs(e_tot - float(args.exact_total_eh))
        row = {
            "R_ang": f"{float(args.R):.2f}",
            "basis": args.basis,
            "ncas": int(args.ncas),
            "nelecas": int(args.nelecas),
            "ncore": int(args.ncore),
            "ndet_max": int(ndet_max),
            "niter": int(args.niter),
            "nadd": int(args.nadd),
            "Kv": int(args.Kv),
            "cipsi_total_eh": f"{e_tot:.12f}",
            "reference_total_eh": f"{float(args.exact_total_eh):.12f}",
            "abs_err_eh": f"{err:.6e}",
            "wall_s": f"{wall_s:.2f}",
        }
        rows.append(row)
        print(row, flush=True)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "N2 hard-point deterministic variational CIPSI comparison",
        f"basis = {args.basis}",
        f"R = {args.R:.2f} Angstrom",
        f"CAS = ({args.nelecas},{args.ncas})",
        f"ncore = {args.ncore}",
        f"niter = {args.niter}",
        f"nadd = {args.nadd}",
        f"Kv = {args.Kv}",
        "",
    ]
    for row in rows:
        lines.append(
            f"ndet_max={row['ndet_max']}  E={row['cipsi_total_eh']}  "
            f"err={row['abs_err_eh']}  wall_s={row['wall_s']}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
