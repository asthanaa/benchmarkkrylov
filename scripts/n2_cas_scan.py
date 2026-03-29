"""Legacy-compatible N2 bond scan driver for the three supported CCIK solvers.

This file remains at the old path for users who prefer a TOML-driven scan
workflow. New learners should start in ``examples/`` instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    cas_spec_from_dict,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_stochastic,
    ccik_params_from_dict,
    ccik_stochastic_params_from_dict,
    ccik_thick_restart_params_from_dict,
    exact_cas_fci_energy_pyscf,
    load_config,
    make_mol_pyscf,
    run_methods_from_dict,
)

from n2_utils import make_n2_atom


def _canonical_method(method: str) -> str:
    name = method.strip().lower().replace("-", "_")
    if name == "ccik":
        return "ccik"
    if name in ("ccik_thick", "ccik_thick_restart", "reuse_ritz", "reuseritz"):
        return "ccik_thick"
    if name in ("ccik_stochastic", "ccikstochastic", "stochastic", "fciqmckrylov", "fciqmc_krylov"):
        return "ccik_stochastic"
    return method.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="N2 CAS scan using the supported CCIK solvers")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config (default: configs/config.toml if present, otherwise packaged defaults)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <repo>/out_n2_cas_scan.csv)",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        local_default = _ROOT / "configs" / "config.toml"
        if local_default.exists():
            config_path = str(local_default)

    cfg = load_config(config_path)
    methods = [_canonical_method(m) for m in run_methods_from_dict(cfg.get("run", {}))]

    scan = cfg.get("scan", {})
    Rs = np.linspace(float(scan.get("R_min", 2.0)), float(scan.get("R_max", 3.0)), int(scan.get("n_points", 4)))

    molcfg = cfg.get("molecule", {})
    basis = str(molcfg.get("basis", "6-31g"))
    unit = str(molcfg.get("unit", "Angstrom"))
    charge = int(molcfg.get("charge", 0))
    spin = int(molcfg.get("spin", 0))
    verbose = int(molcfg.get("verbose", 0))

    cas = cas_spec_from_dict(cfg.get("cas", {}))
    ccik_params = ccik_params_from_dict(cfg.get("ccik", {}))
    thick_params = ccik_thick_restart_params_from_dict(cfg.get("ccik_thick", {}))
    stochastic_params = ccik_stochastic_params_from_dict(cfg.get("ccik_stochastic", {}))

    multi = len(methods) > 1
    header = ["R_Ang", "e_fci_total"]
    if multi:
        for method in methods:
            header.extend([f"{method}_total", f"{method}_delta"])
    else:
        header.extend(["e_alg_total", "delta"])
    rows: list[list[float]] = []

    for R in Rs:
        t_R0 = time.perf_counter()
        print(f"\n=== R={float(R):.4f} Å ===", flush=True)
        mol = make_mol_pyscf(
            atom=make_n2_atom(float(R)),
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            verbose=verbose,
        )

        if spin == 0:
            expected_ne = 2 * int(cas.ncore) + int(cas.nelecas)
            if int(mol.nelectron) != int(expected_ne):
                raise ValueError(
                    "CAS electron mismatch for singlet: expected mol.nelectron == 2*ncore + nelecas. "
                    f"Got mol.nelectron={mol.nelectron}, ncore={cas.ncore}, nelecas={cas.nelecas} (2*ncore+nelecas={expected_ne})."
                )

        h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
        t0 = time.perf_counter()
        e_fci_tot = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
        print(f"CAS-FCI done in {time.perf_counter() - t0:.2f}s | E={e_fci_tot:+.10f}", flush=True)

        row = [float(R), e_fci_tot]
        msg_parts = [f"R={R:5.2f} Å", f"CAS-FCI={e_fci_tot:+.10f}"]

        for method in methods:
            t_method0 = time.perf_counter()
            st: dict[str, int] = {}
            if method == "ccik":
                print("  -> running CCIK...", flush=True)
                e_alg_cas = ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=ccik_params, stats=st)
                label = "CCIK"
            elif method == "ccik_thick":
                print("  -> running CCIK(thick)...", flush=True)
                e_alg_cas = ccik_ground_energy_dense_thick_restart(
                    h1,
                    eri8,
                    ncas,
                    nelec,
                    params=thick_params,
                    stats=st,
                )
                label = "CCIK(thick)"
            elif method == "ccik_stochastic":
                print("  -> running CCIK-stochastic...", flush=True)
                e_alg_cas = ccik_ground_energy_stochastic(
                    h1,
                    eri8,
                    ncas,
                    nelec,
                    params=stochastic_params,
                    stats=st,
                )
                label = "CCIK-stochastic"
            else:
                raise ValueError(
                    f"Unknown run method {method!r}. Expected one of: ccik, ccik_thick, ccik_stochastic."
                )

            dt_method = time.perf_counter() - t_method0
            e_alg_tot = float(ecore + e_alg_cas)
            delta = float(e_alg_tot - e_fci_tot)
            row.extend([e_alg_tot, delta])
            ndet_info = f"ndet_sum={st.get('ndet_sum', -1)}"
            msg_parts.append(f"{label}={e_alg_tot:+.10f} (Δ={delta:+.3e}, {ndet_info}, t={dt_method:.2f}s)")

        rows.append(row)
        print(" | ".join(msg_parts))
        print(f"R={float(R):.4f} Å done in {time.perf_counter() - t_R0:.2f}s", flush=True)

    out_path = Path(args.out) if args.out is not None else (_ROOT / "out_n2_cas_scan.csv")
    np.savetxt(out_path, np.asarray(rows, dtype=float), delimiter=",", header=",".join(header), comments="")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
