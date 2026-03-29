"""Cr2 PES scan with singlet CAS(10,10) FCI (PySCF) using `ccik` helpers.

Run from repo root:
  python scripts/cr2_scan.py

Outputs:
- `out_cr2_cas10_10_pes.csv` (R, energies, <S^2>, multiplicity)
- `cr2_cas10_10_pes.png` (simple PES plot)

Notes
-----
- This is *active-space* FCI (CAS-FCI): E_total = ecore + E_FCI(CAS).
- The active-space electronic state is enforced as a singlet via a spin-penalty projector.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np


# Allow running without installing the package (uses src/ layout)
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import build_cas_hamiltonian_pyscf, make_mol_pyscf  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402


def make_cr2_atom(R_ang: float) -> str:
    z = R_ang / 2.0
    return f"""
    Cr 0.0 0.0 {-z}
    Cr 0.0 0.0 {+z}
    """


def _infer_ncore(nelectron: int, nelecas: int, *, spin: int) -> int:
    if spin != 0:
        raise ValueError("This script enforces a singlet: expected spin=0")
    if nelecas % 2 != 0:
        raise ValueError("Singlet CAS requires an even nelecas")
    if nelectron < nelecas:
        raise ValueError(f"Total electrons {nelectron} < active electrons {nelecas}")
    rem = nelectron - nelecas
    if rem % 2 != 0:
        raise ValueError(
            f"(nelectron - nelecas) must be even for a closed-shell core. Got {nelectron}-{nelecas}={rem}."
        )
    return rem // 2


def _fci_singlet_energy_and_spin(
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    ncas: int,
    nelec_cas: tuple[int, int],
    spin_penalty_shift: float = 20.0,
) -> tuple[float, float, float]:
    """Return (E_FCI_CAS, <S^2>, multiplicity) for the lowest-energy singlet root.

    Important: for stretched bonds the true CAS ground state may become high-spin.
    The user request is to *always* report a singlet, so we search multiple roots
    and pick the lowest-energy one with <S^2> ~ 0.
    """

    from pyscf import fci
    from pyscf.fci import addons

    neleca, nelecb = nelec_cas
    if neleca != nelecb:
        raise ValueError(
            f"Singlet targeting requires neleca == nelecb, got nelec_cas={nelec_cas}. "
            "Double-check nelecas/spin/ncore."
        )

    base = fci.direct_spin1.FCI()
    # Target a singlet by adding a strong spin-penalty. This avoids the expensive multi-root
    # search (important for stretched bonds) while still returning a true singlet CI vector.
    targeted = addons.fix_spin(base, ss=0, shift=float(spin_penalty_shift))

    _e_penalized, ci = targeted.kernel(h1, eri8, ncas, nelec_cas)  # type: ignore[attr-defined]
    # The energy from the penalized Hamiltonian is not the physical energy.
    # Compute the true electronic energy for the *original* Hamiltonian.
    e_phys = base.energy(h1, eri8, ci, ncas, nelec_cas)
    ss, mult = base.spin_square(ci, ncas, nelec_cas)
    return float(e_phys), float(ss), float(mult)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cr2 singlet CAS(10,10) FCI PES scan")
    parser.add_argument("--basis", default="6-31g", help="AO basis (default: 6-31g)")
    parser.add_argument("--unit", default="Angstrom", help="Geometry unit (default: Angstrom)")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge (default: 0)")
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2S (PySCF convention). Enforced to be 0 for singlet (default: 0).",
    )
    parser.add_argument("--Rmin", type=float, default=1.6, help="Min bond length in Angstrom")
    parser.add_argument("--Rmax", type=float, default=2.4, help="Max bond length in Angstrom")
    parser.add_argument("--npoints", type=int, default=12, help="Number of PES points (default: 12)")
    parser.add_argument("--ncas", type=int, default=10, help="Number of active orbitals (default: 10)")
    parser.add_argument("--nelecas", type=int, default=10, help="Number of active electrons (default: 10)")
    parser.add_argument("--verbose", type=int, default=0, help="PySCF verbosity (default: 0)")
    parser.add_argument(
        "--spin-shift",
        type=float,
        default=20.0,
        help="Spin-penalty shift used to target singlet in CAS-FCI (default: 20.0)",
    )
    parser.add_argument(
        "--orbital-method",
        default="casscf",
        choices=("casci", "casscf", "casscf_then_casci"),
        help=(
            "Active-space orbitals: casci (RHF orbitals), casscf (orbital-optimized), "
            "or casscf_then_casci (optimize with CASSCF, then build CASCI Hamiltonian in optimized orbitals). "
            "Default: casscf"
        ),
    )
    parser.add_argument(
        "--x2c1e",
        action="store_true",
        help="Use scalar-relativistic X2C(1e) in the underlying RHF reference (PySCF mf.x2c1e())",
    )
    parser.add_argument(
        "--casscf-nroots",
        type=int,
        default=1,
        help="Number of singlet roots for (state-averaged) CASSCF. Default: 1",
    )
    parser.add_argument(
        "--state-average",
        action="store_true",
        help="Use state-averaged CASSCF over --casscf-nroots roots (helps avoid root flipping)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Output PNG plot path",
    )
    args = parser.parse_args()

    if args.spin != 0:
        raise ValueError("This script requires a singlet ground state: use --spin 0")

    ncas = int(args.ncas)
    nelecas = int(args.nelecas)
    if ncas <= 0 or nelecas <= 0:
        raise ValueError("--ncas and --nelecas must be positive")
    if nelecas % 2 != 0:
        raise ValueError("This script enforces a singlet: --nelecas must be even")

    stem = f"cr2_cas{ncas}_{nelecas}_pes"
    out_path = Path(str(args.out) if args.out is not None else str(_ROOT / f"out_{stem}.csv"))
    plot_path = Path(str(args.plot) if args.plot is not None else str(_ROOT / f"{stem}.png"))

    Rs = np.linspace(float(args.Rmin), float(args.Rmax), int(args.npoints))

    rows: list[tuple[float, float, float, float, float, float]] = []
    # columns: R, e_total, ecore, e_fci_cas, <S^2>, mult

    for R in Rs:
        if str(args.orbital_method).lower().startswith("casscf"):
            print(f"R={R:6.3f} Å | starting CASSCF orbital optimization...")

        mol = make_mol_pyscf(
            atom=make_cr2_atom(float(R)),
            basis=str(args.basis),
            unit=str(args.unit),
            charge=int(args.charge),
            spin=int(args.spin),
            verbose=int(args.verbose),
        )

        ncore = _infer_ncore(mol.nelectron, nelecas, spin=int(args.spin))
        cas = CASSpec(ncas=ncas, nelecas=nelecas, ncore=ncore)

        h1, eri8, ecore, ncas_eff, nelec_cas = build_cas_hamiltonian_pyscf(
            mol,
            cas=cas,
            orbital_method=str(args.orbital_method),
            enforce_ss=0.0,
            spin_shift=float(args.spin_shift),
            casscf_nroots=int(args.casscf_nroots),
            state_average=bool(args.state_average),
            x2c1e=bool(args.x2c1e),
        )
        if str(args.orbital_method).lower().startswith("casscf"):
            print(f"R={R:6.3f} Å | CASSCF done; running final CAS-FCI singlet energy...")
        if ncas_eff != ncas:
            raise RuntimeError(f"Unexpected ncas from builder: got {ncas_eff}, expected {ncas}")

        e_fci_cas, ss, mult = _fci_singlet_energy_and_spin(
            h1=h1,
            eri8=eri8,
            ncas=ncas_eff,
            nelec_cas=nelec_cas,
            spin_penalty_shift=float(args.spin_shift),
        )
        e_tot = float(ecore) + float(e_fci_cas)

        rows.append((float(R), float(e_tot), float(ecore), float(e_fci_cas), float(ss), float(mult)))
        print(f"R={R:6.3f} Å | E_total={e_tot:+.10f} Eh | <S^2>={ss:.6f} | mult={mult:.3f} | ncore={ncore}")

    data = np.asarray(rows, dtype=float)
    header = "R_Ang,E_total_Eh,E_core_Eh,E_fci_cas_Eh,S2,multiplicity"
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")
    print(f"\nWrote: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        # Plot relative energy (binding curve style): E(R) - min(E)
        rel = (data[:, 1] - np.min(data[:, 1])) * 1000.0  # mEh
        plt.plot(data[:, 0], rel, marker="o")
        plt.xlabel("R (Angstrom)")
        plt.ylabel("E_total - E_min (mEh)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        print(f"Wrote: {plot_path}")
    except ModuleNotFoundError as e:
        if ("matplotlib" in str(e)) or (getattr(e, "name", "") == "matplotlib"):
            print("\nPlot skipped: matplotlib not installed. Install with: pip install 'ccik[plot]'")
        else:
            raise


if __name__ == "__main__":
    main()
