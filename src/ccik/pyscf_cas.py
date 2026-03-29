"""Build active-space Hamiltonians from PySCF objects.

This module is the bridge between chemistry input and the Krylov solvers:

- PySCF builds the molecular orbitals
- this module extracts the effective CAS Hamiltonian
- the solver modules consume the resulting ``h1``, ``eri8``, ``ecore``,
  orbital count, and active-space electron tuple

For most new readers, this is the easiest internal module to start with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# PDF mapping note
# ---------------
# `krylov_cipsi.pdf` describes the compressed Krylov/selection algorithm in terms of an abstract
# Hamiltonian H and determinant basis |D_I>.
#
# This file is *support code* to construct that Hamiltonian in a chosen active space using PySCF:
# it builds the effective CAS one-electron integrals (h1eff), two-electron integrals (eri8), and
# the additive constant ecore so that:
#   E_total = ecore + E_CAS_electronic
#
# The actual PDF equations (selection score, projection, orthogonalization, Ritz) are implemented
# in `core.py`.


def make_mol_pyscf(
    *,
    atom: str,
    basis: str,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    verbose: int = 0,
):
    """Create a PySCF Mole object."""
    from pyscf import gto

    return gto.M(
        atom=atom,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        verbose=verbose,
    )


@dataclass(frozen=True)
class CASSpec:
    """Describe the chosen complete active space.

    Plain-language purpose:
    - store the active-space size in one place

    Chemistry meaning:
    - ``ncas`` active orbitals
    - ``nelecas`` active electrons
    - ``ncore`` frozen doubly occupied core orbitals
    """

    ncas: int
    nelecas: int
    ncore: int = 0


def build_cas_hamiltonian_pyscf(
    mol,
    *,
    cas: CASSpec,
    orbital_method: str = "casci",
    enforce_ss: float | None = None,
    spin_shift: float = 20.0,
    casscf_nroots: int = 1,
    state_average: bool = False,
    x2c1e: bool = False,
):
    """Build the effective CAS Hamiltonian from a PySCF molecule.

    Plain-language purpose:
    - turn a molecule into the active-space problem that CCIK solves

    Chemistry meaning:
    - ``h1eff`` and ``eri8`` are the one- and two-electron Hamiltonian pieces
      inside the chosen active space
    - ``ecore`` stores the energy contribution from the frozen core and other
      inactive terms

    Inputs:
    - ``mol``: a PySCF molecule
    - ``cas``: the active-space choice
    - ``orbital_method``: whether to use CASCI or CASSCF orbitals

    Outputs:
    - ``(h1eff, eri8, ecore, ncas, nelec_cas)``
    """

    from pyscf import ao2mo, mcscf, scf

    mf = scf.RHF(mol)
    if x2c1e:
        # Scalar-relativistic X2C one-electron approximation.
        # PySCF exposes this via the SCF object.
        mf = mf.x2c1e()
    mf = mf.run()

    om = orbital_method.strip().lower().replace("-", "_")
    if om == "casci":
        mc = mcscf.CASCI(mf, cas.ncas, cas.nelecas)
        mc.ncore = cas.ncore
        mc.mo_coeff = mf.mo_coeff

        h1eff, ecore = mc.get_h1eff()
        eri_packed = mc.get_h2eff()
    elif om in ("casscf", "casscf_then_casci"):
        mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
        mc.ncore = cas.ncore
        mc.mo_coeff = mf.mo_coeff

        if state_average:
            nroots = max(int(casscf_nroots), 1)
            mc.fcisolver.nroots = nroots
            w = np.ones(nroots) / float(nroots)
            mc = mc.state_average_(w)

        if enforce_ss is not None:
            from pyscf.fci import addons

            mc.fcisolver = addons.fix_spin(mc.fcisolver, ss=float(enforce_ss), shift=float(spin_shift))

        mc.kernel()
        if om == "casscf_then_casci":
            # Build the CAS Hamiltonian in the optimized orbital basis.
            # This corresponds to: optimize orbitals via CASSCF, then do CASCI in those orbitals.
            mc2 = mcscf.CASCI(mf, cas.ncas, cas.nelecas)
            mc2.ncore = cas.ncore
            mc2.mo_coeff = mc.mo_coeff
            h1eff, ecore = mc2.get_h1eff()
            eri_packed = mc2.get_h2eff()
        else:
            h1eff, ecore = mc.get_h1eff()
            eri_packed = mc.get_h2eff()
    else:
        raise ValueError(
            f"Unknown orbital_method={orbital_method!r}. Expected 'casci', 'casscf', or 'casscf_then_casci'."
        )

    eri8 = ao2mo.restore(8, eri_packed, cas.ncas)

    neleca = cas.nelecas // 2
    nelecb = cas.nelecas - neleca
    nelec_cas = (neleca, nelecb)

    return np.asarray(h1eff), np.asarray(eri8), float(ecore), cas.ncas, nelec_cas


def exact_cas_fci_energy_pyscf(h1eff: np.ndarray, eri8: np.ndarray, ncas: int, nelec_cas: tuple[int, int]) -> float:
    """Compute the exact CAS-FCI electronic energy for a small reference problem.

    Plain-language purpose:
    - provide a trustworthy comparison target for the teaching examples

    Chemistry meaning:
    - this is the exact electronic energy inside the chosen active space

    Inputs:
    - active-space integrals and electron counts

    Outputs:
    - the electronic CAS-FCI energy in Hartree
    """

    from pyscf import fci

    cisolver = fci.direct_spin1.FCI()
    # Make the “exact” reference robust at stretched bonds / near-degeneracies.
    # Default tolerances can leave ~1e-5 Eh residual error, which may look like
    # slight non-variational behavior when comparing against other approximations.
    cisolver.conv_tol = 1e-12
    cisolver.max_cycle = 200
    cisolver.max_space = 50
    e_elec, _ = cisolver.kernel(h1eff, eri8, ncas, nelec_cas)
    return float(e_elec)
