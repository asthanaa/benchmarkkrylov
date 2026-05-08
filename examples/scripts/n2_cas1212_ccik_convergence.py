"""N2 CAS(12e,12o) CCIK thick-restart convergence study.

Runs:
  1. Exact CAS-FCI (PySCF Davidson) as the reliable reference.
  2. ccik_thick with production fixed-budget settings, collecting the
     Ritz energy after every new Krylov vector via iteration_callback.

Produces two plots:
  - Top panel : CCIK energy vs cumulative Krylov dimension
                with a dashed line at the exact CAS-FCI reference.
  - Bottom panel: |E_CCIK - E_FCI| (mEh) vs cumulative Krylov dimension
                  on a log scale, with a shaded chemical-accuracy band (1 mEh).

Setup (from project root):
    pip install -e ".[quantum,plot,test]"
    python examples/scripts/n2_cas1212_ccik_convergence.py

Runtime note:
  CAS(12e,12o) has ~854 000 determinants.  The exact FCI takes a few minutes
  on a modern laptop; each CCIK Krylov step is much cheaper.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the package is importable whether or not it is pip-installed.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import time

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; remove if you want a pop-up
import matplotlib.pyplot as plt
import numpy as np

from ccik import (
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense_thick_restart,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKThickRestartParams
from ccik.pyscf_cas import CASSpec
from ccik.thick_restart import ThickRestartIterationRecord


# ---------------------------------------------------------------------------
# Molecule & active space
# ---------------------------------------------------------------------------
#
# N2 at equilibrium bond length (1.10 Å).
# N2 has 14 electrons.  CAS(12e, 12o) freezes 1 core MO pair (ncore=1).
# Basis: 6-31g (matches all published benchmarks in this repository).
#
BOND_LENGTH_ANG = 1.10
BASIS = "6-31g"
CAS = CASSpec(ncas=12, nelecas=12, ncore=1)

# ---------------------------------------------------------------------------
# CCIK solver parameters
# ---------------------------------------------------------------------------
#
# These match the production settings documented in docs/quantum_package_backend.md:
#   fixed_store_keep  = 65 536  (K_store — maximum determinants kept per Krylov vector)
#   fixed_propagate_keep = 512  (K_prop  — determinants used to spawn next candidates)
#   selector_mode = "abs_residual"  (rank by |<I|H|q_k>|)
#   candidate_mode = "connected_sd" (Python equivalent of QP connected_sd_fullq)
#
# ncycles × m_cycle gives the total number of Krylov vectors attempted.
# nroot=3 means the 3 lowest Ritz vectors are recycled as restart seeds.
#
PARAMS = CCIKThickRestartParams(
    m_cycle=25,
    ncycles=8,          # up to 200 Krylov vectors total
    nroot=3,
    tol=1e-6,           # stop early if successive restart energies agree to 1e-6 Eh
    compression_policy="fixed_budget",
    fixed_store_keep=50_000,  # K_store
    fixed_propagate_keep=512, # K_prop
    candidate_mode="connected_sd",
    selector_mode="abs_residual",
    projected_ritz_mode="exact_on_basis",
    tmp_keep_factor=4,
    preserve_prev_support=True,
    orth_tol=1e-12,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def make_n2_atom(distance_ang: float) -> str:
    z = distance_ang / 2.0
    return f"N 0.0 0.0 {-z:.6f}\nN 0.0 0.0 {+z:.6f}"


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  N2 CAS(12e,12o) CCIK convergence study")
    print(f"  Bond length : {BOND_LENGTH_ANG:.2f} Å  |  Basis: {BASIS}")
    print(f"  Active space: CAS({CAS.nelecas}e, {CAS.ncas}o)  |  ncore = {CAS.ncore}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1.  Build the active-space Hamiltonian
    # ------------------------------------------------------------------
    mol = make_mol_pyscf(
        atom=make_n2_atom(BOND_LENGTH_ANG),
        basis=BASIS,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    print("\nBuilding CAS Hamiltonian (CASSCF MOs) ...", flush=True)
    t0 = time.perf_counter()
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=CAS)
    print(f"  Done in {time.perf_counter() - t0:.1f} s  |  "
          f"norb={ncas}, nelec={nelec}, ecore={ecore:.6f} Eh")

    # ------------------------------------------------------------------
    # 2.  Exact CAS-FCI reference (PySCF Davidson)
    #     This is the most reliable reference available on the Python path.
    #     CI space size: C(12,6)^2 ≈ 854 000 determinants — large but feasible.
    # ------------------------------------------------------------------
    print("\nRunning exact CAS-FCI (PySCF Davidson) ...", flush=True)
    t0 = time.perf_counter()
    e_fci_elec = exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec)
    e_fci_total = float(ecore + e_fci_elec)
    print(f"  Exact CAS-FCI total energy : {e_fci_total:+.10f} Eh "
          f"(elapsed {time.perf_counter() - t0:.1f} s)")

    # ------------------------------------------------------------------
    # 3.  CCIK thick-restart, collecting per-step Ritz energies
    #     via iteration_callback
    # ------------------------------------------------------------------
    #
    # ThickRestartIterationRecord fields used below:
    #   .basis_size          : current Krylov dimension (|Q|)
    #   .cum_new_krylov      : total new vectors added across all cycles so far
    #   .cycle               : restart cycle index
    #   .electronic_energy_eh: ground Ritz energy (electronic, no ecore)
    #
    records: list[ThickRestartIterationRecord] = []

    def _callback(rec: ThickRestartIterationRecord) -> None:
        records.append(rec)
        e_total = float(ecore + rec.electronic_energy_eh)
        delta_mEh = 1000.0 * (e_total - e_fci_total)
        print(
            f"  cycle={rec.cycle:2d}  |Q|={rec.basis_size:4d}  "
            f"E={e_total:+.10f} Eh  ΔE={delta_mEh:+.4f} mEh",
            flush=True,
        )

    print(f"\nRunning ccik_thick  "
          f"(m_cycle={PARAMS.m_cycle}, ncycles={PARAMS.ncycles}, "
          f"K_store={PARAMS.fixed_store_keep}, K_prop={PARAMS.fixed_propagate_keep}) ...\n",
          flush=True)
    t0 = time.perf_counter()
    e_ccik_elec = ccik_ground_energy_dense_thick_restart(
        h1, eri8, ncas, nelec,
        params=PARAMS,
        iteration_callback=_callback,
    )
    elapsed_ccik = time.perf_counter() - t0
    e_ccik_total = float(ecore + e_ccik_elec)
    delta_final_mEh = 1000.0 * (e_ccik_total - e_fci_total)

    print(f"\n{'=' * 60}")
    print(f"  CCIK final total energy    : {e_ccik_total:+.10f} Eh  (elapsed {elapsed_ccik:.1f} s)")
    print(f"  Exact CAS-FCI total energy : {e_fci_total:+.10f} Eh")
    print(f"  |ΔE| (final)               : {abs(delta_final_mEh):.4f} mEh")
    print(f"  Chemical accuracy (< 1 mEh): {abs(delta_final_mEh) < 1.0}")
    print("=" * 60)

    if not records:
        print("\nNo iteration records collected — nothing to plot.")
        return

    # ------------------------------------------------------------------
    # 4.  Plots
    # ------------------------------------------------------------------
    krylov_dims = np.array([r.basis_size for r in records], dtype=float)
    e_ccik_steps = np.array([float(ecore + r.electronic_energy_eh) for r in records])
    delta_mEh_steps = 1000.0 * (e_ccik_steps - e_fci_total)

    # Identify restart boundaries (where basis_size resets to a small value)
    restart_x: list[float] = []
    for i in range(1, len(krylov_dims)):
        if krylov_dims[i] < krylov_dims[i - 1]:
            restart_x.append(float(krylov_dims[i - 1]))

    # Use cum_new_krylov as the x-axis so restarts appear as a monotone axis
    cum_krylov = np.array([r.cum_new_krylov for r in records], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(
        f"N2 CAS({CAS.nelecas}e,{CAS.ncas}o)  |  {BASIS} basis  |  "
        f"$R = {BOND_LENGTH_ANG:.2f}$ Å\n"
        f"ccik_thick  $K_{{\\mathrm{{store}}}}={PARAMS.fixed_store_keep}$  "
        f"$K_{{\\mathrm{{prop}}}}={PARAMS.fixed_propagate_keep}$  "
        f"selector: {PARAMS.selector_mode}",
        fontsize=11,
    )

    # --- Panel 1: energy convergence ---
    ax0 = axes[0]
    ax0.plot(cum_krylov, e_ccik_steps, "o-", ms=4, lw=1.5,
             color="#1f77b4", label="CCIK thick-restart")
    ax0.axhline(e_fci_total, color="#d62728", lw=1.5, ls="--",
                label=f"Exact CAS-FCI  ({e_fci_total:+.8f} Eh)")
    for rx in restart_x:
        # find cum_krylov position at the restart
        idx = int(np.argmax(krylov_dims == rx))
        ax0.axvline(cum_krylov[idx], color="gray", lw=0.8, ls=":", alpha=0.6)
    ax0.set_ylabel("Total energy / Eh", fontsize=11)
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.yaxis.set_major_formatter(plt.FormatStrFormatter("%.6f"))
    ax0.set_title("Energy convergence", fontsize=10)

    # --- Panel 2: |ΔE| on log scale ---
    ax1 = axes[1]
    abs_delta = np.abs(delta_mEh_steps)
    # Clip zeros to avoid log(0)
    abs_delta_plot = np.where(abs_delta > 0, abs_delta, np.nan)
    ax1.semilogy(cum_krylov, abs_delta_plot, "s-", ms=4, lw=1.5,
                 color="#ff7f0e", label=r"$|E_{\mathrm{CCIK}} - E_{\mathrm{FCI}}|$")
    ax1.axhspan(0, 1.0, color="#2ca02c", alpha=0.12,
                label="Chemical accuracy (< 1 mEh)")
    ax1.axhline(1.0, color="#2ca02c", lw=1.2, ls="--")
    for rx in restart_x:
        idx = int(np.argmax(krylov_dims == rx))
        ax1.axvline(cum_krylov[idx], color="gray", lw=0.8, ls=":", alpha=0.6,
                    label="_nolegend_")
    ax1.set_ylabel(r"$|E_{\mathrm{CCIK}} - E_{\mathrm{FCI}}|$ / mEh", fontsize=11)
    ax1.set_xlabel("Cumulative Krylov vectors added", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("Error relative to exact CAS-FCI", fontsize=10)

    # Annotate final error
    ax1.annotate(
        f"Final: {abs(delta_final_mEh):.3f} mEh",
        xy=(cum_krylov[-1], abs(delta_final_mEh)),
        xytext=(cum_krylov[-1] * 0.75, abs(delta_final_mEh) * 5),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    plt.tight_layout()
    out_path = ROOT / "n2_cas1212_ccik_convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
