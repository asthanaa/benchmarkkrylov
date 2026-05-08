"""N2 CAS(12e,12o) CCIK thick-restart convergence — Quantum Package (Fortran) backend.

Runs:
  1. Exact CAS-FCI (PySCF Davidson) as a reliable reference energy.
  2. QP Fortran CCIK solver (thick-restart, fixed-budget) for the same active space.
  3. Parses the QP stdout to extract the Ritz energy after every new Krylov
     vector so that the full convergence curve can be plotted.

Produces two plots (saved to the project root):
  - Top panel   : CCIK total energy vs cumulative Krylov dimension,
                  with a dashed line at the exact CAS-FCI reference.
  - Bottom panel : |E_CCIK − E_FCI| (mEh) vs cumulative Krylov dimension
                   on a log scale, with a 1 mEh chemical-accuracy band.

Prerequisites:
  1. Build the ccik plugin and install it into your QP installation::

       cd $QP_ROOT
       qp_plugins install /path/to/CompressK/ccik_qp/stable/ccik
       ninja

  2. Source the QP environment::

       source $QP_ROOT/quantum_package.rc

  3. Run from the project root::

       python examples/scripts/n2_cas1212_ccik_convergence_qp.py

Runtime note:
  The QP Fortran backend is ~10–50× faster than the Python solver for this
  active-space size.  Expect a few minutes total (HF setup + CCIK).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: make ccik (src/) and _shared (scripts/) importable.
# ---------------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parents[2]
SHARED = ROOT / "scripts" / "_shared"
SRC    = ROOT / "src"
for _p in (str(SHARED), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qp_ccik_common import (  # noqa: E402
    CCIKParams,
    discover_qp_paths,
    fixed_budget_bridge_kwargs,
    run_qp_setup,
    set_ccik_params,
)
from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.pyscf_cas import CASSpec  # noqa: E402

# ---------------------------------------------------------------------------
# System definition — N2, equilibrium geometry
# ---------------------------------------------------------------------------
N2_DIST_ANG  = 1.10     # Å
N2_BASIS     = "6-31g"
N2_NCORE     = 1        # freeze 1 core MO pair (2 electrons)
N2_NCAS      = 12       # active orbitals
N2_NMO_TOTAL = 18       # 6-31g gives 18 canonical MOs for N2
N2_NELECAS   = 12       # active electrons  (14 total − 2 frozen)

N2_XYZ = f"""\
2
N2 R={N2_DIST_ANG} Ang
N  0.0  0.0  0.0
N  0.0  0.0  {N2_DIST_ANG}
"""

# ---------------------------------------------------------------------------
# CCIK hyper-parameters (mirrors the Python-backend script)
# ---------------------------------------------------------------------------
FIXED_STORE_KEEP     = 65_536   # K_store: max determinants per Krylov vector
FIXED_PROPAGATE_KEEP = 512      # K_prop : parents for excitation generation
M_CYCLE       = 25              # Krylov vectors per thick-restart cycle
N_CYCLES      = 8               # max thick-restart cycles
NROOT         = 3               # Ritz vectors kept at each restart
CONV_TOL      = 1e-6            # |ΔE| convergence threshold (Eh)
EXTERNAL_KEEP = 2_048           # external-space buffer size

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_PLOT = ROOT / "n2_cas1212_ccik_convergence_qp.png"


# ---------------------------------------------------------------------------
# Step 1 — FCI reference via PySCF
# ---------------------------------------------------------------------------
def compute_fci_reference() -> tuple[float, float]:
    """Return (e_fci_total, ecore) using PySCF Davidson exact FCI.

    Total FCI energy = ecore (frozen-core + nuclear-repulsion term) +
                       CAS electronic energy.
    """
    from pyscf import gto, scf  # noqa: F401 (imported inside to keep top-level clean)

    print("  Building PySCF molecule …", flush=True)
    mol = make_mol_pyscf(
        atom=f"N 0 0 0; N 0 0 {N2_DIST_ANG}",
        basis=N2_BASIS,
        spin=0,
        charge=0,
    )
    cas = CASSpec(ncas=N2_NCAS, nelecas=N2_NELECAS, ncore=N2_NCORE)
    print("  Running PySCF RHF + CAS Hamiltonian construction …", flush=True)
    h1eff, eri8, ecore, ncas, nelec_cas = build_cas_hamiltonian_pyscf(mol, cas=cas)
    print("  Running exact CAS-FCI (Davidson) …", flush=True)
    e_fci_elec = exact_cas_fci_energy_pyscf(h1eff, eri8, ncas, nelec_cas)
    e_fci_total = ecore + e_fci_elec
    return e_fci_total, ecore


# ---------------------------------------------------------------------------
# Step 2 — QP CCIK run
# ---------------------------------------------------------------------------
def _run_ccik_direct(qp, ezfio: Path, log: Path) -> str:
    """Run the ccik binary, writing output to `log` via tee.

    Uses shell redirection instead of capture_output=True so that QP's ZMQ
    layer gets a real stdout (not a pipe), which avoids exit-255 failures.
    """
    import shlex as _shlex
    rc  = _shlex.quote(str(qp.quantum_package_rc))
    ez  = _shlex.quote(str(ezfio))
    cb  = _shlex.quote(str(qp.ccik_binary))
    lg  = _shlex.quote(str(log))
    # Wrap in parens so ALL output (including qp set_file errors) goes to tee.
    # Source rc explicitly in case QP env is not inherited by this subshell.
    # Do NOT use -l (login shell) — that re-sources ~/.bash_profile and can
    # reset PATH, breaking the inherited QP environment.
    cmd = f"( source {rc} && qp set_file {ez} && {cb} {ez} ) 2>&1 | tee {lg}"
    result = subprocess.run(["/bin/bash", "-c", cmd], text=True)
    if not log.exists():
        raise RuntimeError("ccik produced no log file — check QP setup.")
    stdout = log.read_text()
    if result.returncode != 0 and "CCIK_FINAL_TOTAL_ENERGY" not in stdout:
        raise RuntimeError(
            f"ccik exited with code {result.returncode}. "
            f"Check log: {log}"
        )
    return stdout


def build_ccik_params(e_fci_total: float) -> CCIKParams:
    """Construct a CCIKParams with production fixed-budget settings.

    ccik_verbose=True         → per-iteration Ritz energies are printed.
    ccik_print_iteration_error=True + ccik_reference_total_energy=e_fci_total
                              → QP prints |ΔE| vs FCI at every iteration.
    """
    budget_kwargs = fixed_budget_bridge_kwargs(
        fixed_store_keep=FIXED_STORE_KEEP,
        fixed_propagate_keep=FIXED_PROPAGATE_KEEP,
        external_keep=EXTERNAL_KEEP,
    )
    return CCIKParams(
        **budget_kwargs,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_conv_tol=CONV_TOL,
        ccik_max_cycle=N_CYCLES,
        ccik_restart_m_cycle=M_CYCLE,
        ccik_restart_nroot=NROOT,
        ccik_save_final_wf=True,
        ccik_verbose=True,
        ccik_print_iteration_error=True,
        ccik_reference_total_energy=e_fci_total,
    )


def run_qp_ccik(qp, e_fci_total: float) -> tuple[float, int, str]:
    """Set up EZFIO, run CCIK, return (e_tot_final, basis_final, stdout)."""
    params = build_ccik_params(e_fci_total)

    with tempfile.TemporaryDirectory(prefix="n2_cas1212_ccik_qp_") as workdir:
        workdir_path = Path(workdir)
        xyz_path  = workdir_path / "n2.xyz"
        ezfio_path = workdir_path / "n2.ezfio"
        xyz_path.write_text(N2_XYZ)

        print("  Creating EZFIO database and running RHF/SCF …", flush=True)
        run_qp_setup(
            qp,
            basis=N2_BASIS,
            ezfio=ezfio_path,
            xyz=xyz_path,
            ncore=N2_NCORE,
            ncas=N2_NCAS,
            nmo_total=N2_NMO_TOTAL,
        )
        print("  EZFIO ready. Writing CCIK parameters …", flush=True)
        set_ccik_params(qp, ezfio_path, params)

        print(
            f"  Running QP CCIK  "
            f"(m_cycle={M_CYCLE}, n_cycles={N_CYCLES}, "
            f"K_store={FIXED_STORE_KEEP:,}, K_prop={FIXED_PROPAGATE_KEEP}) …",
            flush=True,
        )
        log_path = workdir_path / "ccik.log"
        stdout = _run_ccik_direct(qp, ezfio_path, log_path)
        # Keep a persistent copy for post-mortem inspection
        import shutil as _shutil
        _shutil.copy2(log_path, Path("/tmp/n2_ccik_last_run.log"))
        print("  Log saved to /tmp/n2_ccik_last_run.log", flush=True)
        # Parse final energies from the log
        m_elec = re.search(r"CCIK_FINAL_ELECTRONIC_ENERGY\s*=\s*([\-+0-9.eE]+)", stdout)
        m_tot  = re.search(r"CCIK_FINAL_TOTAL_ENERGY\s*=\s*([\-+0-9.eE]+)", stdout)
        m_bas  = re.search(r"CCIK_FINAL_BASIS\s*=\s*(\d+)", stdout)
        if not (m_tot and m_bas):
            print("--- CCIK LOG (last 4000 chars) ---", flush=True)
            print(stdout[-4000:] if len(stdout) > 4000 else stdout, flush=True)
            print("--- END LOG ---", flush=True)
            raise RuntimeError("Could not find CCIK_FINAL_TOTAL_ENERGY in output.")
        e_tot       = float(m_tot.group(1))
        basis_final = int(m_bas.group(1))
        print(f"  CCIK finished.  E_total = {e_tot:.10f} Eh  (basis = {basis_final})", flush=True)

    return e_tot, basis_final, stdout


# ---------------------------------------------------------------------------
# Step 3 — Parse QP stdout for per-iteration convergence
# ---------------------------------------------------------------------------
# The Fortran driver prints (when ccik_print_iteration_error=True):
#   ccik(thick) cycle= N  iter= M  total= -109.XXXX  ref= ...  delta_mEh= ...
_RE_ITER = re.compile(
    r"ccik\(thick\) cycle=\s*(\d+)\s+iter=\s*(\d+)\s+total=\s*([-+0-9.Ee]+)"
)


@dataclass
class IterPoint:
    cycle: int
    local_iter: int        # Krylov index within the current cycle (1-based)
    total_energy: float    # total energy in Eh


def parse_iteration_data(stdout: str) -> list[IterPoint]:
    """Extract per-iteration total energies from QP verbose stdout."""
    points: list[IterPoint] = []
    for m in _RE_ITER.finditer(stdout):
        points.append(
            IterPoint(
                cycle=int(m.group(1)),
                local_iter=int(m.group(2)),
                total_energy=float(m.group(3)),
            )
        )
    return points


def assign_cumulative_dims(points: list[IterPoint]) -> list[int]:
    """Convert per-cycle local Krylov indices to a cumulative dimension counter.

    Cycle 1 starts from 1 seed (HF reference) and expands to m_cycle.
    Subsequent cycles restart from `nroot` Ritz vectors and expand to m_cycle.
    The local `iter` field in QP output equals the current n_basis within the
    cycle, so new vectors added in cycle c > 1 = (m_cycle - nroot).
    """
    if not points:
        return []

    # Find the maximum local_iter reached in each cycle (= full basis size at end of that cycle).
    max_iter_per_cycle: dict[int, int] = {}
    for pt in points:
        max_iter_per_cycle[pt.cycle] = max(max_iter_per_cycle.get(pt.cycle, 0), pt.local_iter)

    sorted_cycles = sorted(max_iter_per_cycle)
    cum_offset: dict[int, int] = {}
    running_total = 0
    for idx, cyc in enumerate(sorted_cycles):
        if idx == 0:
            # First cycle: no offset; iter already starts at 1 (the seed).
            cum_offset[cyc] = 0
        else:
            # Restart seeds (nroot vectors) were already counted at end of previous cycle;
            # subtract them to avoid double-counting.
            cum_offset[cyc] = running_total - NROOT
        running_total = cum_offset[cyc] + max_iter_per_cycle[cyc]

    return [cum_offset[pt.cycle] + pt.local_iter for pt in points]


# ---------------------------------------------------------------------------
# Step 4 — Plot
# ---------------------------------------------------------------------------
def make_plot(
    iter_points: list[IterPoint],
    cum_dims: list[int],
    e_fci: float,
    e_ccik_final: float,
) -> None:
    energies  = [pt.total_energy for pt in iter_points]
    delta_mEh = [abs(e - e_fci) * 1e3 for e in energies]
    delta_final_mEh = abs(e_ccik_final - e_fci) * 1e3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=bool(iter_points))
    fig.suptitle(
        f"N$_2$ CAS(12e,12o) / 6-31G / $R$ = {N2_DIST_ANG} Å\n"
        f"CCIK thick-restart  (QP Fortran backend,  "
        f"$K_{{\\rm store}}$ = {FIXED_STORE_KEEP:,},  "
        f"$K_{{\\rm prop}}$ = {FIXED_PROPAGATE_KEEP})",
        fontsize=11,
    )

    # --- panel 1: energy vs cumulative Krylov dimension ---------------------
    if iter_points:
        ax1.plot(cum_dims, energies, "o-", color="steelblue", ms=4, lw=1.5,
                 label="CCIK (per Krylov step)")
    ax1.axhline(e_fci, color="crimson", ls="--", lw=1.2, label="Exact CAS-FCI")
    if iter_points:
        ax1.scatter([cum_dims[-1]], [e_ccik_final], color="darkorange",
                    zorder=5, s=70, label=f"CCIK final ({e_ccik_final:.8f} Eh)")
    ax1.set_ylabel("Total energy (Eh)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: |ΔE| vs FCI on log scale ----------------------------------
    if iter_points:
        positive = [(c, d) for c, d in zip(cum_dims, delta_mEh) if d > 0]
        if positive:
            xs, ys = zip(*positive)
            ax2.semilogy(xs, ys, "o-", color="steelblue", ms=4, lw=1.5,
                         label="$|E_{\\rm CCIK} - E_{\\rm FCI}|$")
    ax2.axhline(delta_final_mEh, color="darkorange", ls="-", lw=1.2,
                label=f"CCIK final: {delta_final_mEh:.4f} mEh")
    ax2.axhspan(0, 1.0, alpha=0.08, color="green", label="1 mEh (chem. accuracy)")
    ax2.set_xlabel("Cumulative Krylov dimension")
    ax2.set_ylabel("$|\\Delta E|$  (mEh)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"\nPlot saved → {OUT_PLOT}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── 0. Check QP is available ───────────────────────────────────────────
    print("Locating Quantum Package installation …", flush=True)
    try:
        qp = discover_qp_paths()
    except RuntimeError as exc:
        sys.exit(
            f"ERROR: {exc}\n"
            "Source the QP environment first:  source $QP_ROOT/quantum_package.rc"
        )

    # The shared helper points to plugins/ccik_qp/stable/ccik/ccik (upstream path),
    # but when installed via `qp_plugins install` the binary lands under plugins/local/.
    # Try the local path first, fall back to the upstream path.
    from dataclasses import replace as dc_replace
    local_ccik = qp.qp_root / "plugins" / "local" / "ccik" / "ccik"
    if local_ccik.exists() and not qp.ccik_binary.exists():
        qp = dc_replace(qp, ccik_binary=local_ccik)

    if not qp.ccik_binary.exists():
        sys.exit(
            f"ERROR: CCIK binary not found.\n"
            f"  Tried: {qp.ccik_binary}\n"
            f"  Also tried: {local_ccik}\n"
            "Build the plugin first:\n"
            "  source $QP_ROOT/quantum_package.rc\n"
            "  cp -r <CompressK>/ccik_qp/stable/ccik $QP_ROOT/plugins/local/ccik\n"
            "  qp_plugins install ccik\n"
            "  ninja -f $QP_ROOT/config/build.ninja module_ccik"
        )
    print(f"  QP root : {qp.qp_root}", flush=True)
    print(f"  ccik bin: {qp.ccik_binary}", flush=True)

    # ── 1. FCI reference ───────────────────────────────────────────────────
    print("\n[1/3] Computing exact CAS-FCI reference (PySCF) …", flush=True)
    e_fci_total, ecore = compute_fci_reference()
    print(f"  E_FCI (total) = {e_fci_total:.10f} Eh", flush=True)

    # ── 2. Run QP CCIK ─────────────────────────────────────────────────────
    print("\n[2/3] Running QP CCIK …", flush=True)
    e_ccik_final, basis_final, stdout = run_qp_ccik(qp, e_fci_total)

    # ── 3. Parse & plot ────────────────────────────────────────────────────
    print("\n[3/3] Parsing per-iteration data and plotting …", flush=True)
    iter_points = parse_iteration_data(stdout)
    if not iter_points:
        print(
            "  WARNING: No per-iteration data found in QP stdout.\n"
            "  The plot will show only the final CCIK result.\n"
            "  (This can happen if ccik_verbose or ccik_print_iteration_error\n"
            "   were not properly written to EZFIO.)",
            flush=True,
        )
    else:
        print(f"  Parsed {len(iter_points)} iteration points across "
              f"{max(pt.cycle for pt in iter_points)} restart cycle(s).", flush=True)

    cum_dims = assign_cumulative_dims(iter_points)
    make_plot(iter_points, cum_dims, e_fci_total, e_ccik_final)

    # ── Summary ────────────────────────────────────────────────────────────
    delta_mEh = abs(e_ccik_final - e_fci_total) * 1e3
    print(
        f"\n{'─'*55}\n"
        f"  E_FCI   = {e_fci_total:.10f} Eh\n"
        f"  E_CCIK  = {e_ccik_final:.10f} Eh   (basis = {basis_final})\n"
        f"  |ΔE|    = {delta_mEh:.6f} mEh\n"
        f"{'─'*55}",
        flush=True,
    )


if __name__ == "__main__":
    main()
