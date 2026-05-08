"""Run the QP-backed N2 CAS(12e,12o) convergence case and save plot/CSV/log."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FILE = Path(__file__).resolve()
CASE_DIR = FILE.parent
ROOT = FILE.parents[4]
for import_path in (ROOT / "scripts" / "_shared", ROOT / "src"):
    path_text = str(import_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

os.environ.setdefault("MPLCONFIGDIR", str(CASE_DIR / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


@dataclass(frozen=True)
class SystemSpec:
    distance_ang: float
    basis: str
    ncore: int
    ncas: int
    nmo_total: int
    nelecas: int

    @property
    def atom(self) -> str:
        return f"N 0 0 0; N 0 0 {self.distance_ang}"

    @property
    def xyz(self) -> str:
        return "\n".join(
            (
                "2",
                f"N2 R={self.distance_ang} Ang",
                "N  0.0  0.0  0.0",
                f"N  0.0  0.0  {self.distance_ang}",
                "",
            )
        )

    @property
    def case_tag(self) -> str:
        return f"n2_cas12_12_R{self.distance_ang:.2f}".replace(".", "p")


@dataclass(frozen=True)
class RunSpec:
    fixed_store_keep: int
    fixed_propagate_keep: int
    m_cycle: int
    n_cycles: int
    nroot: int
    conv_tol: float
    selector_mode: str = "abs_residual"
    matvec_backend: str = "connected_sd_fullq"
    ritz_overlap_tol: float = 1.0e-8
    ritz_max_drop: float = 0.0
    save_final_wf: bool = False
    use_existing_wf: bool = False
    do_post_pt2: bool = False
    print_iteration_error: bool = True
    verbose: bool = False

    @property
    def external_keep(self) -> int:
        return max(2_048, self.fixed_store_keep)

    def make_ccik_params(self, reference_total_energy: float) -> CCIKParams:
        budget_kwargs = fixed_budget_bridge_kwargs(
            fixed_store_keep=self.fixed_store_keep,
            fixed_propagate_keep=self.fixed_propagate_keep,
            external_keep=self.external_keep,
        )
        return CCIKParams(
            **budget_kwargs,
            ccik_basis_size=self.m_cycle,
            ccik_selector_mode=self.selector_mode,
            ccik_matvec_backend=self.matvec_backend,
            ccik_conv_tol=self.conv_tol,
            ccik_ritz_overlap_tol=self.ritz_overlap_tol,
            ccik_ritz_max_drop=self.ritz_max_drop,
            ccik_max_cycle=self.n_cycles,
            ccik_restart_m_cycle=self.m_cycle,
            ccik_restart_nroot=self.nroot,
            ccik_save_final_wf=self.save_final_wf,
            ccik_use_existing_wf=self.use_existing_wf,
            ccik_do_post_pt2=self.do_post_pt2,
            ccik_verbose=self.verbose,
            ccik_print_iteration_error=self.print_iteration_error,
            ccik_reference_total_energy=reference_total_energy,
        )


# Edit this section when you want to change the case or solver settings.
SYSTEM = SystemSpec(
    distance_ang=2.00,
    basis="6-31g",
    ncore=2,
    ncas=10,
    nmo_total=18,
    nelecas=10,
)
RUN = RunSpec(
    fixed_store_keep=50_000,
    fixed_propagate_keep=1_00,
    m_cycle=25,
    n_cycles=10,
    nroot=3,
    conv_tol=1.0e-6,
)

OUTPUT_STEM = f"{SYSTEM.case_tag}_qp_convergence"
OUT_PLOT = CASE_DIR / f"{OUTPUT_STEM}.png"
OUT_CSV = CASE_DIR / f"{OUTPUT_STEM}.csv"
OUT_LOG = CASE_DIR / f"{OUTPUT_STEM}.log"


def compute_fci_reference() -> float:
    mol = make_mol_pyscf(atom=SYSTEM.atom, basis=SYSTEM.basis, spin=0, charge=0)
    cas = CASSpec(ncas=SYSTEM.ncas, nelecas=SYSTEM.nelecas, ncore=SYSTEM.ncore)
    h1eff, eri8, ecore, ncas, nelec_cas = build_cas_hamiltonian_pyscf(mol, cas=cas)
    e_fci_elec = exact_cas_fci_energy_pyscf(h1eff, eri8, ncas, nelec_cas)
    return float(ecore + e_fci_elec)


def _run_ccik_direct(qp, ezfio: Path, log: Path) -> str:
    import shlex as _shlex

    rc = _shlex.quote(str(qp.quantum_package_rc))
    ez = _shlex.quote(str(ezfio))
    cb = _shlex.quote(str(qp.ccik_binary))
    lg = _shlex.quote(str(log))
    cmd = f"set -o pipefail && ( source {rc} && qp set_file {ez} && {cb} {ez} ) 2>&1 | tee {lg}"
    result = subprocess.run(["/bin/bash", "-lc", cmd], text=True)
    if not log.exists():
        raise RuntimeError("ccik produced no log file; check your QP setup.")
    stdout = log.read_text(encoding="utf-8", errors="ignore")
    if result.returncode != 0 and "CCIK_FINAL_TOTAL_ENERGY" not in stdout:
        tail = "\n".join(stdout.splitlines()[-30:])
        raise RuntimeError(
            f"ccik exited with code {result.returncode}; see {log}\n\n"
            f"Last log lines:\n{tail}"
        )
    return stdout


def parse_final_result(stdout: str) -> tuple[float, int]:
    m_tot = re.search(r"CCIK_FINAL_TOTAL_ENERGY\s*=\s*([\-+0-9.eE]+)", stdout)
    m_bas = re.search(r"CCIK_FINAL_BASIS\s*=\s*(\d+)", stdout)
    if not (m_tot and m_bas):
        if "ac_operator: iorb > mo_num" in stdout:
            raise RuntimeError(
                "The QP CCIK backend crashed before printing final energies.\n"
                "QP reported `ac_operator: iorb > mo_num`, i.e. it tried to access "
                "orbital 19 while `mo_num = 18`.\n"
                "This is a backend/runtime failure for the chosen run state, not a "
                "Python parsing issue.\n"
                f"See {OUT_LOG} for the full log."
            )
        tail = "\n".join(stdout.splitlines()[-30:])
        raise RuntimeError(
            "Could not find final QP CCIK energies in stdout.\n\n"
            f"Last log lines:\n{tail}"
        )
    return float(m_tot.group(1)), int(m_bas.group(1))


def run_qp_ccik(qp, e_fci_total: float) -> tuple[float, int, str]:
    params = RUN.make_ccik_params(e_fci_total)

    with tempfile.TemporaryDirectory(prefix=f"{SYSTEM.case_tag.lower()}_qp_") as workdir:
        workdir_path = Path(workdir)
        xyz_path = workdir_path / "n2.xyz"
        ezfio_path = workdir_path / "n2.ezfio"
        xyz_path.write_text(SYSTEM.xyz, encoding="ascii")

        run_qp_setup(
            qp,
            basis=SYSTEM.basis,
            ezfio=ezfio_path,
            xyz=xyz_path,
            ncore=SYSTEM.ncore,
            ncas=SYSTEM.ncas,
            nmo_total=SYSTEM.nmo_total,
        )
        set_ccik_params(qp, ezfio_path, params)

        stdout = _run_ccik_direct(qp, ezfio_path, workdir_path / "ccik.log")
        OUT_LOG.write_text(stdout, encoding="utf-8")

    e_total, basis_final = parse_final_result(stdout)
    return e_total, basis_final, stdout


_RE_ITER = re.compile(
    r"ccik\(thick\) cycle=\s*(\d+)\s+iter=\s*(\d+)\s+total=\s*([-+0-9.Ee]+)"
)


@dataclass(frozen=True)
class IterPoint:
    cycle: int
    local_iter: int
    total_energy: float


def parse_iteration_data(stdout: str) -> list[IterPoint]:
    return [
        IterPoint(
            cycle=int(match.group(1)),
            local_iter=int(match.group(2)),
            total_energy=float(match.group(3)),
        )
        for match in _RE_ITER.finditer(stdout)
    ]


def assign_cumulative_dims(points: list[IterPoint]) -> list[int]:
    if not points:
        return []

    max_iter_per_cycle: dict[int, int] = {}
    for pt in points:
        max_iter_per_cycle[pt.cycle] = max(max_iter_per_cycle.get(pt.cycle, 0), pt.local_iter)

    cum_offset: dict[int, int] = {}
    running_total = 0
    for index, cycle in enumerate(sorted(max_iter_per_cycle)):
        cum_offset[cycle] = 0 if index == 0 else running_total - RUN.nroot
        running_total = cum_offset[cycle] + max_iter_per_cycle[cycle]

    return [cum_offset[pt.cycle] + pt.local_iter for pt in points]


def write_iteration_csv(
    *,
    iter_points: list[IterPoint],
    cum_dims: list[int],
    e_fci_total: float,
) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["cycle", "local_iter", "cum_krylov", "total_energy_eh", "abs_error_mEh"],
        )
        writer.writeheader()
        for pt, cum_dim in zip(iter_points, cum_dims):
            writer.writerow(
                {
                    "cycle": pt.cycle,
                    "local_iter": pt.local_iter,
                    "cum_krylov": cum_dim,
                    "total_energy_eh": f"{pt.total_energy:.12f}",
                    "abs_error_mEh": f"{abs(pt.total_energy - e_fci_total) * 1.0e3:.8f}",
                }
            )


def make_plot(
    *,
    iter_points: list[IterPoint],
    cum_dims: list[int],
    e_fci_total: float,
    e_ccik_final: float,
) -> None:
    energies = [pt.total_energy for pt in iter_points]
    delta_mEh = [abs(e - e_fci_total) * 1.0e3 for e in energies]
    delta_final_mEh = abs(e_ccik_final - e_fci_total) * 1.0e3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    fig.suptitle(
        f"N2 CAS(12e,12o) / {SYSTEM.basis.upper()} / R = {SYSTEM.distance_ang:.2f} Angstrom\n"
        f"QP CCIK thick-restart | Kstore={RUN.fixed_store_keep:,} | Kprop={RUN.fixed_propagate_keep}",
        fontsize=11,
    )

    if iter_points:
        ax1.plot(cum_dims, energies, "o-", color="steelblue", ms=4, lw=1.5, label="CCIK total energy")
    ax1.axhline(e_fci_total, color="crimson", ls="--", lw=1.2, label="Exact CAS-FCI")
    if iter_points:
        ax1.scatter([cum_dims[-1]], [e_ccik_final], color="darkorange", s=60, zorder=5, label="CCIK final")
    ax1.set_ylabel("Total energy (Eh)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    positive = [(x, y) for x, y in zip(cum_dims, delta_mEh) if y > 0.0]
    if positive:
        xs, ys = zip(*positive)
        ax2.semilogy(xs, ys, "o-", color="steelblue", ms=4, lw=1.5, label=r"$|E_{\rm CCIK}-E_{\rm FCI}|$")
    ax2.axhline(delta_final_mEh, color="darkorange", ls="-", lw=1.2, label=f"Final: {delta_final_mEh:.4f} mEh")
    ax2.axhspan(0.0, 1.0, alpha=0.08, color="green", label="1 mEh chemical accuracy")
    ax2.set_xlabel("Cumulative Krylov dimension")
    ax2.set_ylabel(r"$|\Delta E|$ (mEh)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=160)
    plt.close(fig)


def main() -> None:
    print("Locating Quantum Package installation...", flush=True)
    qp = discover_qp_paths()
    if not qp.ccik_binary.exists():
        raise SystemExit(
            "Could not locate the installed QP CCIK binary.\n"
            f"Resolved path: {qp.ccik_binary}\n"
            "Source quantum_package.rc and ensure the plugin is installed."
        )

    print("[1/3] Computing exact CAS-FCI reference...", flush=True)
    e_fci_total = compute_fci_reference()
    print(f"  E_FCI(total) = {e_fci_total:.10f} Eh", flush=True)

    print("[2/3] Running QP CCIK backend...", flush=True)
    e_ccik_final, basis_final, stdout = run_qp_ccik(qp, e_fci_total)
    print(f"  E_CCIK(final) = {e_ccik_final:.10f} Eh  | basis = {basis_final}", flush=True)

    print("[3/3] Parsing iterations and plotting...", flush=True)
    iter_points = parse_iteration_data(stdout)
    if not iter_points:
        raise SystemExit("No QP per-iteration data found in stdout; nothing to plot.")

    cum_dims = assign_cumulative_dims(iter_points)
    write_iteration_csv(iter_points=iter_points, cum_dims=cum_dims, e_fci_total=e_fci_total)
    make_plot(
        iter_points=iter_points,
        cum_dims=cum_dims,
        e_fci_total=e_fci_total,
        e_ccik_final=e_ccik_final,
    )

    print(f"Saved plot: {OUT_PLOT}", flush=True)
    print(f"Saved CSV : {OUT_CSV}", flush=True)
    print(f"Saved log : {OUT_LOG}", flush=True)


if __name__ == "__main__":
    main()
