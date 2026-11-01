"""Check (optional): run the unmodified production function QCANT.qulacs_accel.qrte_qulacs exactly as
scripts/Mushir_script/run_qrte_fock_pi_over_w.py does (cached Hamiltonian pinned, second-order Trotter, one step,
overlap_tol 1e-10, early stop at chemical accuracy) and compare its energy history with krylov_bases.py.

usage:  python check_against_qrte_qulacs.py <W|MF> case_id [case_id ...]   (needs work/<rule>/<case>.npz)
"""
import contextlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import paths

warnings.filterwarnings("ignore")
paths.add_qcant()
from QCANT import qulacs_accel as qa  # noqa: E402

rule = sys.argv[1]
rows = paths.fig5_rows()


@contextlib.contextmanager
def cached_hamiltonian(p):
    original = qa._build_hamiltonian_payload
    qa._build_hamiltonian_payload = lambda *a, **k: p
    try:
        yield
    finally:
        qa._build_hamiltonian_payload = original


for cid in sys.argv[2:]:
    t0 = time.time()
    p, summ = paths.payload(cid)
    case = json.load(open(paths.REPO / Path(rows[cid]["summary_path"]).parent / "input_snapshot.json"))
    dt = float(np.pi / json.load(open(paths.WIDTHS / f"{cid}.json"))["W"]) if rule == "W" else float(summ["delta_t"])
    ref = float(case["local_reference_energy_hartree"])
    with cached_hamiltonian(p):
        _e, _s, _t, hist = qa.qrte_qulacs(
            symbols=case["symbols"], geometry=np.asarray(case["geometry_angstrom"]), basis=case["basis"],
            charge=case["charge"], spin=case["spin"], active_electrons=case["active_electrons"],
            active_orbitals=case["active_orbitals"], delta_t=dt, n_steps=100, overlap_tol=1e-10,
            evolution_mode="trotter", trotter_steps=1, trotter_order=2, max_workers=None,
            early_stop_energy=ref + 1.6e-3, return_min_energy_history=True,
            hamiltonian_source=case["hamiltonian_source"], allow_unconverged_casscf=not case["local_reference_converged"])
    hist = np.asarray(hist, float)
    hit = next((i for i, e in enumerate(hist - ref, 1) if abs(e) <= 1.6e-3), None)
    uz = np.load(paths.npz_path(rule, cid))
    uh = np.asarray(uz["hist"]); m = min(len(uh), len(hist))
    print(f"{cid:26s} dt={dt:.4f} qrte_qulacs: K={hit} | krylov_bases.py: K={int(uz['K'])} "
          f"max|dE|={np.max(np.abs(uh[:m] - hist[:m])):.1e} Ha ({time.time() - t0:.0f} s)", flush=True)
