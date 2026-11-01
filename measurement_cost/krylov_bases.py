"""Step 2: QRTE-T2 Krylov bases and projected matrices S, H for the measurement-cost analysis.

Every step is the production code path of QCANT.qulacs_accel.qrte_qulacs (as called by
scripts/Mushir_script/run_qrte_fock_pi_over_w.py for Fig. 5) with evolution_mode="trotter", trotter_order=2,
trotter_steps=1, overlap_tol=1e-10:
  payload      compute_initial_overlap._payload_from_cache_path(cache, hamiltonian_cutoff=1e-20)
  reference    qa._state_from_vector(qa._basis_vector_from_bits(payload["hf_bits"]))
  propagator   qa._build_time_evolution_circuit(..., trotter_order=2), applied with qa._evolve_state and
               qa._normalize_state (exactly the qrte_qulacs loop)
  energy       qa._diagonalize_projected(S, H, overlap_tol=1e-10)[0]
  hit          first iteration with |E - E_ref| <= 1.6 mHa (E_ref = local_reference_energy_hartree)
The only difference from qrte_qulacs is that <phi_j|H|phi_l> is evaluated as vdot(phi_j, H phi_l), with one
H application per vector (same numbers, lower cost).

rule MF: dt = delta_t of the production run (pi/W_MF, Fig. 5); the energy history is compared with the saved
         min_energy_history.csv of that run (audit JSON, summarised by audit_fig5.py).
rule W : dt = pi/W, W = E_max - E_ref of the fixed (N_alpha, N_beta) sector, E_max by Lanczos (scipy eigsh, tol 1e-9);
         W is cached in results/widths/<case>.json.

After chemical accuracy is reached at iteration K, the basis is extended to min(K + EXTRA, NMAX) vectors so that
the noisy analysis (shot_allocation.py) can use larger bases.  Bases used in the paper: EXTRA=24 for all systems,
and EXTRA=100 (100 vectors) where the shot-optimal dimension was the largest available (see README).

usage:  python krylov_bases.py <W|MF> case_id [case_id ...]      (or "all")
env:    EXTRA (default 24), NMAX (default 100), NMAX_NOHIT (default 100)
output: work/<W|MF>/<case>.npz  with S, H (normalised), c_id, lam, K, dt, W, W_MF, E_ref, hist, Cc, Cu, n_qubits
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse.linalg as sla

import paths

warnings.filterwarnings("ignore")
paths.add_qcant()
from QCANT import qulacs_accel as qa  # noqa: E402

CHEM, OVERLAP_TOL = 1.6e-3, 1e-10
EXTRA, NMAX = int(os.environ.get("EXTRA", "24")), int(os.environ.get("NMAX", "100"))
NMAX_NOHIT = int(os.environ.get("NMAX_NOHIT", "100"))

rule = sys.argv[1]
assert rule in paths.RULES, f"rule must be one of {list(paths.RULES)}"
cases = sys.argv[2:]
rows = paths.fig5_rows()
if cases == ["all"]:
    cases = sorted(paths.read_csv(paths.RESULTS / "resources.csv"))
res = paths.read_csv(paths.RESULTS / "resources.csv")
if not res:
    sys.exit("results/resources.csv missing: run resources.py first")
outdir = paths.WORK / rule
outdir.mkdir(parents=True, exist_ok=True)
paths.WIDTHS.mkdir(parents=True, exist_ok=True)


def sector_mask(bits, n):
    # qa._basis_vector_from_bits is MSB-first: hf_bits position i is qubit n-1-i (alpha/beta swap is harmless,
    # the (N_alpha, N_beta) and (N_beta, N_alpha) sectors have the same spectrum)
    bits = np.asarray(bits)
    na, nb = int(bits[0::2].sum()), int(bits[1::2].sum())
    b = np.arange(2 ** n)
    occ = (b[:, None] >> np.arange(n)[None, :]) & 1
    return ((occ[:, 0::2].sum(1) == na) & (occ[:, 1::2].sum(1) == nb)).astype(float)


for cid in cases:
    t0 = time.time()
    p, summ = paths.payload(cid)
    QS, QC, qg = p["QuantumState"], p["QuantumCircuit"], p["qg"]
    n, Hq, terms = p["qubits"], p["h_qulacs"], p["pauli_terms"]
    E_ref = float(summ["local_reference_energy_hartree"])
    W_MF = float(np.pi / float(summ["delta_t"]))

    def happly(v):
        s_in = QS(n); s_in.load(v)
        s_out = QS(n); s_out.set_zero_state()
        Hq.apply_to_state(QS(n), s_in, s_out)
        return s_out.get_vector().copy()

    W = None
    if rule == "MF":
        dt = float(summ["delta_t"])
    else:
        wf = paths.WIDTHS / f"{cid}.json"
        if wf.exists():
            W = json.load(open(wf))["W"]
        else:
            mask = sector_mask(p["hf_bits"], n)
            op = sla.LinearOperator((2 ** n, 2 ** n), dtype=complex,
                                    matvec=lambda x: happly(np.asarray(x, complex).ravel() * mask) * mask)
            v0 = np.random.default_rng(0).standard_normal(2 ** n) * mask
            emax = float(sla.eigsh(op, k=1, which="LA", v0=v0, tol=1e-9, return_eigenvectors=False)[0])
            W = emax - E_ref
            cache = summ["hamiltonian_cache_path"]
            json.dump({"W": W, "E_max": emax, "E_ref": E_ref, "W_MF": W_MF,
                       "method": "scipy eigsh LA, tol 1e-9, (N_alpha, N_beta) sector",
                       "hamiltonian_cache": cache[cache.index("hamiltonian_cache/"):]}, open(wf, "w"), indent=1)
        dt = float(np.pi / W)
    lam = sum(abs(float(np.real(t["coeff"]))) for t in terms if len(t["indices"]) > 0)
    c_id = sum(float(np.real(t["coeff"])) for t in terms if len(t["indices"]) == 0)
    circ = qa._build_time_evolution_circuit(n, terms, dt, 1, qg, QC, np, trotter_order=2)
    cur = qa._state_from_vector(qa._basis_vector_from_bits(p["hf_bits"], n, np), qubits=n, QuantumState=QS, np=np)
    vecs = [np.asarray(cur.get_vector(), complex).copy()]
    hv0 = happly(vecs[0])
    cap = max(NMAX, NMAX_NOHIT) + 1
    S_raw = np.zeros((cap, cap), complex)
    H_raw = np.zeros((cap, cap), complex)
    nrm = np.zeros(cap)
    S_raw[0, 0] = np.vdot(vecs[0], vecs[0]); H_raw[0, 0] = np.vdot(vecs[0], hv0); nrm[0] = np.linalg.norm(vecs[0])
    hist, K = [], None
    while True:
        cur = qa._evolve_state(cur, evolution_mode="trotter", sparse_hamiltonian=None, circuit=circ, time_step=dt,
                               QuantumState=QS, np=np, expm_multiply=None)
        cur = qa._normalize_state(cur, QS, np)
        v = np.asarray(cur.get_vector(), complex).copy()
        hv = happly(v)
        k = len(vecs)
        Vp = np.array(vecs)
        S_raw[:k, k] = Vp.conj() @ v; S_raw[k, :k] = S_raw[:k, k].conj(); S_raw[k, k] = np.vdot(v, v)
        H_raw[:k, k] = Vp.conj() @ hv; H_raw[k, :k] = H_raw[:k, k].conj(); H_raw[k, k] = np.vdot(v, hv)
        nrm[k] = np.linalg.norm(v)
        del Vp
        vecs.append(v)
        if K is None:
            ns = nrm[:k + 1, None] * nrm[None, :k + 1]
            e0 = float(qa._diagonalize_projected(S_raw[:k + 1, :k + 1] / ns, H_raw[:k + 1, :k + 1] / ns, OVERLAP_TOL, np)[0])
            hist.append(e0)
            if abs(e0 - E_ref) <= CHEM:
                K = k
        if (K is not None and k >= min(K + EXTRA, NMAX)) or (K is None and k >= NMAX_NOHIT):
            break
    d = len(vecs)
    ns = nrm[:d, None] * nrm[None, :d]
    S = S_raw[:d, :d] / ns
    Hm = H_raw[:d, :d] / ns
    Hm = 0.5 * (Hm + Hm.conj().T)
    del vecs
    msg = ""
    if rule == "MF":   # audit against the production run behind Fig. 5
        r = rows[cid]
        prod = [float(x["min_energy_hartree"]) for x in
                __import__("csv").DictReader(open(paths.REPO / os.path.dirname(r["summary_path"]) / "min_energy_history.csv"))]
        m = min(len(prod), len(hist))
        dmax = float(np.max(np.abs(np.array(hist[:m]) - np.array(prod[:m]))))
        K_fig5 = int(float(r["iteration"])) if r["iteration"] and r["hit_chemical_accuracy"].lower() == "true" else None
        msg = f" | Fig. 5: K={K_fig5}, max|dE| vs production history {dmax:.1e} Ha, match={K_fig5 == K}"
        json.dump({"case_id": cid, "n_repo": len(prod), "n_compared": m, "max_abs_dE": dmax, "K_fig5": K_fig5, "K": K,
                   "hist": hist, "prod": prod}, open(outdir / f"{cid}.audit.json", "w"))
    np.savez(outdir / f"{cid}.npz", S=S, H=Hm, c_id=c_id, lam=lam, K=-1 if K is None else K, dt=dt,
             W=np.nan if W is None else W, W_MF=W_MF, E_ref=E_ref, hist=np.array(hist),
             Cc=2 * float(res[cid]["cnot_T1_step_ctrl"]), Cu=2 * float(res[cid]["cnot_T1_step_unctrl"]), n_qubits=n)
    print(f"{cid:26s} n={n:2d} rule={rule} dt={dt:.4f} K={K} vectors={d}{msg} ({time.time() - t0:.0f} s)", flush=True)
