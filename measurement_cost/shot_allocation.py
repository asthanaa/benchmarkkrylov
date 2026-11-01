"""Step 3: QRTE-T2 shots to chemical accuracy (Fig. 6) by Monte Carlo over shot noise, with sensitivity-weighted
shot allocation.

Measured quantities for a Krylov subspace of dimension k+1 (vectors U^m|Phi>, m = 0..k, from krylov_bases.py):
  H'_jj  = <Phi|U^-j H' U^j|Phi>           1 circuit, j uncontrolled Trotter steps
  H'_jl  = <Phi|U^-j H' U^l|Phi>, j < l     Re and Im Hadamard tests: U^j, then controlled U^(l-j)
  S_m    = <Phi|U^m|Phi>, m = 1..k          Re and Im Hadamard tests, controlled U^m (S is exactly Toeplitz)
H' = H without its identity term (its eigenvalue shift cancels).  H' is not Toeplitz because the Trotter step does
not commute with H.  Per-shot variance <= lambda^2 for H' circuits (one-norm Pauli sampling) and <= 1 for S circuits.

Shots of circuit c are allocated as n_c ~ sigma_c |dE/dx_c| (first-order sensitivity of the energy; mode "shots",
used for Fig. 6), or n_c ~ sigma_c |dE/dx_c| / sqrt(g_c) (mode "gates", g_c = CNOTs of circuit c; reported as mcg_*,
not used in the paper), floored at 1e-2 of the largest share; uniform shots are also tried.  The energy is obtained
by canonical orthogonalization with relative threshold tau in {1e-10, 10^-9.5, ..., 10^-0.5}.  Candidates (k, tau)
must have a noiseless thresholded energy within 1.6 mHa of the reference; for each, Monte Carlo over noise samples
gives the smallest total budget whose energy standard deviation (about that noiseless value) is <= 1.6 mHa
(400 samples for the final check).  The same criterion is used for ADAPT-VQE (noiseless error <= 1.6 mHa at K_A, plus
shot noise sigma = 1.6 mHa).

usage:  python shot_allocation.py <W|MF> [case_id ...]     (default: all work/<rule>/*.npz)
output: results/alloc_<rule>.csv (rows of the processed systems are replaced, others kept)
        mc_k, mc_tau, mc_alloc, mc_shots (Fig. 6), mc_units = mc_shots / (lambda^2/sigma^2)
"""
import sys

import numpy as np

import paths

TOL = 1.6e-3


def solve(Hm, Sm, tau):
    sv, U = np.linalg.eigh(Sm)
    keep = sv > tau * sv[..., -1:]
    scale = np.where(keep, 1.0 / np.sqrt(np.clip(sv, 1e-300, None)), 0.0)
    X = U * scale[..., None, :]
    Hk = np.conj(np.swapaxes(X, -1, -2)) @ Hm @ X
    d = Hm.shape[-1]
    Hk = Hk + np.eye(d) * np.where(keep, 0.0, 1e6)[..., None, :]
    e, Y = np.linalg.eigh(Hk)
    return e[..., 0], X @ Y[..., :, 0][..., None]


def toeplitz_noise(z, k):
    idx = np.arange(k + 1)
    lag = idx[None, :] - idx[:, None]
    M = np.where(lag > 0, z[..., np.abs(lag)], np.where(lag < 0, np.conj(z[..., np.abs(lag)]), 0.0))
    return M


def analyse(z, ns_search=100, ns_final=400):
    S_all, H_all = z["S"], z["H"]
    c_id, lam, K = float(z["c_id"]), float(z["lam"]), int(z["K"])
    Cc, Cu = float(z["Cc"]), float(z["Cu"])
    Hp_all = H_all - c_id * S_all
    Et = float(z["E_ref"]) - c_id
    kmax = S_all.shape[0] - 1
    rng = np.random.default_rng(11)
    ZH = rng.standard_normal((ns_final, kmax + 1, kmax + 1)) + 1j * rng.standard_normal((ns_final, kmax + 1, kmax + 1))
    ZS = rng.standard_normal((ns_final, kmax + 1)) + 1j * rng.standard_normal((ns_final, kmax + 1))
    cands = {"shots": [], "gates": []}
    for k in range(1, kmax + 1):
        d = k + 1
        H0, S0 = Hp_all[:d, :d], S_all[:d, :d]
        iu = np.triu_indices(d, 1)
        for tau in 10.0 ** np.arange(-10, -0.49, 0.5):
            e0, c = solve(H0, S0, tau)
            if abs(e0 - Et) > TOL:
                continue
            a = np.abs(c[:, 0])
            sens_d, g_d = a ** 2, np.arange(d) * Cu
            sens_o = 2 * a[iu[0]] * a[iu[1]]; g_o = iu[0] * Cu + (iu[1] - iu[0]) * Cc
            sens_s = np.array([2 * abs(e0) * np.sum(a[:d - m] * a[m:]) for m in range(1, d)]); g_s = np.arange(1, d) * Cc
            sig = np.concatenate([np.full(d, lam), np.full(len(sens_o), lam), np.ones(k)])
            sens = np.concatenate([sens_d, sens_o, sens_s])
            g = np.concatenate([g_d, g_o, g_s])
            mult = np.concatenate([np.ones(d), 2 * np.ones(len(sens_o)), 2 * np.ones(k)])
            for mode in ("shots", "gates"):
                for alloc in ("sens", "uniform"):
                    if alloc == "uniform":
                        w = np.ones_like(sens)
                    else:
                        w = sig * sens / (np.sqrt(np.maximum(g, 1.0)) if mode == "gates" else 1.0)
                        w = np.maximum(w, 1e-2 * w.max())
                    w = w / np.sum(mult * w)
                    A1 = float(np.sum(mult * (sens * sig) ** 2 / w)) / TOL ** 2
                    vol_per_A = float(np.sum(mult * w * g))
                    cost1 = A1 if mode == "shots" else A1 * vol_per_A
                    cands[mode].append((cost1, k, float(tau), w, vol_per_A, A1, float(e0), alloc, float(a.sum())))
    out = {}
    for mode in ("shots", "gates"):
        best = None
        lst = cands[mode]
        pick = sorted(lst, key=lambda t: t[0])[:10]
        # also the most robust (largest-threshold) choices, which first-order screening undervalues
        pick += sorted(lst, key=lambda t: (-t[2], t[0]))[:10]
        seen, uniq = set(), []
        for t in pick:
            key_ = (t[1], t[2], t[7])
            if key_ not in seen:
                seen.add(key_); uniq.append(t)
        for cost1, k, tau, w, vol_per_A, A1, e0, alloc, c1 in uniq:
            d = k + 1
            H0, S0 = Hp_all[:d, :d], S_all[:d, :d]
            iu = np.triu_indices(d, 1)
            nd, no, nsl = w[:d], w[d:d + len(iu[0])], w[d + len(iu[0]):]
            sdH = np.zeros((d, d))
            sdH[np.arange(d), np.arange(d)] = lam / np.sqrt(nd)
            sdH[iu] = lam / np.sqrt(no)
            sdS = np.concatenate([[0.0], 1.0 / np.sqrt(nsl)])

            def rms(A, ns):
                zh = ZH[:ns, :d, :d] * sdH / np.sqrt(A)
                nh = np.triu(zh, 1)
                nh = nh + np.conj(np.swapaxes(nh, -1, -2))
                idx = np.arange(d)
                nh[:, idx, idx] = zh[:, idx, idx].real
                nsS = toeplitz_noise(ZS[:ns, :d] * sdS / np.sqrt(A), k)
                e, _ = solve(H0[None] + nh, S0[None] + nsS, tau)
                return float(np.sqrt(np.mean((e - e0) ** 2)))

            lo, hi = A1 / 30.0, A1 * 30.0
            if rms(hi, ns_search) > TOL:
                hi = A1 * 1e4
                if rms(hi, ns_search) > TOL:
                    continue
            if rms(lo, ns_search) <= TOL:
                lo = 1.0
            while hi / lo > 1.12:
                mid = np.sqrt(lo * hi)
                if rms(mid, ns_search) <= TOL:
                    hi = mid
                else:
                    lo = mid
            while rms(hi, ns_final) > TOL:
                hi *= 1.12
            cost = hi if mode == "shots" else hi * vol_per_A
            if best is None or cost < best["cost"]:
                best = {"k": k, "tau": tau, "shots": hi, "vol": hi * vol_per_A, "depth": k * Cc, "cost": cost,
                        "first_order": cost1, "alloc": alloc, "bias": e0 - Et, "c1": c1}
        out[mode] = best
    return out



if __name__ == "__main__":
    rule = sys.argv[1]
    assert rule in paths.RULES
    cases = sys.argv[2:] or sorted(f.stem for f in (paths.WORK / rule).glob("*.npz"))
    out = paths.RESULTS / f"alloc_{rule}.csv"
    rows = paths.read_csv(out)
    for cid in cases:
        z = np.load(paths.npz_path(rule, cid))
        lam = float(z["lam"])
        unit = lam ** 2 / TOL ** 2
        o = analyse(z)
        S = z["S"]; K = int(z["K"])
        sv = np.linalg.eigvalsh(S[:K + 1, :K + 1]) if K > 0 else np.array([np.nan])
        r = {"case_id": cid, "K": K, "vectors": S.shape[0], "lambda1": lam, "kappa_S_prod": float(sv[-1] / sv[0]),
             "mc_k": o["shots"]["k"], "mc_tau": o["shots"]["tau"], "mc_shots": o["shots"]["shots"],
             "mc_units": o["shots"]["shots"] / unit, "mc_alloc": o["shots"]["alloc"], "mc_bias_mHa": o["shots"]["bias"] * 1e3,
             "mc_c1": o["shots"]["c1"], "mc_first_order_units": o["shots"]["first_order"] / unit,
             "mcg_k": o["gates"]["k"], "mcg_tau": o["gates"]["tau"], "mcg_shots": o["gates"]["shots"],
             "mcg_gateshots": o["gates"]["vol"], "mcg_units": o["gates"]["vol"] / unit, "mcg_depth_cnot": o["gates"]["depth"],
             "mcg_alloc": o["gates"]["alloc"], "mcg_c1": o["gates"]["c1"]}
        rows[cid] = r
        print(f"{cid[:24]:24s} K={K:2d} vectors={S.shape[0]:3d} shots: k={r['mc_k']:3d} tau={r['mc_tau']:.0e} "
              f"shots={r['mc_shots']:.3g} ({r['mc_units']:.3g} energy evaluations, {r['mc_alloc']})", flush=True)
        paths.write_csv(out, list(rows.values()))
