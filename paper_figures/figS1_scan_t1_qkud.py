"""Fig. S1 extension: QKUD-T1 (epsilon = f * dt_MF) and QRTE-T1 (dt = f * dt_MF) scans for the
eight Fig. S1 systems, using the repository's cached CASSCF/natural-orbital Hamiltonians, the
saved dt_MF values and the same stopping rules as the published scans (100 iterations,
overlap_tol 1e-10, 1.6 mHa against the cached reference energy).

usage: python paper_figures/figS1_scan_t1_qkud.py [repo_root] [out_json] [--validate] [--fast]
           [--series qkud,trotter1] [--cases a,b] [--compare paper_figures/data/figS1_t1_qkud_runs.json --limit n]
(defaults: repo root, paper_figures/output/figS1_t1_qkud_runs.json; the stored runs are in
paper_figures/data/figS1_t1_qkud_runs.json; a full run is 176 calculations)
"""
import csv, json, sys, time, os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
_pos, _skip = [], False
for _a in sys.argv[1:]:
    if _skip:
        _skip = False
        continue
    if _a in ("--series", "--cases", "--compare", "--limit"):
        _skip = True
        continue
    if not _a.startswith("--"):
        _pos.append(_a)
repo = Path(_pos[0]).resolve() if _pos else HERE.parent
out_json = (Path(_pos[1]) if len(_pos) > 1 else HERE / "output" / "figS1_t1_qkud_runs.json").resolve()
out_json.parent.mkdir(parents=True, exist_ok=True)
validate = "--validate" in sys.argv
series_sel = ("qkud", "trotter1")
case_sel = None
for i, a in enumerate(sys.argv):
    if a == "--series":
        series_sel = tuple(sys.argv[i + 1].split(","))
    if a == "--cases":
        case_sel = set(sys.argv[i + 1].split(","))

sys.path.insert(0, str(repo / "vendor/QCANT"))
sys.path.insert(0, str(repo / "scripts/Mushir_script"))
if not (repo / "vendor/QCANT/QCANT/_version.py").exists():  # git-ignored; generated on install
    import types
    sys.modules["QCANT._version"] = types.ModuleType("QCANT._version")
    sys.modules["QCANT._version"].__version__ = "vendored"
from QCANT import qulacs_accel as qa  # noqa: E402
from compute_initial_overlap import _payload_from_cache_path  # noqa: E402
from run_qrte_fock_pi_over_w import cached_hamiltonian  # noqa: E402

FAST = "--fast" in sys.argv
if FAST:
    # Same Krylov vectors (qulacs circuits) and same generalized eigenproblem; only the projected
    # Hamiltonian entries <v_i|H|v_j> are computed from one sparse H*v_j product instead of
    # term-wise qulacs transition amplitudes.
    _SPARSE_CACHE = {}
    CURRENT_CASE = {"key": None}
    _orig_init = qa._ProjectionAccumulator.__init__

    def _fast_init(self, h_qulacs, np_, *, max_workers, capacity=None):
        _orig_init(self, h_qulacs, np_, max_workers=max_workers, capacity=capacity)
        key = CURRENT_CASE["key"]
        if key not in _SPARSE_CACHE:
            _SPARSE_CACHE.clear()
            _SPARSE_CACHE[key] = h_qulacs.get_matrix().tocsr()
        self._hmat = _SPARSE_CACHE[key]
        self._hvecs = []

    def _fast_append(self, state):
        np_ = self.np
        stored_state = state.copy()
        vector = np_.asarray(stored_state.get_vector(), dtype=complex)
        norm = float(np_.linalg.norm(vector))
        if not np_.isfinite(norm) or norm == 0.0:
            raise ValueError("projected basis contains a zero- or non-finite-norm vector")
        idx = self.size
        self._ensure_capacity(idx + 1)
        if idx:
            existing = np_.stack(self._vectors, axis=0)
            overlaps = existing.conj() @ vector
            self._overlap[:idx, idx] = overlaps
            self._overlap[idx, :idx] = overlaps.conj()
        self._overlap[idx, idx] = norm * norm
        self._states.append(stored_state)
        self._vectors.append(vector)
        self._norms[idx] = norm
        hv = self._hmat @ vector
        self._hvecs.append(hv)
        for row_idx in range(idx + 1):
            value = complex(np_.vdot(self._vectors[row_idx], hv))
            self._h_proj[row_idx, idx] = value
            if row_idx != idx:
                self._h_proj[idx, row_idx] = np_.conj(value)

    qa._ProjectionAccumulator.__init__ = _fast_init
    qa._ProjectionAccumulator.append = _fast_append

CHEM = 1.6e-3
N_STEPS = 100
OVERLAP_TOL = 1e-10
FACTORS = (.01, .025, .05, .1, .25, .5, .75, 1., 1.25, 1.5, 2.)
CASES = ("N2_equi_cc_pVTZ_0", "N2_equi_cc_pVTZ_2", "hexatriene_equi_cc_pVTZ_0", "hexatriene_equi_cc_pVTZ_2",
         "H4_near_equi_sto3g_0", "H6_near_equi_sto3g_0", "H6_near_equi_cc_pVTZ_0", "H6_near_equi_cc_pVTZ_2")

scan_rows = list(csv.DictReader((repo / "plots/qrte_dt_scan_final/qrte_exact_vs_trotter2_loglog.csv").open()))
manifest = {r["case_id"]: r for r in csv.DictReader((repo / "manifests/molvqe21_fig1b_manifest.csv").open())}

cases = {}
for r in scan_rows:
    cid = r["case_id"]
    if cid in cases:
        continue
    m = manifest[cid]
    cases[cid] = {
        "case_id": cid,
        "cache_path": repo / r["hamiltonian_cache_path"],
        "dt_mf": float(r["unscaled_mean_field_delta_t"]),
        "reference_energy": float(r["cached_reference_energy_hartree"]),
        "qubits": int(r["qubits"]),
        "symbols": json.loads(m["symbols"]),
        "geometry": np.asarray(json.loads(m["geometry_angstrom"]), dtype=float),
        "basis": m["basis"], "charge": int(m["charge"]), "spin": int(m["spin"]),
        "active_electrons": int(m["active_electrons"]), "active_orbitals": int(m["active_orbitals"]),
    }
published = {(r["case_id"], r["series"], float(r["dt_scale_factor"])): r for r in scan_rows}


def first_hit(history, reference):
    errors = np.asarray(history, dtype=float) - reference
    return next((i for i, e in enumerate(errors, 1) if abs(e) <= CHEM), None)


def run(case, series, factor):
    payload = _payload_from_cache_path(case["cache_path"], hamiltonian_cutoff=1e-20, qa=qa)
    if FAST:
        CURRENT_CASE["key"] = str(case["cache_path"])
    assert abs(float(payload["reference_energy"]) - case["reference_energy"]) < 1e-8
    early = case["reference_energy"] + CHEM
    common = dict(symbols=case["symbols"], geometry=case["geometry"], basis=case["basis"], charge=case["charge"],
                  spin=case["spin"], active_electrons=case["active_electrons"], active_orbitals=case["active_orbitals"],
                  n_steps=N_STEPS, overlap_tol=OVERLAP_TOL, max_workers=1, early_stop_energy=early,
                  return_min_energy_history=True, hamiltonian_source="casscf_natorb")
    t0 = time.perf_counter()
    with cached_hamiltonian(qa, payload):
        if series == "qkud":
            _, _, hist = qa.qkud_qulacs(epsilon=factor * case["dt_mf"], evolution_mode="trotter", trotter_steps=1, **common)
        elif series == "trotter1":
            _, _, _, hist = qa.qrte_qulacs(delta_t=factor * case["dt_mf"], evolution_mode="trotter", trotter_steps=1,
                                           trotter_order=1, **common)
        elif series == "trotter2":
            _, _, _, hist = qa.qrte_qulacs(delta_t=factor * case["dt_mf"], evolution_mode="trotter", trotter_steps=1,
                                           trotter_order=2, **common)
        elif series == "exact":
            _, _, _, hist = qa.qrte_qulacs(delta_t=factor * case["dt_mf"], evolution_mode="sparse", trotter_steps=1,
                                           trotter_order=1, **common)
        else:
            raise ValueError(series)
    hist = np.asarray(hist, dtype=float)
    hit = first_hit(hist, case["reference_energy"])
    return {"case_id": case["case_id"], "series": series, "dt_scale_factor": factor,
            "parameter_au": factor * case["dt_mf"], "dt_mf": case["dt_mf"], "hit_iteration": hit,
            "n_steps_completed": int(hist.size), "final_error_hartree": float(hist[-1] - case["reference_energy"]),
            "seconds": time.perf_counter() - t0}


if "--compare" in sys.argv:
    ref = json.loads(Path(sys.argv[sys.argv.index("--compare") + 1]).read_text())
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(ref)
    bad = 0
    for p in ref[:limit]:
        r = run(cases[p["case_id"]], p["series"], p["dt_scale_factor"])
        same = (r["hit_iteration"] == p["hit_iteration"] and r["n_steps_completed"] == p["n_steps_completed"])
        bad += not same
        print("COMPARE", p["case_id"], p["series"], p["dt_scale_factor"], "prod:", p["hit_iteration"], p["n_steps_completed"],
              f"{p['final_error_hartree']:.3e}", "fast:", r["hit_iteration"], r["n_steps_completed"], f"{r['final_error_hartree']:.3e}",
              "OK" if same else "MISMATCH", f"{r['seconds']:.1f}s", flush=True)
    print("mismatches:", bad, "of", limit)
    sys.exit(0)

results = json.loads(out_json.read_text()) if out_json.exists() else []
done = {(r["case_id"], r["series"], r["dt_scale_factor"]) for r in results}

if validate:
    for cid, series, factor in (("N2_equi_cc_pVTZ_0", "trotter2", 1.0), ("N2_equi_cc_pVTZ_2", "exact", 0.1),
                                ("H6_near_equi_cc_pVTZ_2", "trotter2", 0.05), ("hexatriene_equi_cc_pVTZ_0", "exact", 2.0)):
        r = run(cases[cid], series, factor)
        p = published[(cid, series, factor)]
        print("VALIDATE", cid, series, factor, "mine:", r["hit_iteration"], r["n_steps_completed"],
              "published:", p["hit_iteration"] or None, p["n_steps_completed"], f"{r['seconds']:.1f}s", flush=True)
    sys.exit(0)

for cid in CASES:
    if case_sel and cid not in case_sel:
        continue
    for series in series_sel:
        for factor in FACTORS:
            if (cid, series, factor) in done:
                continue
            r = run(cases[cid], series, factor)
            results.append(r)
            out_json.write_text(json.dumps(results, indent=1))
            print(f"{cid:28s} {series:8s} f={factor:<5g} param={r['parameter_au']:.4f} hit={r['hit_iteration']} "
                  f"steps={r['n_steps_completed']} {r['seconds']:.1f}s", flush=True)
