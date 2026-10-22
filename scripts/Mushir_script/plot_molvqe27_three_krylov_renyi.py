#!/usr/bin/env python
"""Plot all 27 systems with ADAPT and three QRTE/QKUD configurations.

Read saved histories, retain established corrected/Fe selections, and override
all CO triplet runs with the MolVQE-matching calculations. Compute its entropy
from the new cached Hamiltonian. No molecular propagation is performed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import plot_qkud_eps0p05_vs_qrte_dt0p05 as old
from plot_qrte_mean_field_trotter2_vs_qkud_eps0p05_renyi import QRTE_ROOTS, validate_qrte

np, plt = old.np, old.plt
ROOT = old.BENCHMARK_ROOT
PREFIX = ROOT / "plots/molvqe27_adapt_qrte_dt005_qrte_wmf_qkud_eps005_renyi"
CO = "CO_equi_cc_pVTZ_2"
CO_ROOT = ROOT / "runs/co_spin2_molvqe_matching"
SPECS = {
    "adapt": ("ADAPT-VQE", "#888888", "o"),
    "qrte_fixed": (r"QRTE ($\Delta t=0.05$)", "#0072B2", "s"),
    "qrte_mf": (r"QRTE ($\Delta t=\pi/W_{\rm MF}$)", "#009E73", "D"),
    "qkud": ("QKUD", "#D55E00", "^"),
}
STAR_CASES = {"FeO", "feno_2plus"}
CO_FOLDERS = {"qrte_fixed": "qrte_dt0p05_trotter1",
              "qrte_mf": "qrte_pi_over_wmf_trotter2", "qkud": "qkud_eps0p05_trotter1"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def co_entropy(cache_path):
    from pyscf import fci, lib
    from scipy.linalg import eigh

    lib.num_threads(1)
    with np.load(cache_path, allow_pickle=False) as cache:
        if (int(cache["ncas"]), tuple(cache["nelecas"])) != (6, (4, 2)):
            raise ValueError("Wrong CO cache electron/spin sector")
        addresses, matrix = fci.direct_spin1.pspace(
            cache["h1ecas"], np.swapaxes(cache["two_mo"], 1, 3), 6, (4, 2), np=225)
        if not np.array_equal(np.sort(addresses), np.arange(225)):
            raise ValueError("CO entropy requires the complete CI sector")
        energies, vectors = eigh(matrix)
        ground = float(energies[0] + cache["ecore"])
        if abs(ground - float(cache["reference_energy"])) > 1e-8:
            raise ValueError("CO entropy ground state does not match the new run reference")
    probabilities = np.abs(vectors[:, 0])**2
    probabilities /= probabilities.sum()
    return float(np.log(np.sum(probabilities**.25)) / .75)


def selected_path(cid, series):
    if cid == CO:
        return CO_ROOT / CO_FOLDERS[series] / cid / "summary.json"
    if series == "qrte_fixed" and cid in {"m_Benzyne_equi_cc_pVTZ_0", "m_Benzyne_equi_cc_pVTZ_2"}:
        return ROOT / "runs/qrte_corrected_active_space_dt0.05" / cid / "summary.json"
    if series == "qrte_mf":
        matches = [root / cid / "summary.json" for root in QRTE_ROOTS
                   if (root / cid / "summary.json").exists()]
        if len(matches) != 1:
            raise ValueError(f"{cid}: expected one mean-field QRTE result")
        return matches[0]
    base, overrides = ((old.BASE_QKUD_DIR, old.QKUD_RERUN_SUMMARIES) if series == "qkud"
                       else (old.BASE_QRTE_DIR, old.QRTE_RERUN_SUMMARIES))
    return overrides.get(cid, base / cid / "summary.json")


def validate_run(path, series, cached, fingerprints):
    cid = cached["case_id"]
    summary = json.loads(path.read_text())
    config_path, history_path = path.parent / "run_config.json", path.parent / "min_energy_history.csv"
    config = json.loads(config_path.read_text())
    if summary.get("case_id") != cid or summary.get("status") != "completed":
        raise ValueError(f"{cid}/{series}: missing completed calculation")
    expected_method, order = ("qkud", 1) if series == "qkud" else ("qrte", 2 if series == "qrte_mf" else 1)
    if summary["method"] != expected_method:
        raise ValueError(f"{cid}/{series}: wrong method")
    if (config.get("evolution_mode"), config.get("trotter_order", 1), config.get("trotter_steps")) != ("trotter", order, 1):
        raise ValueError(f"{cid}/{series}: incompatible Trotter settings")
    parameter = "epsilon" if series == "qkud" else "delta_t"
    value = config[parameter]
    if series != "qrte_mf" and not math.isclose(value, .05, rel_tol=0, abs_tol=1e-14):
        raise ValueError(f"{cid}/{series}: expected {parameter}=0.05")
    if series == "qrte_mf":
        validate_qrte(path, cached)
        source = Path(summary["source_mean_field_reference_path"])
        fingerprints[relative(source)] = sha(source)
    cache_path = ROOT / cached["consolidated_cache_path"]
    if sha(cache_path) != cached["sha256"]:
        raise ValueError(f"{cid}: Hamiltonian checksum mismatch")
    saved_cache = summary.get("hamiltonian_cache_path")
    if saved_cache and sha(ROOT / saved_cache) != cached["sha256"]:
        raise ValueError(f"{cid}/{series}: run uses a different Hamiltonian")
    with history_path.open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history or len(history) != int(summary.get("n_steps_completed", summary["n_steps"])):
        raise ValueError(f"{cid}/{series}: incomplete saved history")
    reference = float(summary.get("local_reference_energy_hartree", summary.get("casscf_energy_hartree")))
    errors = np.array([float(row["min_energy_hartree"]) - reference for row in history])
    if not np.all(np.isfinite(errors)) or errors.min() < -.0016:
        raise ValueError(f"{cid}/{series}: invalid reference errors")
    hit = next((int(row["iteration"]) for row, error in zip(history, errors) if abs(error) <= .0016), None)
    if hit != summary["hit_iteration"] or (hit is not None) != summary["hit_chemical_accuracy"]:
        raise ValueError(f"{cid}/{series}: hit iteration disagrees with history")
    for source in (path, config_path, history_path, cache_path):
        fingerprints[relative(source)] = sha(source)
    return {"hit_chemical_accuracy": hit is not None, "iteration": hit if hit is not None else "",
            "n_steps_completed": len(history), "n_steps_requested": summary["n_steps_requested"],
            "reference_energy_hartree": reference, "summary_path": relative(path),
            "hamiltonian_cache_path": relative(cache_path), "hamiltonian_sha256": cached["sha256"],
            "delta_t": value if parameter == "delta_t" else "", "epsilon": value if parameter == "epsilon" else "",
            "trotter_order": order, "trotter_steps": 1,
            "trotter_order_source": "saved config" if "trotter_order" in config else "legacy first-order circuit default",
            "final_absolute_error_hartree": float(abs(errors[-1]))}


def build_rows():
    manifest = {row["case_id"]: row for row in old._read_manifest(old.DEFAULT_MANIFEST)}
    manifest_path = ROOT / "hamiltonian_cache/correct_27_systems/manifest.csv"
    cached_rows = old._read_manifest(manifest_path)
    if len(cached_rows) != 27 or len({row["case_id"] for row in cached_rows}) != 27:
        raise ValueError("Expected exactly 27 validated cases")
    fingerprints = {relative(path): sha(path) for path in (manifest_path, old.DEFAULT_MANIFEST)}
    co_reference_path = CO_ROOT / "reference/reference.json"
    co_reference = json.loads(co_reference_path.read_text())
    fingerprints[relative(co_reference_path)] = sha(co_reference_path)
    co_cache_path = Path(co_reference["case"]["hamiltonian_cache_path"])
    co_h = co_entropy(co_cache_path)
    rows = []
    for cached in cached_rows:
        cid = cached["case_id"]
        molecule = manifest[cid]
        dataset_h = float(molecule["h0.25"])
        raw_path = ROOT / molecule["source_json_path"]
        fingerprints[relative(raw_path)] = sha(raw_path)
        iteration, hit, source, note = old._adapt_iteration(molecule)
        base = {"case_id": cid, "dataset_entropy_nats": dataset_h,
                "dataset_casscf_energy_hartree": float(molecule["casscf_energy_hartree"])}
        rows.append({**base, "series": "adapt", "entropy_nats": dataset_h,
                     "entropy_source": relative(raw_path) + ":meta.casscf.ci_vector",
                     "iteration": iteration, "hit_chemical_accuracy": hit,
                     "reference_energy_hartree": float(molecule["casscf_energy_hartree"]),
                     "adapt_source": source, "note": note})
        if cid == CO:
            cached = {**cached, "consolidated_cache_path": relative(co_cache_path),
                      "sha256": co_reference["case"]["hamiltonian_sha256"]}
        for series in ("qrte_fixed", "qrte_mf", "qkud"):
            result = validate_run(selected_path(cid, series), series, cached, fingerprints)
            if cid == CO and abs(result["reference_energy_hartree"] - co_reference["case"]["local_reference_energy_hartree"]) > 1e-12:
                raise ValueError("A CO run uses the earlier reference")
            rows.append({**base, **result, "series": series,
                         "entropy_nats": co_h if cid == CO else dataset_h,
                         "entropy_source": relative(co_cache_path) + ":exact_CI_ground_state" if cid == CO
                                           else relative(raw_path) + ":meta.casscf.ci_vector",
                         "reference_minus_dataset_hartree": result["reference_energy_hartree"] - base["dataset_casscf_energy_hartree"]})
    return rows, fingerprints


def plot(rows, prefix):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5,
                         "axes.labelsize": 14, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9, 6.6))
    fig.subplots_adjust(left=.13, right=.98, bottom=.21, top=.98)
    fits, counts = {}, {}
    for series, (label, color, marker) in SPECS.items():
        points = [r for r in rows if r["series"] == series and r["hit_chemical_accuracy"]]
        x = np.array([r["entropy_nats"] for r in points])
        y = np.array([float(r["iteration"]) for r in points])
        fit = old._fit_semilog(x, y)
        fits[series] = float(fit[2]) if fit is not None else None
        counts[series] = len(points)
        alpha = .32 if series == "adapt" else .88
        marker_points = [r for r in points if series != "qkud" or r["case_id"] not in STAR_CASES]
        ax.scatter([r["entropy_nats"] for r in marker_points],
                   [float(r["iteration"]) for r in marker_points],
                   marker=marker, s=72 if series != "adapt" else 85,
                   color=color, alpha=alpha, linewidths=.6, edgecolors="white" if series != "adapt" else color,
                   label=label + f", $R^2$={fits[series]:.3f}",
                   zorder=2 if series == "adapt" else 5)
        if fit is not None:
            ax.plot(fit[0], fit[1], color=color, alpha=.3 if series == "adapt" else .8,
                    linestyle="--" if series == "adapt" else "-", linewidth=1.8, zorder=3)
    stars = [r for r in rows if r["series"] == "qkud" and r["case_id"] in STAR_CASES
             and r["hit_chemical_accuracy"]]
    ax.scatter([r["entropy_nats"] for r in stars], [float(r["iteration"]) for r in stars],
               marker="*", s=220, color="#F0E442", edgecolors="#333333", linewidths=.7,
               label="_nolegend_", zorder=8)
    ax.set_yscale("log")
    ax.set_ylim(.8, 650)
    ax.set_xlabel(r"Rényi entropy ($h_{0.25}$)")
    ax.set_ylabel(r"$N_{\mathrm{QKrylov}}$ to $E_{\mathrm{chem}}$ (1.6 mHa)")
    ax.grid(which="major", color="#dddddd", linewidth=.7, alpha=.6)
    ax.grid(which="minor", axis="y", color="#eeeeee", linewidth=.5, alpha=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False,
              columnspacing=2, handletextpad=.7, labelspacing=.8, fontsize=10.5)
    for extension in ("png", "pdf"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=400, bbox_inches="tight")
    plt.close(fig)
    return counts, fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=PREFIX)
    args = parser.parse_args()
    rows, fingerprints = build_rows()
    if len(rows) != 108:
        raise ValueError("Expected four method rows for every one of the 27 systems")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts, fits = plot(rows, args.output_prefix)
    audit = {"case_count": 27, "row_count": len(rows), "successful_cases": counts,
             "r_squared_log_iteration_space": fits, "x_axis": "linear", "y_axis": "logarithmic",
             "renyi_alpha": .25, "entropy_units": "nats", "adapt_opacity": .32,
             "raster_dpi": 400, "title": None, "footnote": None,
             "qrte_no_hit_markers_displayed": False,
             "unlabelled_yellow_qkud_star_cases": sorted(STAR_CASES),
             "fit_definition": "Least squares log(iterations) versus entropy; successful cases only",
             "no_hit_cases": {series: [r["case_id"] for r in rows if r["series"] == series and not r["hit_chemical_accuracy"]]
                              for series in SPECS},
             "co_rows": [r for r in rows if r["case_id"] == CO],
             "entropy_convention": "Dataset CI entropy for ADAPT and other 26 systems. CO Krylov entropy from exact new-cache CI probabilities, normalized with no amplitude cutoff.",
             "notes": ["H10 ADAPT retains the established 324.5-iteration extrapolation, not a measured hit.",
                       "Both fixed-dt m-benzyne cases use corrected-active-space runs matching the validated 27-system Hamiltonians.",
                       "Older configs omit Trotter order and use the first-order circuit default.",
                       "Saved stopping references are retained; Al3-minus has a small legacy dataset/cache offset recorded in CSV.",
                       "fenoco3_minus retains its explicitly unconverged CASSCF/natorb source."],
             "source_sha256": fingerprints, "plot_script_sha256": sha(Path(__file__)),
             "new_molecular_propagations": False}
    args.output_prefix.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Successful cases:", counts)
    print("CO triplet:", {r["series"]: r["iteration"] for r in rows if r["case_id"] == CO})
    for extension in ("png", "pdf", "csv", "json"):
        print(args.output_prefix.with_suffix("." + extension))


if __name__ == "__main__":
    main()
