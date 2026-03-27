from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKParams, CCIKThickRestartParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
import debug_selector_parity as dsp  # noqa: E402
from qp_ccik_common import rebuild_ccik_plugin, validate_fresh_qp_root  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="Threshold-mode validation gate")
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "experiments" / f"threshold_mode_validation_{stamp}",
    )
    p.add_argument("--qp-root", type=Path, default=None)
    p.add_argument("--skip-qp", action="store_true", help="Run only the Python threshold gate")
    p.add_argument("--force", action="store_true", help="Ignore cached summaries and rerun all validation stages")
    return p.parse_args(argv)


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _n2_cas10_hamiltonian(r_ang: float) -> tuple[Any, Any, float, int, tuple[int, int]]:
    mol = make_mol_pyscf(
        atom=f"N 0 0 0; N 0 0 {float(r_ang)}",
        basis="6-31g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    cas = CASSpec(ncas=10, nelecas=10, ncore=2)
    return build_cas_hamiltonian_pyscf(mol, cas=cas)


def run_python_threshold_gate(outdir: Path, *, force: bool = False) -> dict[str, object]:
    summary_path = outdir / "python_threshold_gate.json"
    if not force:
        cached = _load_json(summary_path)
        if cached is not None:
            return cached

    rows: list[dict[str, object]] = []
    dense_params = CCIKParams(
        m=32,
        compression_policy="threshold",
        store_abs_tol=0.0,
        propagate_abs_tol=0.0,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )
    thick_params = CCIKThickRestartParams(
        m_cycle=10,
        ncycles=10,
        nroot=3,
        tol=1e-10,
        compression_policy="threshold",
        store_abs_tol=0.0,
        propagate_abs_tol=0.0,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )
    progress_path = outdir / "python_threshold_gate_progress.json"
    progress = None if force else _load_json(progress_path)
    if isinstance(progress, dict) and isinstance(progress.get("rows"), list):
        rows = list(progress["rows"])
    done = {(float(row["geometry_ang"]), str(row["method"])) for row in rows}

    for r_ang in (2.85,):
        h1, eri8, ecore, ncas, nelec = _n2_cas10_hamiltonian(r_ang)
        e_fci = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))

        if (float(r_ang), "dense") not in done:
            dense_stats: dict[str, object] = {}
            e_dense = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=dense_params, stats=dense_stats))
            rows.append(
                {
                    "geometry_ang": float(r_ang),
                    "method": "dense",
                    "energy_eh": e_dense,
                    "delta_vs_fci_mEh": float((e_dense - e_fci) * 1000.0),
                    "ndet_union": dense_stats.get("ndet_union"),
                    "ndet_sum": dense_stats.get("ndet_sum"),
                    "m_eff": dense_stats.get("m_eff"),
                }
            )
            done.add((float(r_ang), "dense"))
            progress_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if (float(r_ang), "thick") not in done:
            thick_stats: dict[str, object] = {}
            e_thick = float(
                ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params, stats=thick_stats)
            )
            rows.append(
                {
                    "geometry_ang": float(r_ang),
                    "method": "thick",
                    "energy_eh": e_thick,
                    "delta_vs_fci_mEh": float((e_thick - e_fci) * 1000.0),
                    "ndet_union": thick_stats.get("ndet_union"),
                    "ndet_sum": thick_stats.get("ndet_sum"),
                    "m_eff": thick_stats.get("m_eff"),
                }
            )
            done.add((float(r_ang), "thick"))
            progress_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = outdir / "python_threshold_gate.csv"
    _write_csv(csv_path, rows)

    max_dense = max(abs(float(row["delta_vs_fci_mEh"])) for row in rows if row["method"] == "dense")
    max_thick = max(abs(float(row["delta_vs_fci_mEh"])) for row in rows if row["method"] == "thick")
    summary = {
        "rows": rows,
        "dense_max_abs_delta_mEh": float(max_dense),
        "thick_max_abs_delta_mEh": float(max_thick),
        "dense_passed": bool(max_dense <= 1.0),
        "thick_passed": bool(max_thick <= 1.0),
        "overall_passed": bool(max_dense <= 1.0 and max_thick <= 1.0),
        "artifacts": {"csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _base_parity_args(qp_root: Path | None) -> argparse.Namespace:
    cli: list[str] = [
        "--with-qp",
        "--parity-basis-mode",
        "qp_fcidump",
        "--selector-mode",
        "abs_residual",
        "--matvec-backend",
        "connected_sd_fullq",
        "--trace-topn",
        "32",
        "--compression-policy",
        "threshold",
    ]
    if qp_root is not None:
        cli.extend(["--qp-root", str(qp_root)])
    return dsp.parse_args(cli)


def _first_step_meta(summary: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    hq_steps = summary.get("hq_steps", [])
    if not isinstance(hq_steps, list) or not hq_steps:
        return None, None
    first = hq_steps[0]
    next_meta = first.get("next_meta") if isinstance(first, dict) else None
    return first if isinstance(first, dict) else None, next_meta if isinstance(next_meta, dict) else None


def _cached_qp_case_result(run_dir: Path) -> dict[str, object] | None:
    run_summary = _load_json(run_dir / "run_summary.json")
    qp_summary = _load_json(run_dir / "qp_summary.json")
    if run_summary is None or qp_summary is None:
        return None

    first_hq, first_next = _first_step_meta(run_summary)
    return {
        "label": run_dir.name,
        "case": run_summary["case"],
        "store_abs_tol": float(run_summary.get("store_abs_tol", 0.0)),
        "propagate_abs_tol": float(run_summary.get("propagate_abs_tol", 0.0)),
        "energy_gap_mEh": run_summary["energy_gap_mEh"],
        "final_step": run_summary["final_step"],
        "final_step_qnext_l2": run_summary["final_step_qnext_l2"],
        "final_step_projected_l2": run_summary["final_step_projected_l2"],
        "final_step_score_l2": run_summary["final_step_score_l2"],
        "stdout_markers_ok": all(bool(x) for x in qp_summary["observed_markers"].values()),
        "trace_policy_ok": all(bool(x) for x in qp_summary["trace_layout"].values()),
        "stale_binary": bool(qp_summary["likely_stale_generated_binary"]),
        "first_step_fullq_support_size": None if first_hq is None else first_hq.get("fullq_support_size"),
        "first_step_source_used_fallback": None if first_hq is None else first_hq.get("source_used_fallback"),
        "first_step_final_keep": None if first_next is None else first_next.get("final_keep"),
        "first_step_store_used_fallback": None if first_next is None else first_next.get("store_used_fallback"),
        "artifacts": {
            "run_summary": str(run_dir / "run_summary.json"),
            "qp_summary": str(run_dir / "qp_summary.json"),
            "compare": str(run_dir / "python_qp_compare.json"),
        },
    }


def run_qp_threshold_gate(outdir: Path, qp_root: Path | None, *, force: bool = False) -> dict[str, object]:
    summary_path = outdir / "qp_threshold_gate.json"
    if not force:
        cached = _load_json(summary_path)
        if cached is not None:
            return cached

    resolved_qp_root = dsp.resolve_qp_root(qp_root)
    if resolved_qp_root is None:
        raise RuntimeError("No Quantum Package root found. Use --qp-root or source quantum_package.rc.")
    qp = dsp.discover_qp_paths(resolved_qp_root)
    rebuild_summary = rebuild_ccik_plugin(qp, clean=True, sync_workspace=True)
    (outdir / "qp_plugin_rebuild.json").write_text(
        json.dumps(
            {
                "plugin_dir": rebuild_summary["plugin_dir"],
                "command": rebuild_summary["command"],
                "returncode": rebuild_summary["returncode"],
                "binary_path": rebuild_summary["binary_path"],
                "binary_exists": rebuild_summary["binary_exists"],
                "sync_summary": rebuild_summary["sync_summary"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    qp_root_validation = validate_fresh_qp_root(resolved_qp_root, outdir=outdir / "qp_root_validation")
    (outdir / "qp_root_validation.json").write_text(
        json.dumps(qp_root_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not qp_root_validation["passed"]:
        raise RuntimeError(
            f"Fresh-root validation failed for {resolved_qp_root}. "
            f"See {outdir / 'qp_root_validation.json'} for details."
        )

    cases = [
        ("zero_cas10", "n2_cas10_10_R1p10", 0.0, 0.0, 2),
        ("zero_cas12", "n2_cas12_12_R2p50", 0.0, 0.0, 1),
        ("store_nonzero", "n2_cas10_10_R1p10", 1.0e-5, 1.0e-8, 1),
        ("prop_nonzero", "n2_cas10_10_R1p10", 1.0e-8, 1.0e-5, 1),
        ("fallback", "n2_cas10_10_R1p10", 2.0, 2.0, 1),
    ]

    base_args = _base_parity_args(resolved_qp_root)
    rows: list[dict[str, object]] = []
    case_results: list[dict[str, object]] = []
    progress_path = outdir / "qp_threshold_gate_progress.json"

    for label, case_name, store_abs_tol, propagate_abs_tol, steps in cases:
        run_dir = outdir / label
        result = None if force else _cached_qp_case_result(run_dir)
        if result is None:
            case = dsp.DONOR_N2_CASES[case_name]
            args_dict = {
                **vars(base_args),
                "case": case_name,
                "steps": int(steps),
                "store_abs_tol": float(store_abs_tol),
                "propagate_abs_tol": float(propagate_abs_tol),
            }
            summary = dsp._run_single_case_backend(
                argparse.Namespace(**args_dict), case, "connected_sd_fullq", base_outdir=run_dir
            )
            qp_summary = json.loads((run_dir / "qp_summary.json").read_text(encoding="utf-8"))
            first_hq, first_next = _first_step_meta(summary)
            result = {
                "label": label,
                "case": case_name,
                "store_abs_tol": float(store_abs_tol),
                "propagate_abs_tol": float(propagate_abs_tol),
                "energy_gap_mEh": summary["energy_gap_mEh"],
                "final_step": summary["final_step"],
                "final_step_qnext_l2": summary["final_step_qnext_l2"],
                "final_step_projected_l2": summary["final_step_projected_l2"],
                "final_step_score_l2": summary["final_step_score_l2"],
                "stdout_markers_ok": all(bool(x) for x in qp_summary["observed_markers"].values()),
                "trace_policy_ok": all(bool(x) for x in qp_summary["trace_layout"].values()),
                "stale_binary": bool(qp_summary["likely_stale_generated_binary"]),
                "first_step_fullq_support_size": None if first_hq is None else first_hq.get("fullq_support_size"),
                "first_step_source_used_fallback": None if first_hq is None else first_hq.get("source_used_fallback"),
                "first_step_final_keep": None if first_next is None else first_next.get("final_keep"),
                "first_step_store_used_fallback": None if first_next is None else first_next.get("store_used_fallback"),
                "artifacts": {
                    "run_summary": str(run_dir / "run_summary.json"),
                    "qp_summary": str(run_dir / "qp_summary.json"),
                    "compare": str(run_dir / "python_qp_compare.json"),
                },
            }
        case_results.append(result)
        rows.append({k: v for k, v in result.items() if k != "artifacts"})
        progress_path.write_text(json.dumps({"cases": case_results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_csv(outdir / "qp_threshold_gate.csv", rows)

    csv_path = outdir / "qp_threshold_gate.csv"
    _write_csv(csv_path, rows)

    by_label = {row["label"]: row for row in case_results}
    zero_cas10 = by_label["zero_cas10"]
    zero_cas12 = by_label["zero_cas12"]
    store_case = by_label["store_nonzero"]
    prop_case = by_label["prop_nonzero"]
    fallback_case = by_label["fallback"]

    zero_gate = (
        abs(float(zero_cas10["energy_gap_mEh"])) <= 0.2
        and float(zero_cas10["final_step_qnext_l2"]) <= 1.0e-5
        and abs(float(zero_cas12["energy_gap_mEh"])) <= 0.2
        and float(zero_cas12["final_step_qnext_l2"]) <= 1.0e-5
    )
    monotonic_store = int(store_case["first_step_final_keep"]) <= int(zero_cas10["first_step_final_keep"])
    monotonic_prop = int(prop_case["first_step_fullq_support_size"]) <= int(zero_cas10["first_step_fullq_support_size"])
    fallback_reported = bool(fallback_case["first_step_source_used_fallback"]) or bool(fallback_case["first_step_store_used_fallback"])
    nonzero_gate = (
        float(store_case["final_step_qnext_l2"]) <= 1.0e-4
        and float(prop_case["final_step_qnext_l2"]) <= 1.0e-4
        and monotonic_store
        and monotonic_prop
        and fallback_reported
    )
    marker_gate = all(bool(row["stdout_markers_ok"]) and bool(row["trace_policy_ok"]) and (not bool(row["stale_binary"])) for row in case_results)

    summary = {
        "rebuild": {
            "plugin_dir": rebuild_summary["plugin_dir"],
            "binary_path": rebuild_summary["binary_path"],
            "binary_exists": rebuild_summary["binary_exists"],
            "sync_summary": rebuild_summary["sync_summary"],
        },
        "cases": case_results,
        "gates": {
            "marker_gate": bool(marker_gate),
            "zero_threshold_gate": bool(zero_gate),
            "nonzero_threshold_gate": bool(nonzero_gate),
            "monotonic_store": bool(monotonic_store),
            "monotonic_prop": bool(monotonic_prop),
            "fallback_reported": bool(fallback_reported),
            "overall_passed": bool(marker_gate and zero_gate and nonzero_gate),
        },
        "artifacts": {"csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    python_summary = run_python_threshold_gate(outdir, force=args.force)
    qp_summary = None
    if not args.skip_qp:
        qp_summary = run_qp_threshold_gate(outdir, args.qp_root, force=args.force)

    overall = {
        "python_gate": python_summary,
        "qp_gate": qp_summary,
        "overall_passed": bool(
            python_summary["overall_passed"]
            and (True if qp_summary is None else qp_summary["gates"]["overall_passed"])
        ),
    }
    (outdir / "release_gate.json").write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
