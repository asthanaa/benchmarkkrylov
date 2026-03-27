"""Detached overnight CCIK campaign for N2 CAS(12,12) on the exact FCIDUMP.

This orchestrates larger dense and thick-restart CCIK runs, one at a time, and
updates a machine-readable campaign status file after every completed step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = ROOT / "scripts" / "n2_cas12_ccik_python_sweep.py"
DEFAULT_OUTDIR = ROOT / "experiments" / "n2_cas12_ccik_overnight_campaign_20260324"
DEFAULT_PYTHON = ROOT / "experiments" / "block2_cr2_d2p0_sto3g" / ".venv" / "bin" / "python"
DEFAULT_COMPARE = ROOT / "experiments" / "n2_cas12_method_compare_20260324"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a detached overnight N2 CAS(12,12) CCIK campaign")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--sweep-script", type=Path, default=DEFAULT_SWEEP)
    p.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return p.parse_args()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_reference(compare_dir: Path) -> dict[str, Any]:
    exact = _read_json(compare_dir / "exact_fci_summary.json")
    partial = _read_json(compare_dir / "results_partial.json")
    dmrg = _read_json(compare_dir / "dmrg" / "dmrg_summary.json")
    cipsi = partial["cipsi"]
    return {
        "exact_fci_total_eh": float(exact["total_energy_eh"]),
        "dmrg_total_eh": float(dmrg["total_energy_eh"]),
        "dmrg_delta_mEh": 1000.0 * (float(dmrg["total_energy_eh"]) - float(exact["total_energy_eh"])),
        "cipsi_total_eh": float(cipsi["e_var"]),
        "cipsi_delta_mEh": 1000.0 * (float(cipsi["e_var"]) - float(exact["total_energy_eh"])),
        "cipsi_saved_det": int(cipsi["saved_det"]),
        "cipsi_final_n_det": int(cipsi["final_n_det"]),
    }


def _step_result_to_summary(summary: dict[str, Any], ref_exact: float) -> dict[str, Any]:
    best_dense = summary.get("best_dense")
    best_thick = summary.get("best_thick")
    best_overall = summary.get("best_overall")
    out = {
        "best_dense": best_dense,
        "best_thick": best_thick,
        "best_overall": best_overall,
    }
    if best_overall is not None:
        out["best_overall_delta_mEh"] = 1000.0 * (float(best_overall["e_total"]) - ref_exact)
    return out


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    if not args.sweep_script.exists():
        raise FileNotFoundError(args.sweep_script)

    ref = _load_reference(args.compare_dir)
    steps = [
        {"kind": "dense", "label": "dense_full_m60_vweak"},
        {"kind": "thick", "label": "thick_full_m30_vweak"},
        {"kind": "dense", "label": "dense_full_m80_ultra"},
        {"kind": "thick", "label": "thick_full_m40_ultra"},
    ]

    status: dict[str, Any] = {
        "started_at": _now(),
        "state": "running",
        "reference": ref,
        "planned_steps": steps,
        "current_step": None,
        "completed_steps": [],
        "best_dense": None,
        "best_thick": None,
        "best_overall": None,
    }
    _write_json(args.outdir / "campaign_status.json", status)

    for step in steps:
        label = str(step["label"])
        kind = str(step["kind"])
        step_outdir = args.outdir / label
        step_outdir.mkdir(parents=True, exist_ok=True)
        step_log = step_outdir / "run.log"
        status["current_step"] = {**step, "started_at": _now(), "outdir": str(step_outdir)}
        _write_json(args.outdir / "campaign_status.json", status)

        cmd = [str(args.python), str(args.sweep_script), "--outdir", str(step_outdir)]
        if kind == "dense":
            cmd.extend(["--dense-label", label, "--thick-rungs", "0"])
        else:
            cmd.extend(["--dense-rungs", "0", "--thick-label", label])

        with step_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)

        if proc.returncode != 0:
            status["state"] = "failed"
            status["current_step"]["rc"] = int(proc.returncode)
            status["current_step"]["ended_at"] = _now()
            _write_json(args.outdir / "campaign_status.json", status)
            return int(proc.returncode)

        summary = _read_json(step_outdir / "summary.json")
        step_record = {
            "kind": kind,
            "label": label,
            "ended_at": _now(),
            "outdir": str(step_outdir),
            **_step_result_to_summary(summary, ref["exact_fci_total_eh"]),
        }
        status["completed_steps"].append(step_record)
        status["current_step"] = None

        best_dense = [rec["best_dense"] for rec in status["completed_steps"] if rec.get("best_dense") is not None]
        best_thick = [rec["best_thick"] for rec in status["completed_steps"] if rec.get("best_thick") is not None]
        if best_dense:
            status["best_dense"] = min(best_dense, key=lambda row: float(row["e_total"]))
        if best_thick:
            status["best_thick"] = min(best_thick, key=lambda row: float(row["e_total"]))
        all_best = [row for row in (status.get("best_dense"), status.get("best_thick")) if row is not None]
        if all_best:
            status["best_overall"] = min(all_best, key=lambda row: float(row["e_total"]))

        _write_json(args.outdir / "campaign_status.json", status)

    status["state"] = "completed"
    status["ended_at"] = _now()
    _write_json(args.outdir / "campaign_status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
