from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "advanced" / "plot_qp_kprop_cutoff_scan.py"
SPEC = importlib.util.spec_from_file_location("plot_qp_kprop_cutoff_scan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_docs_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "system",
                "title",
                "reference_kind",
                "reference_label",
                "method",
                "engine",
                "phase",
                "budget_name",
                "budget_value",
                "fixed_store_keep",
                "fixed_propagate_keep",
                "config_tag",
                "reference_total_energy_eh",
                "e_total_eh",
                "delta_vs_ref_mEh",
                "abs_delta_vs_ref_mEh",
                "wall_s",
                "source_kind",
                "stochastic_walkers",
                "failure",
            ],
        )
        writer.writeheader()
        for budget, energy, delta in [
            (512, -108.761354078884, 0.01923672242298835),
            (1024, -108.761365000000, 0.00831560642),
            (2048, -108.761372000000, 0.00131560642),
        ]:
            writer.writerow(
                {
                    "system": "n2_cas12_12_R2p50",
                    "title": "N2 / 6-31G / R = 2.50 A / CAS(12e,12o)",
                    "reference_kind": "exact_fci",
                    "reference_label": "Exact FCI",
                    "method": "ccik_thick_qp",
                    "engine": "quantum_package",
                    "phase": "kprop",
                    "budget_name": "fixed_propagate_keep",
                    "budget_value": str(budget),
                    "fixed_store_keep": "65536",
                    "fixed_propagate_keep": str(budget),
                    "config_tag": "restart_30_30_8",
                    "reference_total_energy_eh": "-108.76137331560642",
                    "e_total_eh": str(energy),
                    "delta_vs_ref_mEh": str(delta),
                    "abs_delta_vs_ref_mEh": str(delta),
                    "wall_s": "100.0",
                    "source_kind": "fresh_qp_publication_setup",
                    "stochastic_walkers": "",
                    "failure": "",
                }
            )


def _write_log(path: Path, *, target_delta: float, final_delta: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ccik(thick) cycle=   1 iter=   1  total=       -108.100000000000  ref=       -108.761373315606  delta_mEh=      661.373316",
                "ccik(thick) cycle=   1 iter=  30  total=       -108.759300000000  ref=       -108.761373315606  delta_mEh=        2.073316",
                f"ccik(thick) cycle=   5 iter=  13  total=       -108.761100000000  ref=       -108.761373315606  delta_mEh=        {target_delta:.6f}",
                f"ccik(thick) cycle=  10 iter=  20  total=       -108.761200000000  ref=       -108.761373315606  delta_mEh=        {final_delta:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_cutoff_rows_selects_exact_cum100_and_writes_scan_artifacts(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    docs_csv = tmp_path / "results.csv"
    _write_docs_csv(docs_csv)

    run_base = stage_root / "runs" / "n2_cas12_12_R2p50" / "ccik_thick_qp" / "kprop"
    _write_log(run_base / "512" / "ccik_stdout.log", target_delta=0.018911, final_delta=0.000134)
    _write_log(run_base / "1024" / "ccik_stdout.log", target_delta=0.239793, final_delta=0.000671)

    rows, published_rows, missing_budgets = MODULE.build_cutoff_rows(
        stage_root=stage_root,
        docs_results=docs_csv,
        system="n2_cas12_12_R2p50",
        method="ccik_thick_qp",
        phase="kprop",
        fixed_store_keep=65536,
        target_cum_new_krylov=100,
        restart_m_cycle=30,
        restart_nroot=8,
        first_cycle_seed_count=1,
    )
    assert [row["budget_value"] for row in rows] == [512, 1024]
    assert all(row["selection_mode"] == "exact_target" for row in rows)
    assert all(row["cum_new_krylov"] == 100 for row in rows)
    assert missing_budgets == [2048]
    assert len(published_rows) == 3

    rc = MODULE.main(
        [
            "--stage-root",
            str(stage_root),
            "--docs-results",
            str(docs_csv),
            "--system",
            "n2_cas12_12_R2p50",
            "--method",
            "ccik_thick_qp",
            "--phase",
            "kprop",
            "--fixed-store-keep",
            "65536",
            "--target-cum-new-krylov",
            "100",
            "--restart-m-cycle",
            "30",
            "--restart-nroot",
            "8",
        ]
    )
    assert rc == 0
    assert (stage_root / "kprop_scan_cum100.csv").exists()
    assert (stage_root / "kprop_scan_cum100.json").exists()
    assert (stage_root / "kprop_scan_cum100.png").exists()
    assert (stage_root / "kprop_scan_cum100.pdf").exists()

    payload = json.loads((stage_root / "kprop_scan_cum100.json").read_text(encoding="utf-8"))
    assert payload["missing_budgets"] == [2048]
    assert payload["rows"][0]["budget_value"] == 512
    assert payload["rows"][0]["cum_new_krylov"] == 100
    assert payload["rows"][1]["budget_value"] == 1024
