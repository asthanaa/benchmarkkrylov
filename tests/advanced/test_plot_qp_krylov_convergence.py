from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "advanced" / "plot_qp_krylov_convergence.py"
SPEC = importlib.util.spec_from_file_location("plot_qp_krylov_convergence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cumulative_new_krylov_spot_checks_match_thick_restart_shape() -> None:
    kwargs = {
        "restart_m_cycle": 30,
        "restart_nroot": 8,
        "first_cycle_seed_count": 1,
    }
    assert MODULE.cumulative_new_krylov(1, 1, **kwargs) == 0
    assert MODULE.cumulative_new_krylov(1, 30, **kwargs) == 29
    assert MODULE.cumulative_new_krylov(2, 8, **kwargs) == 29
    assert MODULE.cumulative_new_krylov(9, 17, **kwargs) == 192


def test_retained_docs_row_for_kprop512_matches_expected_publication_point() -> None:
    row = MODULE.load_golden_docs_row(
        case_name="n2_cas12_12_R2p50",
        phase="kprop",
        budget_value=512,
        method="ccik_thick_qp",
    )
    assert row["fixed_store_keep"] == 65536
    assert row["fixed_propagate_keep"] == 512
    assert row["delta_vs_ref_mEh"] == pytest.approx(0.01923672242298835, abs=1.0e-15)


def test_parse_thick_iteration_log_extracts_points_and_first_beat(tmp_path: Path) -> None:
    log_path = tmp_path / "ccik_stdout.log"
    log_path.write_text(
        "\n".join(
            [
                "ccik(thick) cycle=   1 iter=   1  total=       -108.106424063303  ref=       -108.761373315606  delta_mEh=      654.949252",
                "ccik(thick) cycle=   1 iter=  30  total=       -108.761325831319  ref=       -108.761373315606  delta_mEh=        0.047484",
                "ccik(thick) cycle=   2 iter=   8  total=       -108.761325779325  ref=       -108.761373315606  delta_mEh=        0.047536",
                "ccik(thick) cycle=   9 iter=  17  total=       -108.761354404857  ref=       -108.761373315606  delta_mEh=        0.018911",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    points = MODULE.parse_thick_iteration_log(
        log_path,
        restart_m_cycle=30,
        restart_nroot=8,
        first_cycle_seed_count=1,
    )
    assert [point.cum_new_krylov for point in points] == [0, 29, 29, 192]
    beat = MODULE.first_iteration_beating_reference(points, 0.01923672242298835)
    assert beat is not None
    assert beat.cycle == 9
    assert beat.iter == 17
    assert beat.cum_new_krylov == 192
    assert beat.delta_mEh == pytest.approx(0.018911)


def test_main_writes_sidecar_csv_json_and_plots(tmp_path: Path) -> None:
    log_path = tmp_path / "ccik_stdout.log"
    log_path.write_text(
        "\n".join(
            [
                "ccik(thick) cycle=   1 iter=   1  total=       -108.106424063303  ref=       -108.761373315606  delta_mEh=      654.949252",
                "ccik(thick) cycle=   1 iter=  30  total=       -108.761325831319  ref=       -108.761373315606  delta_mEh=        0.047484",
                "ccik(thick) cycle=   9 iter=  17  total=       -108.761354404857  ref=       -108.761373315606  delta_mEh=        0.018911",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    docs_csv = tmp_path / "results.csv"
    with docs_csv.open("w", encoding="utf-8", newline="") as fh:
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
                "budget_value": "512",
                "fixed_store_keep": "65536",
                "fixed_propagate_keep": "512",
                "config_tag": "restart_30_30_8",
                "reference_total_energy_eh": "-108.76137331560642",
                "e_total_eh": "-108.761354078884",
                "delta_vs_ref_mEh": "0.01923672242298835",
                "abs_delta_vs_ref_mEh": "0.01923672242298835",
                "wall_s": "1413.0249629579484",
                "source_kind": "fresh_qp_donor_seeded",
                "stochastic_walkers": "",
                "failure": "",
            }
        )

    outdir = tmp_path / "out"
    rc = MODULE.main(
        [
            "--log",
            str(log_path),
            "--outdir",
            str(outdir),
            "--docs-results",
            str(docs_csv),
            "--system",
            "n2_cas12_12_R2p50",
            "--method",
            "ccik_thick_qp",
            "--phase",
            "kprop",
            "--budget-value",
            "512",
            "--fixed-store-keep",
            "65536",
            "--restart-m-cycle",
            "30",
            "--restart-nroot",
            "8",
        ]
    )
    assert rc == 0
    assert (outdir / "krylov_convergence.csv").exists()
    assert (outdir / "krylov_convergence.json").exists()
    assert (outdir / "krylov_convergence.png").exists()
    assert (outdir / "krylov_convergence.pdf").exists()

    payload = json.loads((outdir / "krylov_convergence.json").read_text(encoding="utf-8"))
    assert payload["first_iteration_beating_published_row"]["cycle"] == 9
    assert payload["first_iteration_beating_published_row"]["iter"] == 17
    assert payload["first_iteration_beating_published_row"]["cum_new_krylov"] == 192
