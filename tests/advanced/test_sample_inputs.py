from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = ROOT / "sample_inputs" / "n2_error_iterations"


def test_n2_error_iteration_sample_inputs_are_qp_ckci0_cases() -> None:
    input_files = sorted((SAMPLE_ROOT / "cases").glob("*/input.json"))
    assert [path.parent.name for path in input_files] == [
        "n2_cas10_10_k1000_s050000",
        "n2_cas12_12_k1000_s050000",
    ]

    active_spaces = []
    for input_path in input_files:
        spec = json.loads(input_path.read_text(encoding="utf-8"))
        active_spaces.append((spec["cas"]["nelecas"], spec["cas"]["ncas"]))
        assert spec["study"]["max_cumulative_iterations"] == 300
        assert spec["study"]["reference_mode"] == "exact_fci"
        assert len(spec["runs"]) == 1

        run = spec["runs"][0]
        params = run["solver"]["params"]
        assert run["name"] == "k1000_s050000"
        assert run["solver"]["backend"] == "qp"
        assert params["compression_policy"] == "fixed_budget"
        assert params["fixed_propagate_keep"] == 1000
        assert params["fixed_store_keep"] == 50000
        assert params["exactized_checkpoint_every"] == 10

    assert active_spaces == [(10, 10), (12, 12)]


def test_n2_error_iteration_wrapper_points_at_tracked_inputs() -> None:
    script = (SAMPLE_ROOT / "run_n2_error_plots.py").read_text(encoding="utf-8")
    assert "n2_cas10_10_k1000_s050000" in script
    assert "n2_cas12_12_k1000_s050000" in script
    assert "n2_error_vs_iteration_k1000_s050000" in script
