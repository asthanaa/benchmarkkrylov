import csv
import subprocess
import sys
from pathlib import Path

import pytest


pyscf = pytest.importorskip("pyscf")


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "benchmark_selector_refactor.py"


def test_fixed_budget_selector_benchmark_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "selector_benchmark"
    cmd = [
        sys.executable,
        str(_SCRIPT),
        "--outdir",
        str(outdir),
        "--cases",
        "h4_cas4_4",
        "--methods",
        "ccik",
        "--scopes",
        "realistic",
        "--selectors",
        "abs_hq",
        "abs_residual",
        "pt2_residual",
        "ground_state_pt2",
    ]
    subprocess.run(cmd, cwd=_ROOT, check=True, capture_output=True, text=True)

    rows = list(csv.DictReader((outdir / "results.csv").open()))
    assert rows
    assert {row["selector_mode"] for row in rows} == {
        "abs_hq",
        "abs_residual",
        "pt2_residual",
        "ground_state_pt2",
    }
    for col in ("projected_h_fro_error", "projected_h_maxabs_error", "same_basis_ritz_error"):
        assert col in rows[0]

    summary = (outdir / "SUMMARY.md").read_text()
    assert "winner" in summary
