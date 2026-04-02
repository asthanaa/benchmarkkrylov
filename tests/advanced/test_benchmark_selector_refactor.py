import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


pyscf = pytest.importorskip("pyscf")


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "benchmark_selector_refactor.py"
_MODULE_PATH = _ROOT / "scripts" / "advanced" / "benchmark_selector_refactor.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_selector_refactor", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


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
    assert (outdir / "abs_err_vs_molecule_realistic_ccik.png").exists()
    assert (outdir / "abs_err_vs_molecule_realistic_ccik.pdf").exists()


def test_hydrogen_series_cases_use_large_fixed_budget_targets() -> None:
    assert _MODULE.HYDROGEN_SERIES_CASE_NAMES == (
        "h4_series_cas4_4",
        "h10_series_cas10_10",
        "h16_series_cas16_16",
    )
    for name in _MODULE.HYDROGEN_SERIES_CASE_NAMES:
        case = next(case for case in _MODULE.CASES if case.name == name)
        assert case.fixed_store_keep == 65536
        assert case.fixed_propagate_keep == 1024
        assert case.Rs == (1.5,)
