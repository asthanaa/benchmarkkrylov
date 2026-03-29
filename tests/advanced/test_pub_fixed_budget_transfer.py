from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "advanced" / "pub_fixed_budget_transfer.py"
SPEC = importlib.util.spec_from_file_location("pub_fixed_budget_transfer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cr2_only_default_outdir_is_cr2_specific() -> None:
    args = MODULE.parse_args(["--systems", "cr2_sto3g_R1p68_cas28_20"])
    assert "cr2_fixed_budget_pub_" in str(args.outdir)


def test_cr2_summary_mentions_best_kprop_and_kstore(tmp_path: Path) -> None:
    case = next(case for case in MODULE.CASES if case.name == "cr2_sto3g_R1p68_cas28_20")
    rows = {
        (case.name, "ccik_thick_qp", "kprop", 1024): {
            "system": case.name,
            "method": "ccik_thick_qp",
            "phase": "kprop",
            "budget_value": 1024,
            "fixed_store_keep": 65536,
            "fixed_propagate_keep": 1024,
            "abs_delta_vs_ref_mEh": 1.25,
            "wall_s": 10.0,
            "failure": None,
        },
        (case.name, "ccik_thick_qp", "kprop", 512): {
            "system": case.name,
            "method": "ccik_thick_qp",
            "phase": "kprop",
            "budget_value": 512,
            "fixed_store_keep": 65536,
            "fixed_propagate_keep": 512,
            "abs_delta_vs_ref_mEh": 2.5,
            "wall_s": 11.0,
            "failure": None,
        },
        (case.name, "ccik_thick_qp", "kprop", 2048): {
            "system": case.name,
            "method": "ccik_thick_qp",
            "phase": "kprop",
            "budget_value": 2048,
            "fixed_store_keep": 65536,
            "fixed_propagate_keep": 2048,
            "abs_delta_vs_ref_mEh": 1.75,
            "wall_s": 12.0,
            "failure": None,
        },
        (case.name, "ccik_thick_qp", "kstore", 65536): {
            "system": case.name,
            "method": "ccik_thick_qp",
            "phase": "kstore",
            "budget_value": 65536,
            "fixed_store_keep": 65536,
            "fixed_propagate_keep": 1024,
            "abs_delta_vs_ref_mEh": 0.9,
            "wall_s": 14.0,
            "failure": None,
        },
        (case.name, "ccik_stochastic", "kstore", 65536): {
            "system": case.name,
            "method": "ccik_stochastic",
            "phase": "kstore",
            "budget_value": 65536,
            "fixed_store_keep": 65536,
            "fixed_propagate_keep": 1024,
            "abs_delta_vs_ref_mEh": 3.2,
            "wall_s": 18.0,
            "failure": None,
        },
    }
    MODULE._write_summary_markdown(tmp_path, rows, [case])
    text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Cr2 Publication Summary" in text
    assert "best thick-QP Kprop: 1024" in text
    assert "best thick-QP Kstore: 65536" in text
