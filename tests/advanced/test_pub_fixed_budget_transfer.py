from __future__ import annotations

import importlib.util
import json
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
    args = MODULE.parse_args(["--systems", "cr2_ccpvdzdk_R1p68_cas12_22"])
    assert "cr2_fixed_budget_pub_" in str(args.outdir)
    assert args.methods == "ccik_thick_qp"


def test_cr2_summary_mentions_best_kprop_and_kstore(tmp_path: Path) -> None:
    case = next(case for case in MODULE.CASES if case.name == "cr2_ccpvdzdk_R1p68_cas12_22")
    choice_dir = tmp_path / "systems" / case.name
    choice_dir.mkdir(parents=True)
    (choice_dir / "reference_choice.json").write_text(
        json.dumps({"selected_reference_label": "CIPSI variational"}),
        encoding="utf-8",
    )
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
    }
    MODULE._write_summary_markdown(tmp_path, rows, [case], ("ccik_thick_qp",))
    text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Cr2 Publication Summary" in text
    assert "selected reference: CIPSI variational" in text
    assert "best thick-QP Kprop: 1024" in text
    assert "best thick-QP Kstore: 65536" in text


def test_cr2_reference_pipeline_falls_back_to_cipsi(tmp_path: Path, monkeypatch) -> None:
    case = next(case for case in MODULE.CASES if case.name == "cr2_ccpvdzdk_R1p68_cas12_22")
    setup = {
        "fcidump_path": str(tmp_path / "qp_active.FCIDUMP"),
        "ezfio_path": str(tmp_path / "cr2.ezfio"),
    }
    Path(setup["fcidump_path"]).write_text("dummy", encoding="utf-8")
    Path(setup["ezfio_path"]).mkdir()

    dmrg_attempt = {"success": False, "failure": "cap", "reference_kind": "dmrg", "reference_label": "DMRG"}
    cipsi_attempt = {
        "success": True,
        "reference_kind": "cipsi",
        "reference_label": "CIPSI variational",
        "total_energy_eh": -1.234,
    }

    monkeypatch.setattr(MODULE, "_attempt_cr2_dmrg_reference_bounded", lambda **kwargs: dmrg_attempt)
    monkeypatch.setattr(MODULE, "_attempt_cr2_cipsi_reference", lambda **kwargs: cipsi_attempt)

    ref = MODULE._load_or_compute_reference(
        case=case,
        setup=setup,
        outdir=tmp_path,
        force=True,
        qp_root=tmp_path,
        omp_threads=4,
        reference_options=MODULE.ReferenceOptions(),
    )
    assert ref["reference_kind"] == "cipsi"
    choice = json.loads((tmp_path / "reference_choice.json").read_text(encoding="utf-8"))
    assert choice["selected_reference_kind"] == "cipsi"
    assert choice["selected_reference_label"] == "CIPSI variational"
