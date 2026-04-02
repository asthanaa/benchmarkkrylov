from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import subprocess

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


def test_qp_publication_rows_clear_debug_flags_even_if_seed_is_dirty(tmp_path: Path, monkeypatch) -> None:
    case = next(case for case in MODULE.CASES if case.name == "n2_cas12_12_R2p50")
    base_ezfio = tmp_path / "seed.ezfio"
    base_ezfio.mkdir()
    run_dir = tmp_path / "run"
    rc = tmp_path / "quantum_package.rc"
    ccik = tmp_path / "ccik"
    rc.write_text("", encoding="utf-8")
    ccik.write_text("", encoding="utf-8")

    writes: list[tuple[str, object]] = []

    monkeypatch.setattr(
        MODULE,
        "discover_qp_paths_from_root",
        lambda _root: SimpleNamespace(qp_root=tmp_path, quantum_package_rc=rc, ccik_binary=ccik),
    )
    monkeypatch.setattr(MODULE, "ensure_ezfio_defaults", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "set_ccik_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "qp_parallel_env", lambda _threads: {})
    def fake_mkdtemp(**_kwargs: object) -> str:
        path = tmp_path / "runtime_parent"
        path.mkdir(exist_ok=True)
        return str(path)

    monkeypatch.setattr(MODULE.tempfile, "mkdtemp", fake_mkdtemp)

    def fake_write_ezfio_value(ezfio: Path, key: str, value: object) -> None:
        writes.append((key, value))

    monkeypatch.setattr(MODULE, "write_ezfio_value", fake_write_ezfio_value)
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout_fh = kwargs["stdout"]
        stderr_fh = kwargs["stderr"]
        assert stdout_fh is not None
        assert stderr_fh is not None
        stdout_fh.write("ccik(thick) cycle=1 iter=1 total=-108.0 ref=-108.761373315606 delta_mEh=761.373315606\n")
        stdout_fh.write("CCIK_FINAL_TOTAL_ENERGY = -108.0\n")
        stderr_fh.write("")
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)

    payload = MODULE._run_qp_thick_job(
        case=case,
        reference={"reference_kind": "exact_fci", "reference_label": "Exact FCI", "total_energy_eh": -108.76137331560642},
        qp_root=tmp_path,
        base_ezfio=base_ezfio,
        fixed_store_keep=65536,
        fixed_propagate_keep=512,
        run_dir=run_dir,
        omp_threads=24,
        force=True,
        selector_mode="abs_residual",
    )

    assert payload["failure"] is None
    assert ("ccik/ccik_debug_trace", False) in writes
    assert ("ccik/ccik_print_iteration_error", True) in writes
    assert ("ccik/ccik_reference_total_energy", -108.76137331560642) in writes
    assert "CCIK_FINAL_TOTAL_ENERGY = -108.0" in (run_dir / "ccik_stdout.log").read_text(encoding="utf-8")
