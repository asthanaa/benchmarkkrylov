from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qp_ccik_common import load_prepared_active_space

spec = importlib.util.spec_from_file_location("debug_selector_parity", SCRIPTS / "debug_selector_parity.py")
assert spec is not None and spec.loader is not None
dsp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dsp
spec.loader.exec_module(dsp)


def test_case_registry_contains_donor_sweep_cases() -> None:
    assert dsp.DONOR_N2_SWEEP == (
        "n2_cas10_10_R1p10",
        "n2_cas12_12_R2p50",
        "n2_cas14_14_R2p50",
    )


def test_r_alias_maps_to_cas10_smoke_case() -> None:
    args = dsp.parse_args(["--R", "1.10"])
    cases = dsp._resolve_requested_cases(args)
    assert [case.name for case in cases] == ["n2_cas10_10_R1p10"]


def test_load_prepared_active_space_cas12_donor() -> None:
    ezfio = ROOT / "experiments" / "threshold_test" / "n2_qp_ccik_cas12_12_R2p50_b" / "n2_2p50.ezfio"
    active = load_prepared_active_space(ezfio)
    assert active.ncore == 1
    assert active.ncas == 12
    assert active.nelecas == 12


def test_load_prepared_active_space_cas14_donor() -> None:
    ezfio = ROOT / "experiments" / "threshold_test" / "n2_qp_ccik_cas14_14_hard_budget1e7" / "n2_2p50.ezfio"
    active = load_prepared_active_space(ezfio)
    assert active.ncore == 0
    assert active.ncas == 14
    assert active.nelecas == 14


def test_set_ccik_params_direct_disables_donor_restart_settings(tmp_path: Path) -> None:
    ezfio = tmp_path / "case.ezfio"
    args = argparse.Namespace(
        steps=4,
        selector_mode="abs_residual",
        matvec_backend="connected_sd_fullq",
        compression_policy="threshold",
        fixed_store_keep=123456,
        fixed_propagate_keep=789,
        store_abs_tol=1e-5,
        propagate_abs_tol=1e-6,
        trace_topn=16,
    )
    dsp._set_ccik_params_direct(ezfio, args)
    assert (ezfio / "ccik" / "ccik_basis_size").read_text().strip() == "5"
    assert (ezfio / "ccik" / "ccik_m").read_text().strip() == "5"
    assert (ezfio / "ccik" / "ccik_compression_policy").read_text().strip() == "threshold"
    assert (ezfio / "ccik" / "ccik_fixed_store_keep").read_text().strip() == "123456"
    assert (ezfio / "ccik" / "ccik_fixed_propagate_keep").read_text().strip() == "789"
    assert float((ezfio / "ccik" / "ccik_store_abs_tol").read_text().strip()) == 1.0e-5
    assert float((ezfio / "ccik" / "ccik_propagate_abs_tol").read_text().strip()) == 1.0e-6
    assert (ezfio / "ccik" / "ccik_total_det_budget").read_text().strip() == "0"
    assert (ezfio / "ccik" / "ccik_save_final_wf").read_text().strip() == "F"
    assert (ezfio / "ccik" / "ccik_use_existing_wf").read_text().strip() == "F"
    assert (ezfio / "ccik" / "ccik_do_post_pt2").read_text().strip() == "F"
    assert (ezfio / "ccik" / "ccik_max_cycle").read_text().strip() == "1"
    assert (ezfio / "ccik" / "ccik_restart_m_cycle").read_text().strip() == "0"
    assert (ezfio / "ccik" / "ccik_restart_nroot").read_text().strip() == "2"


def test_parse_args_accepts_fixed_budget_knobs() -> None:
    args = dsp.parse_args(
        [
            "--case",
            "n2_cas10_10_R1p10",
            "--compression-policy",
            "fixed_budget",
            "--fixed-store-keep",
            "4000",
            "--fixed-propagate-keep",
            "512",
            "--omp-threads",
            "4",
        ]
    )
    assert args.compression_policy == "fixed_budget"
    assert args.fixed_store_keep == 4000
    assert args.fixed_propagate_keep == 512
    assert args.omp_threads == "4"


def test_generate_qp_fcidump_prefers_cached_case_file(monkeypatch, tmp_path: Path) -> None:
    from pyscf.tools import fcidump as _pyscf_fcidump  # noqa: F401

    cached = tmp_path / "cached.FCIDUMP"
    cached.write_text(
        "&FCI NORB= 1,NELEC= 2,MS2=0,\n"
        " ORBSYM=1,\n"
        " ISYM=1,\n"
        "&END\n"
        " 0.5 1 1 0 0\n"
        " 0.0 0 0 0 0\n",
        encoding="ascii",
    )
    case = dsp.DONOR_N2_CASES["n2_cas10_10_R1p10"]
    cached_case = dsp.DonorCaseSpec(
        name=case.name,
        donor_ezfio=case.donor_ezfio,
        cached_qp_fcidump=cached,
        basis=case.basis,
        geometry_label=case.geometry_label,
        r_ang=case.r_ang,
        ncore_hint=case.ncore_hint,
        ncas_hint=case.ncas_hint,
        mo_num_hint=case.mo_num_hint,
        recommended_steps=case.recommended_steps,
        exact_hq_diagnostics=case.exact_hq_diagnostics,
        charge=case.charge,
        spin=case.spin,
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when a cached FCIDUMP is available")

    monkeypatch.setattr(dsp.subprocess, "run", fail_run)
    qp = type("DummyQP", (), {"qp_root": tmp_path / "qp", "quantum_package_rc": tmp_path / "qp" / "quantum_package.rc"})()
    ezfio = tmp_path / "case.ezfio"
    ezfio.mkdir()
    outdir = tmp_path / "out"
    outdir.mkdir()

    copied = dsp._generate_qp_fcidump(qp, ezfio, outdir, case_spec=cached_case)

    assert copied == outdir / "qp_active.FCIDUMP"
    assert copied.read_text(encoding="ascii") == cached.read_text(encoding="ascii")
    summary = json.loads((outdir / "qp_fcidump_summary.json").read_text(encoding="utf-8"))
    assert summary["source_kind"] == "cached_case_fcidump"
    assert summary["source_path"] == str(cached)


def test_trace_policy_parser_accepts_old_and_new_layouts(tmp_path: Path) -> None:
    old_trace = tmp_path / "old.log"
    old_trace.write_text(
        "CCIK_TRACE_HQ 1 abs_residual connected_sd_fullq 12 8 4 4 -1.0 0.5 0.4\n"
        "CCIK_TRACE_NEXT 1 abs_residual 16 8 0 1.0 0.5 0.4 0.1 0.2 0.1\n",
        encoding="ascii",
    )
    new_trace = tmp_path / "new.log"
    new_trace.write_text(
        "CCIK_TRACE_HQ 1 abs_residual connected_sd_fullq threshold 12 8 4 4 -1.0 F 0.0 0.5 0.4\n"
        "CCIK_TRACE_NEXT 1 abs_residual threshold 16 8 0 F 1.0 0.5 0.4 0.1 0.2 0.1 0.0\n",
        encoding="ascii",
    )
    fixed_trace = tmp_path / "fixed.log"
    fixed_trace.write_text(
        "CCIK_TRACE_HQ 1 abs_residual connected_sd_fullq fixed_budget 12 8 4 4 -1.0 F 0.0 0.5 0.4\n"
        "CCIK_TRACE_NEXT 1 abs_residual fixed_budget 16 8 0 F 1.0 0.5 0.4 0.1 0.2 0.1 0.0\n",
        encoding="ascii",
    )

    assert dsp._trace_exposes_compression_policy(old_trace) == {"hq": False, "next": False}
    assert dsp._trace_exposes_compression_policy(new_trace) == {"hq": True, "next": True}
    assert dsp._trace_exposes_compression_policy(fixed_trace) == {"hq": True, "next": True}
    parsed_old = dsp._parse_qp_debug_trace(old_trace)
    parsed_new = dsp._parse_qp_debug_trace(new_trace)
    parsed_fixed = dsp._parse_qp_debug_trace(fixed_trace)
    assert parsed_old[1]["hq_meta"]["compression_policy"] == "topk"
    assert parsed_new[1]["hq_meta"]["compression_policy"] == "threshold"
    assert parsed_fixed[1]["hq_meta"]["compression_policy"] == "fixed_budget"


def test_run_qp_trace_resolves_prepared_paths(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp"
    qp_root.mkdir()
    rc = qp_root / "quantum_package.rc"
    rc.write_text("", encoding="ascii")
    ccik = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik" / "ccik"
    ccik.parent.mkdir(parents=True)
    ccik.write_text("", encoding="ascii")

    class DummyQP:
        quantum_package_rc = rc
        ccik_binary = ccik

    captured: dict[str, object] = {}

    def fake_run(cmd, check, text, cwd, capture_output, env, timeout=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout"] = timeout
        trace_path = Path(cwd) / "ccik_debug_trace.log"
        trace_path.write_text(
            "CCIK_TRACE_HQ 1 abs_residual connected_sd_fullq threshold 1 1 1 1 -1.0 F 0.0 0.0 0.0\n"
            "CCIK_TRACE_NEXT 1 abs_residual threshold 1 1 0 F 1.0 0.0 0.0 0.0 0.0 0.0 0.0\n",
            encoding="ascii",
        )
        stdout = "\n".join(
            [
                "ccik_generator_abs_coeff_cutoff = 0.0",
                "ccik_selector_mode = abs_residual",
                "ccik_matvec_backend = connected_sd_fullq",
                "ccik_compression_policy = threshold",
                "ccik_store_abs_tol = 0.0",
                "ccik_propagate_abs_tol = 0.0",
                "ccik_tmp_keep_factor = 1",
                "ccik_debug_trace = T",
                "CCIK_FINAL_ELECTRONIC_ENERGY = -1.0",
                "CCIK_FINAL_TOTAL_ENERGY = -0.5",
                "CCIK_FINAL_BASIS = 2",
                "",
            ]
        )
        return type("Run", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(dsp, "resolve_qp_root", lambda explicit: qp_root)
    monkeypatch.setattr(dsp, "discover_qp_paths", lambda root: DummyQP())
    monkeypatch.setattr(dsp.subprocess, "run", fake_run)

    workdir = tmp_path / "work"
    outdir = tmp_path / "out"
    ezfio = workdir / "case.ezfio"
    ezfio.mkdir(parents=True)
    outdir.mkdir()
    args = argparse.Namespace(
        qp_root=qp_root,
        parity_basis_mode="qp_fcidump",
        matvec_backend="connected_sd_fullq",
        omp_threads="4",
    )
    rel_workdir = Path(os.path.relpath(workdir, Path.cwd()))
    rel_ezfio = Path(os.path.relpath(ezfio, Path.cwd()))
    summary = dsp.run_qp_trace(args, outdir, prepared=(rel_workdir, rel_ezfio))

    assert Path(captured["cwd"]).is_absolute()
    assert str(ezfio.resolve()) in captured["cmd"][2]
    assert captured["env"]["OMP_NUM_THREADS"] == "4"
    assert captured["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert summary["final_total_energy"] == -0.5


def test_run_qp_trace_timeout_handles_bytes(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp"
    qp_root.mkdir()
    rc = qp_root / "quantum_package.rc"
    rc.write_text("", encoding="ascii")
    ccik = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik" / "ccik"
    ccik.parent.mkdir(parents=True)
    ccik.write_text("", encoding="ascii")

    class DummyQP:
        quantum_package_rc = rc
        ccik_binary = ccik

    def fake_run(cmd, check, text, cwd, capture_output, env, timeout=None):
        trace_path = Path(cwd) / "ccik_debug_trace.log"
        trace_path.write_text("partial trace\n", encoding="ascii")
        raise dsp.subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(dsp, "resolve_qp_root", lambda explicit: qp_root)
    monkeypatch.setattr(dsp, "discover_qp_paths", lambda root: DummyQP())
    monkeypatch.setattr(dsp.subprocess, "run", fake_run)

    workdir = tmp_path / "work"
    outdir = tmp_path / "out"
    ezfio = workdir / "case.ezfio"
    ezfio.mkdir(parents=True)
    outdir.mkdir()
    args = argparse.Namespace(
        qp_root=qp_root,
        parity_basis_mode="qp_fcidump",
        matvec_backend="connected_sd_fullq",
        omp_threads="4",
    )

    with pytest.raises(RuntimeError, match="timed out"):
        dsp.run_qp_trace(args, outdir, prepared=(workdir, ezfio), timeout_s=12)

    assert (outdir / "ccik_stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (outdir / "ccik_stderr.log").read_text(encoding="utf-8") == "partial stderr"
