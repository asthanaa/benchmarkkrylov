from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "reference_backends.py"
SPEC = importlib.util.spec_from_file_location("reference_backends", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalize_reference_summary_keeps_known_kinds() -> None:
    dmrg = MODULE.normalize_reference_summary({"method": "dmrg"})
    fci = MODULE.normalize_reference_summary({"reference_kind": "fci"})
    cipsi = MODULE.normalize_reference_summary({"reference_kind": "variational_cipsi"})

    assert dmrg["reference_kind"] == "dmrg"
    assert dmrg["reference_label"] == "DMRG"
    assert fci["reference_kind"] == "exact_fci"
    assert fci["reference_label"] == "Exact FCI"
    assert cipsi["reference_kind"] == "cipsi"
    assert cipsi["reference_label"] == "CIPSI variational"


def test_normalized_c1_isym_maps_zero_to_one() -> None:
    assert MODULE._normalized_c1_isym(0) == 1
    assert MODULE._normalized_c1_isym("0") == 1
    assert MODULE._normalized_c1_isym(3) == 3
    assert MODULE._normalized_c1_isym("bad") == 1


def test_dmrg_reference_script_uses_explicit_hamiltonian_init() -> None:
    script = MODULE._dmrg_reference_script()

    assert 'dmrg.init_hamiltonian(' in script
    assert "init_hamiltonian_fcidump" not in script
    assert '"init_api": init_api' in script


def test_run_capped_command_stops_on_file_cap(tmp_path: Path) -> None:
    watch_root = tmp_path / "watch"
    watch_root.mkdir()
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    capped = MODULE.run_capped_command(
        cmd=[
            sys.executable,
            "-c",
            (
                "import pathlib, time; "
                "p=pathlib.Path(r'%s')/'big.bin'; "
                "p.write_bytes(b'x' * (2 * 1024 * 1024)); "
                "time.sleep(10)"
            )
            % watch_root,
        ],
        cwd=None,
        env=os.environ.copy(),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        watch_roots=[watch_root],
        max_file_mb=1,
        max_total_scratch_gb=1.0,
        poll_s=0.05,
    )

    assert capped["stop_reason"] == "max_file_mb_exceeded"
    assert float(capped["largest_file_mb"]) >= 1.0


def test_run_capped_command_stops_on_total_scratch_cap(tmp_path: Path) -> None:
    watch_root = tmp_path / "watch"
    watch_root.mkdir()
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    capped = MODULE.run_capped_command(
        cmd=[
            sys.executable,
            "-c",
            (
                "import pathlib, time; "
                "root=pathlib.Path(r'%s'); "
                "(root/'a.bin').write_bytes(b'x' * (2 * 1024 * 1024)); "
                "(root/'b.bin').write_bytes(b'y' * (2 * 1024 * 1024)); "
                "time.sleep(10)"
            )
            % watch_root,
        ],
        cwd=None,
        env=os.environ.copy(),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        watch_roots=[watch_root],
        max_file_mb=100,
        max_total_scratch_gb=0.001,
        poll_s=0.05,
    )

    assert capped["stop_reason"] == "total_scratch_cap_exceeded"
    assert float(capped["peak_total_scratch_gb"]) > 0.0
