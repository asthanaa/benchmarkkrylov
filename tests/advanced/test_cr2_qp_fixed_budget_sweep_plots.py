from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "advanced" / "cr2_qp_fixed_budget_sweep_plots.py"
SPEC = importlib.util.spec_from_file_location("cr2_qp_fixed_budget_sweep_plots", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_args_defaults_match_cr2_plan() -> None:
    args = MODULE.parse_args([])
    assert args.fixed_store_keep == 131_072
    assert args.fixed_propagate_keep == 4_096
    assert args.store_grid == [32_768, 65_536, 131_072, 262_144, 524_288]
    assert args.propagate_grid == [1_024, 2_048, 4_096, 8_192, 16_384]
    assert "cr2_qp_cr2_sto3g_R1p68_cas28_20_fixed_budget_centered_" in str(args.outdir)


def test_case_definition_matches_requested_system() -> None:
    assert MODULE.CASE.basis == "sto-3g"
    assert MODULE.CASE.distance_ang == 1.68
    assert MODULE.CASE.ncas == 20
    assert MODULE.CASE.nelecas == 28
    assert MODULE.CASE.ncore == 10
    assert MODULE.CASE.fixed_config.method == "thick"
    assert MODULE.CASE.fixed_config.tag == "restart_50_60_12"


def test_calibration_payload_is_fixed_thick_config() -> None:
    payload = MODULE._calibration_payload()
    assert payload["calibration_mode"] == "fixed_structural_config"
    assert payload["dense"]["chosen"] is None
    assert payload["thick"]["chosen"]["restart_m_cycle"] == 50
    assert payload["thick"]["chosen"]["max_cycle"] == 60
    assert payload["thick"]["chosen"]["restart_nroot"] == 12


def test_system_setup_paths_are_stable(tmp_path: Path) -> None:
    paths = MODULE._system_setup_paths(tmp_path)
    assert paths["xyz"] == tmp_path / "system_setup" / "cr2.xyz"
    assert paths["ezfio"] == tmp_path / "system_setup" / "cr2.ezfio"
    assert paths["fcidump"] == tmp_path / "system_setup" / "qp_active.FCIDUMP"


def test_normalized_c1_isym_maps_zero_to_one() -> None:
    assert MODULE._normalized_c1_isym(0) == 1
    assert MODULE._normalized_c1_isym(None) == 1
    assert MODULE._normalized_c1_isym(1) == 1
    assert MODULE._normalized_c1_isym(4) == 4
