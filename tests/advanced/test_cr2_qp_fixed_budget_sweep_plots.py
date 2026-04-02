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
    assert args.fixed_store_keep == 65_536
    assert args.fixed_propagate_keep == 1_024
    assert args.store_grid == [32_768, 49_152, 65_536, 88_474, 98_304, 131_072]
    assert args.propagate_grid == [512, 1_024, 2_048]
    assert "cr2_qp_cr2_ccpvdzdk_R1p68_cas12_22_fixed_budget_centered_" in str(args.outdir)


def test_case_definition_matches_requested_system() -> None:
    assert MODULE.CASE.basis == "cc-pvdz-dk"
    assert MODULE.CASE.distance_ang == 1.68
    assert MODULE.CASE.ncas == 22
    assert MODULE.CASE.nelecas == 12
    assert MODULE.CASE.ncore == 18
    assert MODULE.CASE.fixed_config.method == "thick"
    assert MODULE.CASE.fixed_config.tag == "restart_50_60_12"


def test_calibration_payload_is_fixed_thick_config() -> None:
    payload = MODULE._calibration_payload()
    assert payload["calibration_mode"] == "fixed_structural_config"
    assert payload["dense"]["chosen"] is None
    assert payload["thick"]["chosen"]["restart_m_cycle"] == 50
    assert payload["thick"]["chosen"]["max_cycle"] == 60
    assert payload["thick"]["chosen"]["restart_nroot"] == 12


def test_make_fixed_budget_params_tracks_public_budgets() -> None:
    config = MODULE.CASE.fixed_config
    params = MODULE._make_fixed_budget_params(config, fixed_store_keep=88_474, fixed_propagate_keep=1_024)
    assert params.ccik_compression_policy == "fixed_budget"
    assert params.ccik_fixed_store_keep == 88_474
    assert params.ccik_fixed_propagate_keep == 1_024
    assert params.ccik_generator_keep == 1_024
    assert params.ccik_max_parents_for_expansion == 1_024
    assert params.ccik_selector_keep == 88_474
    assert params.ccik_nkeep == 88_474
    assert params.ccik_external_candidate_factor == 4
    assert params.ccik_tmp_keep_factor == 4
    assert params.ccik_preserve_prev_support is True
    assert params.ccik_preserve_prev_support_cap == 1_024


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
