from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pyscf = pytest.importorskip("pyscf")

ROOT = Path(__file__).resolve().parents[2]


def _load_example_module(filename: str):
    path = ROOT / "examples" / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beginner_docs_reference_real_example_files() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "docs" / "quickstart.md",
        ROOT / "docs" / "examples.md",
        ROOT / "docs" / "glossary.md",
        ROOT / "examples" / "README.md",
        ROOT / "examples" / "scripts" / "01_h2_ground_state.py",
        ROOT / "examples" / "scripts" / "02_h4_compare_solvers.py",
        ROOT / "examples" / "scripts" / "03_n2_intro_active_space.py",
        ROOT / "examples" / "notebooks" / "01_h2_ground_state.ipynb",
        ROOT / "examples" / "notebooks" / "02_h4_compare_solvers.ipynb",
        ROOT / "examples" / "notebooks" / "03_n2_intro_active_space.ipynb",
    ]
    for path in required:
        assert path.exists(), f"expected documented path to exist: {path}"


def test_h2_script_matches_exact_reference() -> None:
    module = _load_example_module("01_h2_ground_state.py")
    result = module.run_example()
    assert abs(result["delta_millihartree"]) < 1e-8


def test_h4_script_produces_solver_comparison() -> None:
    module = _load_example_module("02_h4_compare_solvers.py")
    result = module.run_example()
    assert "e_dense_total" in result
    assert "e_thick_total" in result
    assert "e_stochastic_total" in result
    assert abs(result["dense_delta_mEh"]) < 1.0
    assert abs(result["thick_delta_mEh"]) < 1.0
    assert abs(result["stochastic_delta_mEh"]) < 1.0


def test_n2_script_runs_and_returns_named_active_space_result() -> None:
    module = _load_example_module("03_n2_intro_active_space.py")
    result = module.run_example()
    assert result["ncas"] == 6.0
    assert result["nelecas"] == 6.0
    assert result["solver_name"] == "ccik_dense"
    assert result["meets_target"] is True
    assert abs(result["delta_millihartree"]) < 1.0
