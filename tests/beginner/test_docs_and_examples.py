from __future__ import annotations

import importlib.util
import json
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
        ROOT / "docs" / "fixed_budget_results.md",
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


def test_beginner_notebooks_are_self_contained() -> None:
    notebook_paths = [
        ROOT / "examples" / "notebooks" / "01_h2_ground_state.ipynb",
        ROOT / "examples" / "notebooks" / "02_h4_compare_solvers.ipynb",
        ROOT / "examples" / "notebooks" / "03_n2_intro_active_space.ipynb",
    ]
    for path in notebook_paths:
        notebook = json.loads(path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "importlib.util" not in source
        assert "spec_from_file_location" not in source
        assert "examples/scripts" not in source


def test_h2_script_matches_exact_reference() -> None:
    module = _load_example_module("01_h2_ground_state.py")
    result = module.run_example()
    assert abs(result["delta_millihartree"]) < 1e-8


def test_h4_script_produces_thick_budget_comparison() -> None:
    module = _load_example_module("02_h4_compare_solvers.py")
    result = module.run_example()
    assert "e_light_total" in result
    assert "e_refined_total" in result
    assert abs(result["light_delta_mEh"]) < 1.0
    assert abs(result["refined_delta_mEh"]) < 1.0e-5
    assert abs(result["refined_delta_mEh"]) <= abs(result["light_delta_mEh"])


def test_n2_script_runs_and_returns_named_active_space_result() -> None:
    module = _load_example_module("03_n2_intro_active_space.py")
    result = module.run_example()
    assert result["ncas"] == 6.0
    assert result["nelecas"] == 6.0
    assert result["solver_name"] == "ccik_thick"
    assert result["meets_target"] is True
    assert abs(result["delta_millihartree"]) < 1.0
