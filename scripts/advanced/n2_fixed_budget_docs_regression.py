from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
ADVANCED = SCRIPTS / "advanced"
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ADVANCED) not in sys.path:
    sys.path.insert(0, str(ADVANCED))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import debug_selector_parity as dsp  # noqa: E402
import pub_fixed_budget_transfer as pub  # noqa: E402
from _shared.n2_utils import make_n2_atom  # noqa: E402
from ccik import exact_cas_fci_energy_pyscf  # noqa: E402
from ccik.params import CCIKParams, CCIKThickRestartParams  # noqa: E402
from ccik.propagation import analyze_krylov_step  # noqa: E402
from ccik.ritz import solve_projected_ritz  # noqa: E402
from ccik.selection import make_full_hdiag_map  # noqa: E402
from ccik.sparse_ci import (  # noqa: E402
    SparseCIVector,
    inner_sparse,
    linear_combination_sparse,
    make_hf_sparse_vector,
    normalize_sparse,
    orthogonalize_sparse,
    sparse_norm,
)
from ccik.thick_restart import _compress_with_stage1, gram_schmidt_list  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    CCIKParams as QPCCIKParams,
    discover_qp_paths_from_root,
    ensure_ezfio_defaults,
    fixed_budget_bridge_kwargs,
    load_prepared_active_space,
    qp_parallel_env,
    run_qp_setup,
    set_ccik_params,
    write_ezfio_value,
)


DOCS_RESULTS = ROOT / "docs" / "assets" / "pub" / "n2_abs_residual_qp_thick" / "results.csv"
LOCAL_SEED_ROOT = ROOT / "docs" / "assets" / "local_qp_seeds"
DEFAULT_CASE = "n2_cas12_12_R2p50"
DEFAULT_PHASE = "kprop"
DEFAULT_BUDGET = 2048
DEFAULT_METHOD = "ccik_thick_qp"
DEFAULT_OUTDIR = ROOT / "experiments" / f"n2_fixed_budget_docs_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_PARITY_OUTDIR = ROOT / "experiments" / f"n2_fixed_budget_small_parity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
ENERGY_TOL_EH = 1.0e-9
DELTA_TOL_MEH = 1.0e-6
PATH_CHOICES = ("donor_seeded_qp", "python_seed_fcidump", "fresh_publication_setup_qp")
BRIDGE_POLICIES = ("current", "notebook", "docs_legacy")
QP_MODES = ("thick", "baseline")
RUN_MODES = ("regression", "small_shared_parity")
SMALL_SHARED_SEED = LOCAL_SEED_ROOT / "n2_cas10_10_R1p10"
SMALL_SHARED_EZFIO = SMALL_SHARED_SEED / "n2.ezfio"
SMALL_SHARED_FCIDUMP = SMALL_SHARED_SEED / "qp_active.FCIDUMP"
SMALL_PARITY_THICK_MCYCLE = 4
SMALL_PARITY_THICK_NCYCLES = 3
SMALL_PARITY_THICK_NROOT = 2
BOUNDARY_SCORE_TOL = 1.0e-14
REPO_PLUGIN_FINGERPRINT_FILES = (
    ROOT / "ccik_qp" / "stable" / "ccik" / "README.rst",
    ROOT / "ccik_qp" / "stable" / "ccik" / "EZFIO.cfg",
    ROOT / "ccik_qp" / "stable" / "ccik" / "ccik.irp.f",
    ROOT / "ccik_qp" / "stable" / "ccik" / "ccik_driver.irp.f",
    ROOT / "ccik_qp" / "stable" / "ccik" / "ccik_selection_providers.irp.f",
    ROOT / "ccik_qp" / "stable" / "ccik" / "ccik_sparse.f90",
)
HISTORICAL_GIT_ANCHORS = (
    {
        "commit": "d4b2255",
        "label": "docs-era fixed-budget introduction",
        "bridge_policy": "docs_legacy",
        "driver_candidates": (
            "scripts/n2_qp_fixed_budget_sweep_plots.py",
            "scripts/qp_ccik_common.py",
        ),
        "comparison_mode": "bridge-only historical replay",
    },
    {
        "commit": "ae53767",
        "label": "docs/tooling reorganization",
        "bridge_policy": "current",
        "driver_candidates": (
            "scripts/advanced/pub_fixed_budget_transfer.py",
            "scripts/_shared/qp_ccik_common.py",
        ),
        "comparison_mode": "bridge-only historical replay",
    },
    {
        "commit": "ef3921e",
        "label": "fixed-budget default and legacy cleanup",
        "bridge_policy": "current",
        "driver_candidates": (
            "scripts/advanced/pub_fixed_budget_transfer.py",
            "scripts/_shared/qp_ccik_common.py",
        ),
        "comparison_mode": "bridge-only historical replay",
    },
    {
        "commit": "5a76802",
        "label": "current base before local fixed-budget debug edits",
        "bridge_policy": "current",
        "driver_candidates": (
            "scripts/advanced/pub_fixed_budget_transfer.py",
            "scripts/_shared/qp_ccik_common.py",
        ),
        "comparison_mode": "bridge-only historical replay",
    },
)
TRACKED_QP_FIELDS = (
    "ccik_basis_size",
    "ccik_generator_keep",
    "ccik_generator_abs_coeff_cutoff",
    "ccik_max_parents_for_expansion",
    "ccik_include_current_support",
    "ccik_selector_keep",
    "ccik_external_keep",
    "ccik_external_candidate_factor",
    "ccik_selector_mode",
    "ccik_matvec_backend",
    "ccik_compression_policy",
    "ccik_fixed_store_keep",
    "ccik_fixed_propagate_keep",
    "ccik_tmp_keep_factor",
    "ccik_preserve_prev_support",
    "ccik_preserve_prev_support_cap",
    "ccik_max_cycle",
    "ccik_restart_m_cycle",
    "ccik_restart_nroot",
    "ccik_reference_total_energy",
    "ccik_print_iteration_error",
)


def _create_xyz(path: Path, atom: str, *, comment: str) -> None:
    lines = [line.strip() for line in atom.strip().splitlines() if line.strip()]
    path.write_text(f"{len(lines)}\n{comment}\n" + "\n".join(lines) + "\n", encoding="ascii")


def _normalize_csv_float(text: str) -> float | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    return float(stripped)


def _load_docs_rows(path: Path = DOCS_RESULTS) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_golden_docs_row(
    *,
    path: Path = DOCS_RESULTS,
    case_name: str = DEFAULT_CASE,
    phase: str = DEFAULT_PHASE,
    budget_value: int = DEFAULT_BUDGET,
    method: str = DEFAULT_METHOD,
) -> dict[str, object]:
    for raw in _load_docs_rows(path):
        if raw["system"] != case_name:
            continue
        if raw["phase"] != phase:
            continue
        if raw["method"] != method:
            continue
        if int(raw["budget_value"]) != int(budget_value):
            continue
        return {
            "system": raw["system"],
            "title": raw["title"],
            "reference_kind": raw["reference_kind"],
            "reference_label": raw["reference_label"],
            "method": raw["method"],
            "engine": raw["engine"],
            "phase": raw["phase"],
            "budget_name": raw["budget_name"],
            "budget_value": int(raw["budget_value"]),
            "fixed_store_keep": int(raw["fixed_store_keep"]),
            "fixed_propagate_keep": int(raw["fixed_propagate_keep"]),
            "config_tag": raw["config_tag"],
            "reference_total_energy_eh": float(raw["reference_total_energy_eh"]),
            "e_total_eh": float(raw["e_total_eh"]),
            "delta_vs_ref_mEh": float(raw["delta_vs_ref_mEh"]),
            "abs_delta_vs_ref_mEh": float(raw["abs_delta_vs_ref_mEh"]),
            "wall_s": _normalize_csv_float(raw["wall_s"]),
            "source_kind": raw["source_kind"],
            "stochastic_walkers": None if not raw["stochastic_walkers"].strip() else int(raw["stochastic_walkers"]),
            "failure": raw["failure"] or None,
        }
    raise FileNotFoundError(
        f"Retained golden row not found in {path}: "
        f"system={case_name}, phase={phase}, budget_value={budget_value}, method={method}"
    )


def resolve_publication_case(case_name: str = DEFAULT_CASE) -> pub.PublicationCase:
    for case in pub.CASES:
        if case.name == case_name:
            return case
    raise KeyError(f"Unknown publication case {case_name!r}")


def _resolve_qp_root_with_binary(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(pub._candidate_qp_roots())
    env_root = os.environ.get("QP_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    seen: set[Path] = set()
    checked: list[dict[str, object]] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        qp = discover_qp_paths_from_root(resolved)
        payload = {
            "qp_root": str(resolved),
            "root_exists": resolved.exists(),
            "quantum_package_rc_exists": qp.quantum_package_rc.exists(),
            "ccik_binary_exists": qp.ccik_binary.exists(),
        }
        checked.append(payload)
        if payload["root_exists"] and payload["quantum_package_rc_exists"] and payload["ccik_binary_exists"]:
            return resolved
    raise RuntimeError(f"No QP root with a built ccik binary was found.\n{json.dumps(checked, indent=2, sort_keys=True)}")


def _require_seed_paths(case: pub.PublicationCase) -> tuple[Path, Path]:
    if case.seed_ezfio is None or case.seed_fcidump is None:
        raise FileNotFoundError(f"Case {case.name} does not define local seed paths")
    if not case.seed_ezfio.exists():
        raise FileNotFoundError(f"Missing local seed EZFIO for {case.name}: {case.seed_ezfio}")
    if not case.seed_fcidump.exists():
        raise FileNotFoundError(f"Missing local seed FCIDUMP for {case.name}: {case.seed_fcidump}")
    return case.seed_ezfio, case.seed_fcidump


def _fcidump_metadata(fcidump_path: Path) -> dict[str, object]:
    from pyscf.tools import fcidump as pyscf_fcidump

    data = pyscf_fcidump.read(str(fcidump_path))
    return {
        "path": str(fcidump_path),
        "NORB": int(data["NORB"]),
        "NELEC": int(data["NELEC"]),
        "MS2": int(data.get("MS2", 0)),
        "ECORE": float(data.get("ECORE", 0.0)),
    }


def _load_fcidump_hamiltonian(fcidump_path: Path) -> dict[str, object]:
    from pyscf.tools import fcidump as pyscf_fcidump

    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec_total = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec_total + ms2) // 2
    nelecb = nelec_total - neleca
    eri = np.asarray(data["H2"], dtype=float)
    h1 = np.asarray(data["H1"], dtype=float)
    ecore = float(data.get("ECORE", 0.0))
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri, norb, (neleca, nelecb)))
    return {
        "path": str(fcidump_path),
        "h1": h1,
        "eri": eri,
        "norb": norb,
        "nelec": (int(neleca), int(nelecb)),
        "ecore": ecore,
        "exact_total_energy_eh": e_fci_total,
        "metadata": _fcidump_metadata(fcidump_path),
    }


def _read_ezfio_float(ezfio: Path, relpath: str) -> float:
    text = (Path(ezfio) / relpath).read_text(encoding="ascii").strip()
    return float(text.split()[0])


def _apply_constant_shift_to_fixed_nelec_h1(
    h1: np.ndarray,
    *,
    nelec: tuple[int, int],
    shift: float,
) -> np.ndarray:
    out = np.array(h1, copy=True, dtype=float)
    nelec_total = int(nelec[0] + nelec[1])
    if nelec_total <= 0:
        return out
    delta = float(shift) / float(nelec_total)
    idx = np.diag_indices_from(out)
    out[idx] += delta
    return out


def _sorted_sparse_items_map(coeffs: dict[tuple[int, int], float], *, limit: int | None = None) -> list[dict[str, object]]:
    ranked = sorted(coeffs.items(), key=lambda item: (abs(item[1]), int(item[0][0]), int(item[0][1])), reverse=True)
    if limit is not None:
        ranked = ranked[: int(limit)]
    return [
        {
            "stra": int(key[0]),
            "strb": int(key[1]),
            "coef": float(val),
            "abs_coef": float(abs(val)),
        }
        for key, val in ranked
    ]


def _sorted_sparse_items(vec: Any, *, limit: int | None = None) -> list[dict[str, object]]:
    return _sorted_sparse_items_map(getattr(vec, "coeffs", {}), limit=limit)


def _restrict_sparse_items(vec: Any, support: set[tuple[int, int]], *, limit: int | None = None) -> list[dict[str, object]]:
    coeffs = {key: float(val) for key, val in getattr(vec, "coeffs", {}).items() if key in support}
    return _sorted_sparse_items_map(coeffs, limit=limit)


def _sorted_det_keys(keys: set[tuple[int, int]], *, limit: int | None = None) -> list[dict[str, int]]:
    ranked = sorted(keys)
    if limit is not None:
        ranked = ranked[: int(limit)]
    return [{"stra": int(stra), "strb": int(strb)} for stra, strb in ranked]


def _vector_from_trace_items(items: list[dict[str, object]]) -> dict[tuple[int, int], float]:
    return {(int(item["stra"]), int(item["strb"])): float(item["coef"]) for item in items}


def _compare_vector_maps_direct(lhs: dict[tuple[int, int], float], rhs: dict[tuple[int, int], float]) -> dict[str, object]:
    keys = set(lhs) | set(rhs)
    if not keys:
        return {
            "support_overlap": 0,
            "support_union": 0,
            "lhs_size": 0,
            "rhs_size": 0,
            "exact_support_match": True,
            "l2_diff": 0.0,
            "max_abs_coef_diff": 0.0,
            "first_diff_key": None,
        }
    max_abs = 0.0
    first_diff = None
    sq = 0.0
    lhs_keys = set(lhs)
    rhs_keys = set(rhs)
    for key in sorted(keys):
        diff = float(lhs.get(key, 0.0) - rhs.get(key, 0.0))
        sq += diff * diff
        if abs(diff) > max_abs:
            max_abs = abs(diff)
            first_diff = key
    return {
        "support_overlap": int(len(lhs_keys & rhs_keys)),
        "support_union": int(len(keys)),
        "lhs_size": int(len(lhs_keys)),
        "rhs_size": int(len(rhs_keys)),
        "exact_support_match": lhs_keys == rhs_keys,
        "l2_diff": float(math.sqrt(sq)),
        "max_abs_coef_diff": float(max_abs),
        "first_diff_key": None if first_diff is None else {"stra": int(first_diff[0]), "strb": int(first_diff[1])},
    }


def _compare_vector_maps(
    lhs: dict[tuple[int, int], float],
    rhs: dict[tuple[int, int], float],
    *,
    allow_global_sign_flip: bool = False,
) -> dict[str, object]:
    direct = _compare_vector_maps_direct(lhs, rhs)
    if not allow_global_sign_flip:
        return direct
    flipped = _compare_vector_maps_direct(lhs, {key: -float(val) for key, val in rhs.items()})
    direct_key = (
        not bool(direct["exact_support_match"]),
        float(direct["max_abs_coef_diff"]),
        float(direct["l2_diff"]),
    )
    flipped_key = (
        not bool(flipped["exact_support_match"]),
        float(flipped["max_abs_coef_diff"]),
        float(flipped["l2_diff"]),
    )
    best = flipped if flipped_key < direct_key else direct
    if best is flipped:
        best = dict(best)
        best["global_sign_flip_equivalent"] = True
        return best
    best = dict(best)
    best["global_sign_flip_equivalent"] = False
    return best


def _compare_scalar(lhs: float | None, rhs: float | None, *, tol: float = 1.0e-12) -> dict[str, object]:
    if lhs is None or rhs is None:
        return {"lhs": lhs, "rhs": rhs, "abs_diff": None, "matches": False}
    diff = abs(float(lhs) - float(rhs))
    return {"lhs": float(lhs), "rhs": float(rhs), "abs_diff": float(diff), "matches": bool(diff <= tol)}


def _spin_swap_pairs(keys: set[tuple[int, int]]) -> int:
    count = 0
    for stra, strb in keys:
        if stra < strb and (strb, stra) in keys:
            count += 1
    return count


def _boundary_cutoff_diag_from_items(items: list[dict[str, object]], *, keep: int) -> dict[str, object]:
    ranked = [
        (
            float(item["abs_coef"]),
            int(item["stra"]),
            int(item["strb"]),
        )
        for item in items
    ]
    total = len(ranked)
    if total == 0:
        return {
            "keep": int(keep),
            "total_size": 0,
            "cutoff_abs_coef": None,
            "equal_score_group_size": 0,
            "equal_score_group_kept": 0,
            "equal_score_group_spin_swap_pairs": 0,
            "first_group_key": None,
            "last_group_key": None,
        }
    keep_eff = min(max(1, int(keep)), total)
    cutoff = ranked[keep_eff - 1][0]
    in_group = [row for row in ranked if abs(row[0] - cutoff) <= BOUNDARY_SCORE_TOL]
    first = min((row[1], row[2]) for row in in_group)
    last = max((row[1], row[2]) for row in in_group)
    kept_in_group = sum(1 for row in ranked[:keep_eff] if abs(row[0] - cutoff) <= BOUNDARY_SCORE_TOL)
    group_keys = {(row[1], row[2]) for row in in_group}
    return {
        "keep": int(keep_eff),
        "total_size": int(total),
        "cutoff_abs_coef": float(cutoff),
        "equal_score_group_size": int(len(in_group)),
        "equal_score_group_kept": int(kept_in_group),
        "equal_score_group_spin_swap_pairs": int(_spin_swap_pairs(group_keys)),
        "first_group_key": {"stra": int(first[0]), "strb": int(first[1])},
        "last_group_key": {"stra": int(last[0]), "strb": int(last[1])},
    }


def _boundary_diag_for_step(
    row: dict[str, object],
    *,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
) -> dict[str, object]:
    return {
        "propagation_from_q_k": _boundary_cutoff_diag_from_items(list(row["q_k"]["top"]), keep=fixed_propagate_keep),
        "store_from_orthogonalized_residual": _boundary_cutoff_diag_from_items(
            list(row["orthogonalized_residual"]["top"]),
            keep=fixed_store_keep,
        ),
        "current_support_unioned": bool(row.get("current_support_unioned", False)),
    }


def _map_qp_det_to_active(det: int, *, ncore: int, ncas: int) -> int:
    out = 0
    for iact in range(int(ncas)):
        full_bit = int(ncore) + iact
        if int(det) & (1 << full_bit):
            out |= 1 << iact
    return out


def _mapped_trace_items(items: list[dict[str, object]], *, ncore: int, ncas: int) -> list[dict[str, object]]:
    return [
        {
            **item,
            "stra": _map_qp_det_to_active(int(item["stra"]), ncore=ncore, ncas=ncas),
            "strb": _map_qp_det_to_active(int(item["strb"]), ncore=ncore, ncas=ncas),
        }
        for item in items
    ]


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="ascii", errors="ignore").strip()


def _ccik_param_snapshot(ezfio: Path) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for name in TRACKED_QP_FIELDS:
        text = _read_text_if_exists(ezfio / "ccik" / name)
        snapshot[name] = None if text is None else text
    return snapshot


def _snapshot_diff(before: dict[str, object], after: dict[str, object]) -> dict[str, dict[str, object]]:
    diff: dict[str, dict[str, object]] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) == after.get(key):
            continue
        diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


def _active_space_snapshot(ezfio: Path) -> dict[str, object]:
    active = load_prepared_active_space(ezfio)
    return asdict(active)


def _make_small_parity_python_params(
    *,
    steps: int,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
) -> CCIKParams:
    return CCIKParams(
        m=max(2, int(steps) + 1),
        compression_policy="fixed_budget",
        fixed_store_keep=max(1, int(fixed_store_keep)),
        fixed_propagate_keep=max(1, int(fixed_propagate_keep)),
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        selector_depth=1,
        generator_keep=max(1, int(fixed_propagate_keep)),
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=max(1, int(fixed_propagate_keep)),
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=max(1, int(fixed_propagate_keep)),
        exact_hq_diagnostics=False,
        verbose=False,
        orth_tol=1.0e-12,
    )


def _make_small_parity_python_thick_params(
    *,
    m_cycle: int,
    ncycles: int,
    nroot: int,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
) -> CCIKThickRestartParams:
    return CCIKThickRestartParams(
        m_cycle=max(2, int(m_cycle)),
        ncycles=max(1, int(ncycles)),
        nroot=max(1, int(nroot)),
        tol=1.0e-12,
        compression_policy="fixed_budget",
        fixed_store_keep=max(1, int(fixed_store_keep)),
        fixed_propagate_keep=max(1, int(fixed_propagate_keep)),
        candidate_mode="connected_sd",
        selector_mode="abs_residual",
        selector_depth=1,
        generator_keep=max(1, int(fixed_propagate_keep)),
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=max(1, int(fixed_propagate_keep)),
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=max(1, int(fixed_propagate_keep)),
        exact_hq_diagnostics=False,
        verbose=False,
        orth_tol=1.0e-12,
    )


def _make_small_parity_qp_params(
    *,
    steps: int,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    qp_mode: str = "baseline",
    m_cycle: int = SMALL_PARITY_THICK_MCYCLE,
    ncycles: int = SMALL_PARITY_THICK_NCYCLES,
    nroot: int = SMALL_PARITY_THICK_NROOT,
) -> QPCCIKParams:
    bridge = fixed_budget_bridge_kwargs(
        fixed_store_keep=max(1, int(fixed_store_keep)),
        fixed_propagate_keep=max(1, int(fixed_propagate_keep)),
        external_keep=max(2048, int(fixed_store_keep)),
    )
    if qp_mode == "baseline":
        basis_size = max(2, int(steps) + 1)
        max_cycle = max(1, int(steps))
        restart_m_cycle = 1
        restart_nroot = 0
    elif qp_mode == "thick":
        basis_size = max(2, int(m_cycle))
        max_cycle = max(1, int(ncycles))
        restart_m_cycle = max(2, int(m_cycle))
        restart_nroot = max(1, int(nroot))
    else:
        raise ValueError(f"Unknown small parity QP mode: {qp_mode!r}")
    return QPCCIKParams(
        ccik_basis_size=basis_size,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_max_cycle=max_cycle,
        ccik_restart_m_cycle=restart_m_cycle,
        ccik_restart_nroot=restart_nroot,
        ccik_reference_total_energy=0.0,
        ccik_print_iteration_error=True,
        ccik_verbose=False,
        **bridge,
    )


def _python_step_payload(
    *,
    step: int,
    q_k: Any,
    q_next: Any,
    basis: list[Any],
    ws: Any,
    diag: Any,
    artifacts: Any,
) -> dict[str, object]:
    qbasis_entry = _sorted_sparse_items(q_k, limit=None)
    projected_top = _sorted_sparse_items(ws.v_proj, limit=None)
    orth_top = _sorted_sparse_items(artifacts.orthogonalized_residual, limit=None)
    qnext_top = _sorted_sparse_items(q_next, limit=None)
    return {
        "step": int(step),
        "candidate_mode": str(diag.candidate_mode_used),
        "selector_mode": str(diag.selector_mode_used),
        "compression_policy": str(diag.compression_policy_used),
        "current_support_unioned": bool(getattr(artifacts, "candidate_pool", set()) >= q_k.support()),
        "rayleigh": float(inner_sparse(q_k, ws.v_proj)),
        "lowest_ritz_electronic_energy": None,
        "basis_overlap_max": float(diag.max_basis_overlap),
        "q_k": {
            "size": int(len(q_k.coeffs)),
            "norm": float(sparse_norm(q_k)),
            "top": qbasis_entry,
        },
        "qbasis_entry": {
            "size": int(len(q_k.coeffs)),
            "top": qbasis_entry,
        },
        "propagation_support": {
            "size": int(len(ws.propagation_support)),
            "top": _restrict_sparse_items(q_k, set(ws.propagation_support), limit=None),
        },
        "parent_support": {
            "size": int(len(artifacts.parent_support)),
            "top": _restrict_sparse_items(q_k, set(artifacts.parent_support), limit=None),
        },
        "candidate_pool": {
            "size": int(len(artifacts.candidate_pool)),
            "top": _sorted_det_keys(set(artifacts.candidate_pool), limit=None),
        },
        "projected_hq": {
            "size": int(len(ws.v_proj.coeffs)),
            "norm": float(sparse_norm(ws.v_proj)),
            "top": projected_top,
        },
        "hq_basis_entry": {
            "size": int(len(ws.v_proj.coeffs)),
            "top": projected_top,
        },
        "orthogonalized_residual": {
            "size": int(len(artifacts.orthogonalized_residual.coeffs)),
            "norm": float(sparse_norm(artifacts.orthogonalized_residual)),
            "top": orth_top,
        },
        "q_next": {
            "size": int(len(q_next.coeffs)),
            "norm": float(sparse_norm(q_next)),
            "top": qnext_top,
        },
    }


def _run_python_small_step_trace(
    *,
    fcidump_path: Path,
    outdir: Path,
    steps: int,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    electronic_shift_eh: float = 0.0,
) -> dict[str, object]:
    hdata = _load_fcidump_hamiltonian(fcidump_path)
    h1 = np.asarray(hdata["h1"], dtype=float)
    eri8 = hdata["eri"]
    norb = int(hdata["norb"])
    nelec = tuple(hdata["nelec"])
    if abs(float(electronic_shift_eh)) > 0.0:
        h1 = _apply_constant_shift_to_fixed_nelec_h1(
            h1,
            nelec=nelec,
            shift=float(electronic_shift_eh),
        )
    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)
    params = _make_small_parity_python_params(
        steps=steps,
        fixed_store_keep=fixed_store_keep,
        fixed_propagate_keep=fixed_propagate_keep,
    )
    q0 = make_hf_sparse_vector(norb=norb, nelec=nelec)
    basis = [q0]
    rows: list[dict[str, object]] = []
    for step in range(1, int(steps) + 1):
        q_k = basis[-1]
        q_next, diag, ws, artifacts = analyze_krylov_step(
            q_k,
            basis,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        row = _python_step_payload(step=step, q_k=q_k, q_next=q_next, basis=basis, ws=ws, diag=diag, artifacts=artifacts)
        projected = solve_projected_ritz(
            Q=basis,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        row["lowest_ritz_electronic_energy"] = float(projected.evals[0])
        rows.append(row)
        if not q_next.coeffs:
            break
        basis.append(q_next)
    payload = {
        "oracle": "python",
        "candidate_mode": "connected_sd",
        "omp_threads": 1,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "basis_size": int(len(basis)),
        "exact_total_energy_eh": float(hdata["exact_total_energy_eh"]),
        "electronic_shift_eh": float(electronic_shift_eh),
        "steps": rows,
        "params": asdict(params),
    }
    _write_json(outdir / "python_steps.json", payload)
    return payload


def _python_restart_root_payload(
    *,
    cycle: int,
    root: int,
    ritz_eigenvalue: float,
    raw: SparseCIVector,
    compressed: SparseCIVector,
    normalized: SparseCIVector,
    post_orth: SparseCIVector,
    final: SparseCIVector,
) -> dict[str, object]:
    return {
        "cycle": int(cycle),
        "root": int(root),
        "ritz_eigenvalue_electronic": float(ritz_eigenvalue),
        "raw_restart_vector": {
            "size": int(len(raw.coeffs)),
            "norm": float(sparse_norm(raw)),
            "top": _sorted_sparse_items(raw, limit=None),
        },
        "compressed_restart_vector": {
            "size": int(len(compressed.coeffs)),
            "norm": float(sparse_norm(compressed)),
            "top": _sorted_sparse_items(compressed, limit=None),
        },
        "normalized_restart_vector": {
            "size": int(len(normalized.coeffs)),
            "norm": float(sparse_norm(normalized)),
            "top": _sorted_sparse_items(normalized, limit=None),
        },
        "post_orthogonalization_restart_vector": {
            "size": int(len(post_orth.coeffs)),
            "norm": float(sparse_norm(post_orth)),
            "top": _sorted_sparse_items(post_orth, limit=None),
        },
        "final_kept_restart_basis": {
            "size": int(len(final.coeffs)),
            "norm": float(sparse_norm(final)),
            "top": _sorted_sparse_items(final, limit=None),
        },
    }


def _run_python_small_thick_trace(
    *,
    fcidump_path: Path,
    outdir: Path,
    fixed_store_keep: int,
    fixed_propagate_keep: int,
    m_cycle: int,
    ncycles: int,
    nroot: int,
    electronic_shift_eh: float = 0.0,
) -> dict[str, object]:
    hdata = _load_fcidump_hamiltonian(fcidump_path)
    h1 = np.asarray(hdata["h1"], dtype=float)
    eri8 = hdata["eri"]
    norb = int(hdata["norb"])
    nelec = tuple(hdata["nelec"])
    if abs(float(electronic_shift_eh)) > 0.0:
        h1 = _apply_constant_shift_to_fixed_nelec_h1(
            h1,
            nelec=nelec,
            shift=float(electronic_shift_eh),
        )
    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)
    params = _make_small_parity_python_thick_params(
        m_cycle=m_cycle,
        ncycles=ncycles,
        nroot=nroot,
        fixed_store_keep=fixed_store_keep,
        fixed_propagate_keep=fixed_propagate_keep,
    )
    starts: list[SparseCIVector] = [make_hf_sparse_vector(norb=norb, nelec=nelec)]
    step_rows: list[dict[str, object]] = []
    restart_cycles: list[dict[str, object]] = []
    global_step = 0

    for cycle_idx in range(1, int(params.ncycles) + 1):
        Q = gram_schmidt_list(starts, tol=float(getattr(params, "orth_tol", 1.0e-12)))
        if not Q:
            break
        frontier = list(range(len(Q)))
        head = 0

        while len(Q) < int(params.m_cycle) and head < len(frontier):
            parent_idx = frontier[head]
            head += 1
            q_k = Q[parent_idx]
            q_next, diag, ws, artifacts = analyze_krylov_step(
                q_k,
                Q,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
            global_step += 1
            row = _python_step_payload(step=global_step, q_k=q_k, q_next=q_next, basis=Q, ws=ws, diag=diag, artifacts=artifacts)
            projected = solve_projected_ritz(
                Q=Q,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
            row["cycle"] = int(cycle_idx)
            row["iter_in_cycle"] = int(len(Q) + 1)
            row["parent_idx"] = int(parent_idx + 1)
            row["lowest_ritz_electronic_energy"] = float(projected.evals[0])
            step_rows.append(row)
            if not q_next.coeffs:
                break
            Q.append(q_next)
            frontier.append(len(Q) - 1)

        projected = solve_projected_ritz(
            Q=Q,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        evals = projected.evals
        X = projected.eigvecs
        nroot_eff = min(int(params.nroot), X.shape[1])
        root_rows: list[dict[str, object]] = []
        pre_gs: list[SparseCIVector] = []
        gs_basis: list[SparseCIVector] = []

        for root_idx in range(nroot_eff):
            coeffs = X[:, root_idx]
            raw = linear_combination_sparse((float(coeffs[i]), Q[i]) for i in range(len(Q)))
            compressed = _compress_with_stage1(raw, int(params.nkeep), params)
            normalized = normalize_sparse(compressed)
            pre_gs.append(normalized)
            post_orth = orthogonalize_sparse(normalized, gs_basis)
            if sparse_norm(post_orth) > float(getattr(params, "orth_tol", 1.0e-12)):
                final = normalize_sparse(post_orth)
                gs_basis.append(final)
            else:
                final = SparseCIVector({})
            root_rows.append(
                _python_restart_root_payload(
                    cycle=cycle_idx,
                    root=root_idx + 1,
                    ritz_eigenvalue=float(evals[root_idx]),
                    raw=raw,
                    compressed=compressed,
                    normalized=normalized,
                    post_orth=post_orth,
                    final=final,
                )
            )

        starts = list(gs_basis)
        if not starts and pre_gs:
            starts = [pre_gs[0]]
        restart_cycles.append(
            {
                "cycle": int(cycle_idx),
                "basis_size": int(len(Q)),
                "lowest_ritz_electronic_energy": float(evals[0]),
                "ritz_eigenvalues_electronic": [float(val) for val in evals[:nroot_eff]],
                "roots": root_rows,
                "final_restart_basis": [
                    {
                        "root": int(idx + 1),
                        "size": int(len(vec.coeffs)),
                        "norm": float(sparse_norm(vec)),
                        "top": _sorted_sparse_items(vec, limit=None),
                    }
                    for idx, vec in enumerate(starts)
                ],
            }
        )

    step_payload = {
        "oracle": "python_thick",
        "candidate_mode": "connected_sd",
        "omp_threads": 1,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "m_cycle": int(m_cycle),
        "ncycles": int(ncycles),
        "nroot": int(nroot),
        "exact_total_energy_eh": float(hdata["exact_total_energy_eh"]),
        "electronic_shift_eh": float(electronic_shift_eh),
        "steps": step_rows,
        "params": asdict(params),
    }
    restart_payload = {
        "oracle": "python_thick",
        "candidate_mode": "connected_sd",
        "omp_threads": 1,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "m_cycle": int(m_cycle),
        "ncycles": int(ncycles),
        "nroot": int(nroot),
        "cycles": restart_cycles,
        "params": asdict(params),
    }
    _write_json(outdir / "python_steps.json", step_payload)
    _write_json(outdir / "python_restart.json", restart_payload)
    return {"steps": step_payload, "restart": restart_payload}


def _parse_qp_trace_with_steps(trace_path: Path, *, ncore: int, ncas: int) -> dict[int, dict[str, object]]:
    parsed: dict[int, dict[str, object]] = {}
    basis_entries: list[dict[str, object]] = []
    current_basis_entry: dict[str, object] | None = None
    current_step_entry: dict[str, object] | None = None
    current_vec_name: str | None = None
    current_vec_store: dict[str, list[dict[str, object]]] | None = None
    last_iter_error: dict[str, object] | None = None
    last_ritz_h: dict[str, object] | None = None
    restart_reset_pending = False
    next_step_id = 0
    pending_step_vectors: dict[str, list[dict[str, object]]] = {}

    basis_vec_names = {
        "load_input",
        "load_roundtrip",
        "propagation_source",
        "parent_source",
        "projected_hq_full",
        "projected_hq_internal",
        "projected_hq_external",
        "projected_hq",
    }
    pre_next_step_vec_names = {"orth_residual"}
    post_next_step_vec_names = {"score_vec", "q_next"}

    def _new_basis_entry(load_parts: list[str]) -> dict[str, object]:
        return {
            "vectors": {},
            "load_meta": {
                "compression_policy": load_parts[1],
                "loaded_size": int(load_parts[2]),
                "nload": int(load_parts[3]),
            },
        }

    def _hq_meta(hq_parts: list[str]) -> dict[str, object]:
        return {
            "projected_call_index": int(hq_parts[1]),
            "selector_mode": hq_parts[2],
            "matvec_backend": hq_parts[3],
            "compression_policy": hq_parts[4] if len(hq_parts) >= 14 else "topk",
            "fullq_support_size": int(hq_parts[5] if len(hq_parts) >= 14 else hq_parts[4]),
            "parent_count": int(hq_parts[6] if len(hq_parts) >= 14 else hq_parts[5]),
            "generated_before_dedup": int(hq_parts[7] if len(hq_parts) >= 14 else hq_parts[6]),
            "generated_unique": int(hq_parts[8] if len(hq_parts) >= 14 else hq_parts[7]),
            "rayleigh": float(hq_parts[9] if len(hq_parts) >= 14 else hq_parts[8]),
            "source_used_fallback": hq_parts[10] == "T" if len(hq_parts) >= 14 else False,
            "source_dropped_norm": float(hq_parts[11]) if len(hq_parts) >= 14 else 0.0,
            "projected_full_norm": float(hq_parts[12]) if len(hq_parts) >= 14 else None,
            "projected_final_norm": float(hq_parts[13]) if len(hq_parts) >= 14 else float(hq_parts[10]),
        }

    def _next_meta(next_parts: list[str]) -> dict[str, object]:
        return {
            "parent_idx": int(next_parts[1]),
            "selector_mode": next_parts[2],
            "compression_policy": next_parts[3] if len(next_parts) >= 15 else "topk",
            "tmp_keep": int(next_parts[4] if len(next_parts) >= 15 else next_parts[3]),
            "final_keep": int(next_parts[5] if len(next_parts) >= 15 else next_parts[4]),
            "preserve_cap": int(next_parts[6] if len(next_parts) >= 15 else next_parts[5]),
            "store_used_fallback": next_parts[7] == "T" if len(next_parts) >= 15 else False,
            "q_norm": float(next_parts[8] if len(next_parts) >= 15 else next_parts[6]),
            "projected_norm": float(next_parts[9] if len(next_parts) >= 15 else next_parts[7]),
            "v_temp_norm": float(next_parts[10] if len(next_parts) >= 15 else next_parts[8]),
            "discarded_weight": float(next_parts[11] if len(next_parts) >= 15 else next_parts[9]),
            "orthogonalized_residual_norm": float(next_parts[12] if len(next_parts) >= 15 else next_parts[10]),
            "post_compression_residual_norm": float(next_parts[13] if len(next_parts) >= 15 else next_parts[11]),
            "store_dropped_norm": float(next_parts[14] if len(next_parts) >= 15 else 0.0),
            "max_basis_overlap": float(next_parts[15]) if len(next_parts) >= 16 else None,
        }

    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "CCIK_TRACE_RESTART":
            current_vec_name = None
            current_vec_store = None
            current_basis_entry = None
            current_step_entry = None
            if len(parts) >= 2 and parts[1] == "final":
                restart_reset_pending = True
            continue
        if tag == "CCIK_TRACE_LOAD":
            if restart_reset_pending:
                basis_entries = []
                restart_reset_pending = False
            current_basis_entry = _new_basis_entry(parts)
            basis_entries.append(current_basis_entry)
            current_step_entry = None
            current_vec_name = None
            current_vec_store = None
        elif tag == "CCIK_TRACE_HQ":
            if current_basis_entry is None:
                current_basis_entry = {"vectors": {}}
                basis_entries.append(current_basis_entry)
            current_basis_entry["hq_meta"] = _hq_meta(parts)
            current_step_entry = None
            current_vec_name = None
            current_vec_store = None
        elif tag == "CCIK_TRACE_RITZ_H":
            last_ritz_h = {
                "nkeep": int(parts[1]),
                "lowest_electronic_energy_eh": float(parts[2]),
                "second_electronic_energy_eh": float(parts[3]),
                "diag_floor": float(parts[4]),
            }
            current_vec_name = None
            current_vec_store = None
        elif tag == "CCIK_TRACE_ITERERR":
            last_iter_error = {
                "mode": parts[1],
                "cycle": int(parts[2]),
                "iter": int(parts[3]),
                "total_energy_eh": float(parts[4]),
                "reference_total_energy_eh": float(parts[5]),
                "delta_mEh": float(parts[6]),
            }
            pending_step_vectors = {}
            current_vec_name = None
            current_vec_store = None
        elif tag == "CCIK_TRACE_NEXT":
            parent_idx = int(parts[1])
            source_entry = basis_entries[parent_idx - 1] if 1 <= parent_idx <= len(basis_entries) else {"vectors": {}}
            next_step_id += 1
            current_step_entry = {
                "source_entry": source_entry,
                "next_meta": _next_meta(parts),
                "iter_error": last_iter_error,
                "ritz_h": last_ritz_h,
                "vectors": {name: list(items) for name, items in pending_step_vectors.items()},
            }
            parsed[next_step_id] = current_step_entry
            current_vec_name = None
            current_vec_store = None
            last_iter_error = None
            last_ritz_h = None
            pending_step_vectors = {}
        elif tag == "CCIK_TRACE_VEC":
            vec_name = parts[1]
            if _RESTART_TAG_RE.match(vec_name):
                current_vec_name = None
                current_vec_store = None
                continue
            if vec_name in post_next_step_vec_names and current_step_entry is not None:
                current_vec_store = current_step_entry.setdefault("vectors", {})
            elif vec_name in pre_next_step_vec_names:
                current_vec_store = pending_step_vectors
            elif vec_name in basis_vec_names and current_basis_entry is not None:
                current_vec_store = current_basis_entry.setdefault("vectors", {})
            elif current_step_entry is not None:
                current_vec_store = current_step_entry.setdefault("vectors", {})
            elif current_basis_entry is not None:
                current_vec_store = current_basis_entry.setdefault("vectors", {})
            else:
                current_vec_store = None
            current_vec_name = vec_name if current_vec_store is not None else None
            if current_vec_store is not None and current_vec_name is not None:
                current_vec_store.setdefault(current_vec_name, [])
        elif tag == "CCIK_TRACE_COEF" and current_vec_name is not None and current_vec_store is not None:
            vec_name = parts[1]
            if vec_name != current_vec_name:
                current_vec_name = vec_name
                current_vec_store.setdefault(current_vec_name, [])
            current_vec_store[current_vec_name].append(
                {
                    "rank": int(parts[2]),
                    "stra": _map_qp_det_to_active(int(parts[3]), ncore=ncore, ncas=ncas),
                    "strb": _map_qp_det_to_active(int(parts[4]), ncore=ncore, ncas=ncas),
                    "coef": float(parts[5]),
                    "abs_coef": float(parts[6]),
                }
            )
    return parsed


_RESTART_TAG_RE = re.compile(r"^restart_fb_(?P<stage>[a-z_]+)_c(?P<cycle>\d+)_r(?P<root>\d+)$")


def _parse_qp_restart_trace(trace_path: Path, *, ncore: int, ncas: int) -> list[dict[str, object]]:
    cycles: dict[int, dict[str, object]] = {}
    current_restart: tuple[int, int, str] | None = None
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "CCIK_TRACE_RESTART":
            stage = str(parts[1])
            cycle = int(parts[2])
            root = int(parts[3])
            cycle_row = cycles.setdefault(cycle, {"cycle": cycle, "roots": {}, "ritz_eigenvalues_electronic": []})
            root_row = cycle_row["roots"].setdefault(root, {"cycle": cycle, "root": root, "stages": {}})
            root_row["stages"][stage] = {
                "ritz_eigenvalue_electronic": float(parts[4]),
                "size": int(parts[5]),
                "ok": parts[6] == "T",
            }
            if root == 1 and stage == "raw":
                cycle_row["ritz_eigenvalues_electronic"] = []
            stage_eigs = cycle_row.setdefault("ritz_eigenvalues_electronic", [])
            while len(stage_eigs) < root:
                stage_eigs.append(None)
            stage_eigs[root - 1] = float(parts[4])
            current_restart = None
        elif tag == "CCIK_TRACE_VEC":
            match = _RESTART_TAG_RE.match(parts[1])
            if match is None:
                current_restart = None
                continue
            current_restart = (int(match.group("cycle")), int(match.group("root")), str(match.group("stage")))
            cycle_row = cycles.setdefault(current_restart[0], {"cycle": current_restart[0], "roots": {}, "ritz_eigenvalues_electronic": []})
            root_row = cycle_row["roots"].setdefault(current_restart[1], {"cycle": current_restart[0], "root": current_restart[1], "stages": {}})
            root_row["stages"].setdefault(current_restart[2], {})["top"] = []
        elif tag == "CCIK_TRACE_COEF" and current_restart is not None:
            cycle, root, stage = current_restart
            cycle_row = cycles.setdefault(cycle, {"cycle": cycle, "roots": {}, "ritz_eigenvalues_electronic": []})
            root_row = cycle_row["roots"].setdefault(root, {"cycle": cycle, "root": root, "stages": {}})
            root_row["stages"].setdefault(stage, {}).setdefault("top", []).append(
                {
                    "rank": int(parts[2]),
                    "stra": _map_qp_det_to_active(int(parts[3]), ncore=ncore, ncas=ncas),
                    "strb": _map_qp_det_to_active(int(parts[4]), ncore=ncore, ncas=ncas),
                    "coef": float(parts[5]),
                    "abs_coef": float(parts[6]),
                }
            )

    out: list[dict[str, object]] = []
    for cycle in sorted(cycles):
        cycle_row = cycles[cycle]
        roots: list[dict[str, object]] = []
        for root in sorted(cycle_row["roots"]):
            stages = cycle_row["roots"][root]["stages"]
            roots.append(
                {
                    "cycle": int(cycle),
                    "root": int(root),
                    "ritz_eigenvalue_electronic": None
                    if root - 1 >= len(cycle_row["ritz_eigenvalues_electronic"])
                    else cycle_row["ritz_eigenvalues_electronic"][root - 1],
                    "raw_restart_vector": {
                        "size": int(stages.get("raw", {}).get("size", 0)),
                        "norm": None,
                        "top": list(stages.get("raw", {}).get("top", [])),
                    },
                    "compressed_restart_vector": {
                        "size": int(stages.get("compressed", {}).get("size", 0)),
                        "norm": None,
                        "top": list(stages.get("compressed", {}).get("top", [])),
                    },
                    "normalized_restart_vector": {
                        "size": int(stages.get("normalized", {}).get("size", 0)),
                        "norm": None,
                        "top": list(stages.get("normalized", {}).get("top", [])),
                    },
                    "post_orthogonalization_restart_vector": {
                        "size": int(stages.get("postorth", {}).get("size", 0)),
                        "norm": None,
                        "top": list(stages.get("postorth", {}).get("top", [])),
                    },
                    "final_kept_restart_basis": {
                        "size": int(stages.get("final", {}).get("size", 0)),
                        "norm": None,
                        "top": list(stages.get("final", {}).get("top", [])),
                    },
                }
            )
        out.append(
            {
                "cycle": int(cycle),
                "ritz_eigenvalues_electronic": [
                    float(val) for val in cycle_row["ritz_eigenvalues_electronic"] if val is not None
                ],
                "roots": roots,
            }
        )
    return out


def _qp_step_payload(step: int, parsed_step: dict[str, object], *, include_current_support: bool) -> dict[str, object]:
    source_entry = dict(parsed_step.get("source_entry", {}))
    source_vectors = dict(source_entry.get("vectors", {}))
    source_load_meta = dict(source_entry.get("load_meta", {}))
    source_hq_meta = dict(source_entry.get("hq_meta", {}))
    vectors = dict(parsed_step.get("vectors", {}))
    qk_top = list(source_vectors.get("load_input", []))
    projected_top = list(source_vectors.get("projected_hq", []))
    orth_top = list(vectors.get("orth_residual", []))
    qnext_top = list(vectors.get("q_next", []))
    prop_top = list(source_vectors.get("propagation_source", []))
    parent_top = list(source_vectors.get("parent_source", []))
    return {
        "step": int(step),
        "cycle": parsed_step.get("iter_error", {}).get("cycle"),
        "iter_in_cycle": parsed_step.get("iter_error", {}).get("iter"),
        "parent_idx": parsed_step.get("next_meta", {}).get("parent_idx"),
        "candidate_mode": "connected_sd",
        "selector_mode": parsed_step.get("next_meta", {}).get("selector_mode"),
        "compression_policy": parsed_step.get("next_meta", {}).get("compression_policy"),
        "current_support_unioned": bool(include_current_support),
        "rayleigh": source_hq_meta.get("rayleigh"),
        "lowest_ritz_electronic_energy": parsed_step.get("ritz_h", {}).get("lowest_electronic_energy_eh"),
        "lowest_ritz_total_energy_eh": parsed_step.get("iter_error", {}).get("total_energy_eh"),
        "basis_overlap_max": parsed_step.get("next_meta", {}).get("max_basis_overlap"),
        "q_k": {
            "size": int(source_load_meta.get("loaded_size", len(qk_top))),
            "norm": parsed_step.get("next_meta", {}).get("q_norm"),
            "top": qk_top,
        },
        "qbasis_entry": {
            "size": int(source_load_meta.get("loaded_size", len(qk_top))),
            "top": qk_top,
        },
        "propagation_support": {
            "size": int(len(prop_top)),
            "top": prop_top,
        },
        "parent_support": {
            "size": int(source_hq_meta.get("parent_count", len(parent_top))),
            "top": parent_top,
        },
        "candidate_pool": {
            "size": source_hq_meta.get("generated_unique"),
            "top": [],
        },
        "projected_hq": {
            "size": int(len(projected_top)),
            "norm": source_hq_meta.get("projected_final_norm", parsed_step.get("next_meta", {}).get("projected_norm")),
            "top": projected_top,
        },
        "hq_basis_entry": {
            "size": int(len(projected_top)),
            "top": projected_top,
        },
        "orthogonalized_residual": {
            "size": int(len(orth_top)),
            "norm": parsed_step.get("next_meta", {}).get("orthogonalized_residual_norm"),
            "top": orth_top,
        },
        "q_next": {
            "size": int(len(qnext_top)),
            "norm": None if not qnext_top else 1.0,
            "top": qnext_top,
        },
    }


def _compare_restart_cycles(
    python_cycles: list[dict[str, object]],
    qp_cycles: list[dict[str, object]],
) -> dict[str, object]:
    cycle_rows: list[dict[str, object]] = []
    first_mismatch: dict[str, object] | None = None
    stage_keys = (
        "raw_restart_vector",
        "compressed_restart_vector",
        "normalized_restart_vector",
        "post_orthogonalization_restart_vector",
        "final_kept_restart_basis",
    )
    ncycles = min(len(python_cycles), len(qp_cycles))
    for idx in range(ncycles):
        py_cycle = python_cycles[idx]
        qp_cycle = qp_cycles[idx]
        roots_rows: list[dict[str, object]] = []
        py_roots = list(py_cycle.get("roots", []))
        qp_roots = list(qp_cycle.get("roots", []))
        nroots = min(len(py_roots), len(qp_roots))
        for root_idx in range(nroots):
            py_root = py_roots[root_idx]
            qp_root = qp_roots[root_idx]
            root_row: dict[str, object] = {
                "cycle": int(py_cycle["cycle"]),
                "root": int(py_root["root"]),
                "ritz_eigenvalue_electronic": _compare_scalar(
                    py_root.get("ritz_eigenvalue_electronic"),
                    qp_root.get("ritz_eigenvalue_electronic"),
                    tol=1.0e-9,
                ),
            }
            for stage_key in stage_keys:
                root_row[stage_key] = _compare_vector_maps(
                    _vector_from_trace_items(list(py_root[stage_key]["top"])),
                    _vector_from_trace_items(list(qp_root[stage_key]["top"])),
                    allow_global_sign_flip=True,
                )
                if first_mismatch is None:
                    stage_diff = root_row[stage_key]
                    if (not stage_diff["exact_support_match"]) or float(stage_diff["max_abs_coef_diff"]) > 1.0e-9:
                        first_mismatch = {
                            "cycle": int(py_cycle["cycle"]),
                            "root": int(py_root["root"]),
                            "stage": stage_key,
                            "bug_class": "thick_restart_semantics",
                            "details": stage_diff,
                        }
            if first_mismatch is None and not root_row["ritz_eigenvalue_electronic"]["matches"]:
                first_mismatch = {
                    "cycle": int(py_cycle["cycle"]),
                    "root": int(py_root["root"]),
                    "stage": "ritz_eigenvalue_electronic",
                    "bug_class": "ritz_solver",
                    "details": root_row["ritz_eigenvalue_electronic"],
                }
            roots_rows.append(root_row)
        cycle_rows.append(
            {
                "cycle": int(py_cycle["cycle"]),
                "ritz_eigenvalues_electronic": _compare_scalar(
                    py_cycle.get("lowest_ritz_electronic_energy"),
                    qp_cycle.get("ritz_eigenvalues_electronic", [None])[0] if qp_cycle.get("ritz_eigenvalues_electronic") else None,
                    tol=1.0e-9,
                ),
                "roots": roots_rows,
            }
        )
    return {
        "cycles": cycle_rows,
        "first_mismatch": first_mismatch,
    }


def _compare_stage_vectors(py_row: dict[str, object], qp_row: dict[str, object], key: str) -> dict[str, object]:
    return _compare_vector_maps(
        _vector_from_trace_items(list(py_row[key]["top"])),
        _vector_from_trace_items(list(qp_row[key]["top"])),
        allow_global_sign_flip=True,
    )


def _localize_first_mismatch(step_diffs: list[dict[str, object]]) -> dict[str, object]:
    stages = (
        ("propagation_support", "provider_path_or_propagation_support"),
        ("parent_support", "provider_path_or_parent_support"),
        ("projected_hq", "projected_hq_builder"),
        ("orthogonalized_residual", "next_vector_orthogonalization"),
        ("q_next", "next_vector_storage"),
        ("lowest_ritz_energy", "ritz_solver"),
    )
    for row in step_diffs:
        qnext_payload = row.get("q_next", {})
        qnext_exact = bool(qnext_payload.get("exact_support_match", False)) and float(qnext_payload.get("max_abs_coef_diff", 0.0)) <= 1.0e-9
        for stage_name, bug_class in stages:
            payload = row.get(stage_name, {})
            if stage_name == "lowest_ritz_energy":
                if not payload.get("matches", False):
                    return {"step": int(row["step"]), "stage": stage_name, "bug_class": bug_class, "details": payload}
                continue
            if stage_name in {"projected_hq", "orthogonalized_residual"} and qnext_exact:
                if float(payload.get("max_abs_coef_diff", 0.0)) <= 1.0e-9:
                    continue
            if not payload.get("exact_support_match", False):
                return {"step": int(row["step"]), "stage": stage_name, "bug_class": bug_class, "details": payload}
            if float(payload.get("max_abs_coef_diff", 0.0)) > 1.0e-9:
                return {"step": int(row["step"]), "stage": stage_name, "bug_class": bug_class, "details": payload}
    return {"step": None, "stage": None, "bug_class": "no_mismatch_detected", "details": None}


def _doc_compare(golden_row: dict[str, object], e_total_eh: float | None) -> dict[str, object]:
    if e_total_eh is None:
        return {
            "e_total_mismatch_eh": None,
            "delta_vs_docs_mEh": None,
            "abs_delta_vs_docs_mEh": None,
            "matches_docs": False,
        }
    energy_mismatch = float(e_total_eh) - float(golden_row["e_total_eh"])
    delta_vs_ref = float((float(e_total_eh) - float(golden_row["reference_total_energy_eh"])) * 1000.0)
    delta_mismatch = delta_vs_ref - float(golden_row["delta_vs_ref_mEh"])
    return {
        "e_total_mismatch_eh": energy_mismatch,
        "delta_vs_docs_mEh": delta_mismatch,
        "abs_delta_vs_docs_mEh": abs(delta_mismatch),
        "matches_docs": abs(energy_mismatch) <= ENERGY_TOL_EH and abs(delta_mismatch) <= DELTA_TOL_MEH,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _hash_manifest(paths: tuple[Path, ...] | list[Path]) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in paths:
        digest = _sha256_file(path)
        if digest is not None:
            manifest[str(path)] = digest
    return manifest


def _aggregate_manifest_hash(manifest: dict[str, str]) -> str | None:
    if not manifest:
        return None
    payload = "\n".join(f"{key}:{manifest[key]}" for key in sorted(manifest))
    return _sha256_bytes(payload.encode("utf-8"))


def _git_stdout(*args: str) -> str | None:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _git_rev_parse(rev: str) -> str | None:
    stdout = _git_stdout("rev-parse", rev)
    if stdout is None:
        return None
    return stdout.strip() or None


def _git_path_exists(commit: str, path: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{path}"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _historical_bridge_anchor_payload() -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    for raw in HISTORICAL_GIT_ANCHORS:
        existing_paths = [path for path in raw["driver_candidates"] if _git_path_exists(str(raw["commit"]), path)]
        anchors.append(
            {
                "commit": str(raw["commit"]),
                "commit_resolved": _git_rev_parse(str(raw["commit"])),
                "label": str(raw["label"]),
                "bridge_policy": str(raw["bridge_policy"]),
                "comparison_mode": str(raw["comparison_mode"]),
                "driver_candidates": list(raw["driver_candidates"]),
                "existing_driver_paths": existing_paths,
            }
        )
    return anchors


def _runtime_and_plugin_fingerprints(qp_root: Path) -> dict[str, object]:
    qp = discover_qp_paths_from_root(qp_root)
    runtime_files = [
        qp.ccik_binary,
        qp.qp_root / "lib" / "libf77zmq.dylib",
        qp.qp_root / "lib" / "libzmq.5.dylib",
        qp.qp_root / "src" / "hartree_fock" / "scf",
    ]
    runtime_manifest = _hash_manifest(runtime_files)
    repo_manifest = _hash_manifest(list(REPO_PLUGIN_FINGERPRINT_FILES))
    return {
        "comparison_mode": "bridge-only historical replay",
        "repo_git_head": _git_rev_parse("HEAD"),
        "repo_plugin_file_sha256": repo_manifest,
        "repo_plugin_tree_sha256": _aggregate_manifest_hash(repo_manifest),
        "runtime_file_sha256": runtime_manifest,
        "runtime_tree_sha256": _aggregate_manifest_hash(runtime_manifest),
        "git_anchors": _historical_bridge_anchor_payload(),
    }


def _run_qp_ccik(
    *,
    qp_root: Path,
    runtime_parent: Path,
    ezfio: Path,
    log_dir: Path,
    omp_threads: int,
) -> dict[str, object]:
    qp = discover_qp_paths_from_root(qp_root)
    runtime_parent = Path(runtime_parent).resolve()
    ezfio = Path(ezfio).resolve()
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    quoted_rc = shlex.quote(str(qp.quantum_package_rc.resolve()))
    quoted_ccik = shlex.quote(str(qp.ccik_binary.resolve()))
    short_ezfio = shlex.quote(ezfio.name)
    stdout_path = log_dir / "ccik_stdout.log"
    stderr_path = log_dir / "ccik_stderr.log"
    trace_path = log_dir / "ccik_debug_trace.log"
    raw_trace_path = runtime_parent / "ccik_debug_trace.log"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"source {quoted_rc} >/dev/null 2>&1 && qp set_file {short_ezfio} >/dev/null && {quoted_ccik} {short_ezfio}",
        ],
        cwd=str(runtime_parent),
        env={**os.environ, "QP_ROOT": str(qp_root), **qp_parallel_env(int(omp_threads))},
        text=True,
        capture_output=True,
        check=False,
    )
    wall_s = float(time.perf_counter() - t0)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if raw_trace_path.exists():
        shutil.copy2(raw_trace_path, trace_path)
    e_total = pub._extract_optional(proc.stdout, "CCIK_FINAL_TOTAL_ENERGY")
    e_elec = pub._extract_optional(proc.stdout, "CCIK_FINAL_ELECTRONIC_ENERGY")
    basis_size = pub._extract_optional(proc.stdout, "CCIK_FINAL_BASIS")
    failure = None
    if proc.returncode != 0 or e_total is None:
        failure = f"QP thick run failed (rc={proc.returncode})"
    payload = {
        "returncode": int(proc.returncode),
        "wall_s": wall_s,
        "e_total_eh": None if e_total is None else float(e_total),
        "e_electronic_eh": None if e_elec is None else float(e_elec),
        "basis_size_reported": None if basis_size is None else int(basis_size),
        "failure": failure,
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "trace": str(trace_path) if trace_path.exists() else None,
        },
    }
    _write_json(log_dir / "qp_summary.json", payload)
    return payload


def _make_qp_params(case: pub.PublicationCase, golden_row: dict[str, object]) -> Any:
    return _make_qp_params_with_policy(case, golden_row, bridge_policy="current", qp_mode="thick")


def _apply_notebook_bridge_policy(case: pub.PublicationCase, params: Any, golden_row: dict[str, object]) -> Any:
    fixed_store_keep = int(golden_row["fixed_store_keep"])
    fixed_propagate_keep = int(golden_row["fixed_propagate_keep"])
    return replace(
        params,
        ccik_generator_keep=max(256, min(fixed_store_keep, 4 * fixed_propagate_keep)),
        ccik_max_parents_for_expansion=0,
        ccik_selector_keep=max(1024, 2 * fixed_propagate_keep),
        ccik_external_keep=max(2048, min(20_000, 4 * fixed_store_keep)),
        ccik_nkeep=fixed_store_keep,
        ccik_tmp_keep_factor=4,
        ccik_preserve_prev_support=True,
        ccik_preserve_prev_support_cap=min(2048, max(32, fixed_store_keep // 4)),
    )


def _apply_docs_legacy_bridge_policy(case: pub.PublicationCase, params: Any) -> Any:
    return replace(
        params,
        ccik_generator_keep=1,
        ccik_max_parents_for_expansion=0,
        ccik_selector_keep=1,
        ccik_external_keep=int(case.external_keep),
        ccik_external_candidate_factor=1,
        ccik_nkeep=1,
        ccik_tmp_keep_factor=1,
        ccik_preserve_prev_support=False,
        ccik_preserve_prev_support_cap=0,
    )


def _force_qp_baseline_mode(params: Any) -> Any:
    return replace(params, ccik_restart_m_cycle=1, ccik_restart_nroot=0)


def _make_qp_params_with_policy(
    case: pub.PublicationCase,
    golden_row: dict[str, object],
    *,
    bridge_policy: str,
    qp_mode: str,
) -> Any:
    params = pub._make_qp_fixed_budget_params(
        case,
        fixed_store_keep=int(golden_row["fixed_store_keep"]),
        fixed_propagate_keep=int(golden_row["fixed_propagate_keep"]),
        selector_mode="abs_residual",
    )
    if bridge_policy == "notebook":
        params = _apply_notebook_bridge_policy(case, params, golden_row)
    elif bridge_policy == "docs_legacy":
        params = _apply_docs_legacy_bridge_policy(case, params)
    elif bridge_policy != "current":
        raise ValueError(f"Unknown bridge policy: {bridge_policy!r}")

    if qp_mode == "baseline":
        params = _force_qp_baseline_mode(params)
    elif qp_mode != "thick":
        raise ValueError(f"Unknown qp_mode: {qp_mode!r}")
    return params


def _make_python_params(case: pub.PublicationCase, golden_row: dict[str, object]) -> Any:
    return pub._make_python_thick_fixed_budget_params(
        case,
        fixed_store_keep=int(golden_row["fixed_store_keep"]),
        fixed_propagate_keep=int(golden_row["fixed_propagate_keep"]),
        selector_mode="abs_residual",
    )


def _enable_iteration_error_reporting(ezfio: Path, golden_row: dict[str, object]) -> None:
    write_ezfio_value(
        ezfio,
        "ccik/ccik_reference_total_energy",
        float(golden_row["reference_total_energy_eh"]),
    )
    write_ezfio_value(ezfio, "ccik/ccik_print_iteration_error", True)


def _prepare_fresh_publication_setup(case: pub.PublicationCase, qp_root: Path, setup_dir: Path) -> dict[str, Path]:
    if setup_dir.exists():
        shutil.rmtree(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = setup_dir / "n2.xyz"
    ezfio_path = setup_dir / "n2.ezfio"
    fcidump_outdir = setup_dir / "fcidump"
    _create_xyz(xyz_path, make_n2_atom(float(case.distance_ang)), comment=case.title)
    qp = discover_qp_paths_from_root(qp_root)
    run_qp_setup(
        qp,
        basis=case.basis,
        ezfio=ezfio_path,
        xyz=xyz_path,
        ncore=int(case.ncore),
        ncas=int(case.ncas),
        nmo_total=int(case.nmo_total or 0),
    )
    fcidump_path = dsp._generate_qp_fcidump(qp, ezfio_path, fcidump_outdir, case_spec=None)
    return {
        "xyz": xyz_path,
        "ezfio": ezfio_path,
        "fcidump": fcidump_path,
    }


def run_donor_seeded_qp_reproduction(
    *,
    case: pub.PublicationCase,
    golden_row: dict[str, object],
    qp_root: Path,
    outdir: Path,
    omp_threads: int,
    bridge_policy: str = "current",
    qp_mode: str = "thick",
    force: bool = False,
) -> dict[str, object]:
    run_label = f"donor_seeded_qp__{bridge_policy}__{qp_mode}"
    summary_path = outdir / run_label / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    seed_ezfio, seed_fcidump = _require_seed_paths(case)
    run_dir = outdir / run_label
    runtime_parent = run_dir / "runtime"
    runtime_ezfio = runtime_parent / seed_ezfio.name
    if runtime_parent.exists():
        shutil.rmtree(runtime_parent)
    runtime_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(seed_ezfio, runtime_ezfio)
    ensure_ezfio_defaults(runtime_ezfio, qp_root=qp_root)

    seed_snapshot = _ccik_param_snapshot(seed_ezfio)
    prewrite_snapshot = _ccik_param_snapshot(runtime_ezfio)
    params = _make_qp_params_with_policy(case, golden_row, bridge_policy=bridge_policy, qp_mode=qp_mode)
    set_ccik_params(discover_qp_paths_from_root(qp_root), runtime_ezfio, params)
    _enable_iteration_error_reporting(runtime_ezfio, golden_row)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_m", int(params.ccik_basis_size))
    # The small-seed parity harness already exercises the full trace path.
    # Keep the long donor replay lightweight and rely on per-iteration
    # energy-error logging for diagnostics.
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace", False)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace_topn", 32)
    postwrite_snapshot = _ccik_param_snapshot(runtime_ezfio)

    qp_summary = _run_qp_ccik(
        qp_root=qp_root,
        runtime_parent=runtime_parent,
        ezfio=runtime_ezfio,
        log_dir=run_dir,
        omp_threads=omp_threads,
    )
    payload = {
        "path_name": "donor_seeded_qp",
        "bridge_policy": bridge_policy,
        "comparison_mode": "bridge-only historical replay",
        "qp_mode": qp_mode,
        "source_path": str(seed_ezfio),
        "source_fcidump_path": str(seed_fcidump),
        "qp_root": str(qp_root),
        "golden_row": golden_row,
        "runtime_and_plugin_fingerprints": _runtime_and_plugin_fingerprints(qp_root),
        "seed_fcidump_metadata": _fcidump_metadata(seed_fcidump),
        "active_space": _active_space_snapshot(runtime_ezfio),
        "qp_params_requested": asdict(params),
        "seed_ccik_snapshot": seed_snapshot,
        "runtime_ccik_snapshot_before_write": prewrite_snapshot,
        "runtime_ccik_snapshot_after_write": postwrite_snapshot,
        "seed_to_runtime_param_diff": _snapshot_diff(seed_snapshot, postwrite_snapshot),
        "run": qp_summary,
        "docs_comparison": _doc_compare(golden_row, qp_summary["e_total_eh"]),
    }
    _write_json(summary_path, payload)
    return payload


def run_python_seed_fcidump_reproduction(
    *,
    case: pub.PublicationCase,
    golden_row: dict[str, object],
    outdir: Path,
    force: bool = False,
) -> dict[str, object]:
    summary_path = outdir / "python_seed_fcidump" / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    _, seed_fcidump = _require_seed_paths(case)
    run_dir = outdir / "python_seed_fcidump"
    run_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "reference_kind": golden_row["reference_kind"],
        "reference_label": golden_row["reference_label"],
        "total_energy_eh": float(golden_row["reference_total_energy_eh"]),
    }
    result = pub._run_python_thick_job(
        case=case,
        reference=reference,
        fcidump_path=seed_fcidump,
        fixed_store_keep=int(golden_row["fixed_store_keep"]),
        fixed_propagate_keep=int(golden_row["fixed_propagate_keep"]),
        run_dir=run_dir,
        force=force,
        selector_mode="abs_residual",
    )
    payload = {
        "path_name": "python_seed_fcidump",
        "source_fcidump_path": str(seed_fcidump),
        "golden_row": golden_row,
        "seed_fcidump_metadata": _fcidump_metadata(seed_fcidump),
        "python_params_requested": _make_python_params(case, golden_row).__dict__,
        "run": result,
        "docs_comparison": _doc_compare(golden_row, result["e_total_eh"]),
    }
    _write_json(summary_path, payload)
    return payload


def run_fresh_publication_setup_qp_reproduction(
    *,
    case: pub.PublicationCase,
    golden_row: dict[str, object],
    qp_root: Path,
    outdir: Path,
    omp_threads: int,
    bridge_policy: str = "current",
    qp_mode: str = "thick",
    force: bool = False,
) -> dict[str, object]:
    run_label = f"fresh_publication_setup_qp__{bridge_policy}__{qp_mode}"
    summary_path = outdir / run_label / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir = outdir / run_label
    setup_paths = _prepare_fresh_publication_setup(case, qp_root, run_dir / "setup")
    runtime_parent = run_dir / "runtime"
    runtime_ezfio = runtime_parent / setup_paths["ezfio"].name
    if runtime_parent.exists():
        shutil.rmtree(runtime_parent)
    runtime_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(setup_paths["ezfio"], runtime_ezfio)
    ensure_ezfio_defaults(runtime_ezfio, qp_root=qp_root)
    prewrite_snapshot = _ccik_param_snapshot(runtime_ezfio)
    params = _make_qp_params_with_policy(case, golden_row, bridge_policy=bridge_policy, qp_mode=qp_mode)
    set_ccik_params(discover_qp_paths_from_root(qp_root), runtime_ezfio, params)
    _enable_iteration_error_reporting(runtime_ezfio, golden_row)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_m", int(params.ccik_basis_size))
    # The large fresh-setup replay has the same performance profile as the
    # donor replay; keep trace off here as well.
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace", False)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace_topn", 32)
    postwrite_snapshot = _ccik_param_snapshot(runtime_ezfio)

    qp_summary = _run_qp_ccik(
        qp_root=qp_root,
        runtime_parent=runtime_parent,
        ezfio=runtime_ezfio,
        log_dir=run_dir,
        omp_threads=omp_threads,
    )
    payload = {
        "path_name": "fresh_publication_setup_qp",
        "bridge_policy": bridge_policy,
        "comparison_mode": "bridge-only historical replay",
        "qp_mode": qp_mode,
        "setup_paths": {key: str(value) for key, value in setup_paths.items()},
        "qp_root": str(qp_root),
        "golden_row": golden_row,
        "runtime_and_plugin_fingerprints": _runtime_and_plugin_fingerprints(qp_root),
        "fresh_fcidump_metadata": _fcidump_metadata(setup_paths["fcidump"]),
        "active_space": _active_space_snapshot(runtime_ezfio),
        "qp_params_requested": asdict(params),
        "runtime_ccik_snapshot_before_write": prewrite_snapshot,
        "runtime_ccik_snapshot_after_write": postwrite_snapshot,
        "runtime_param_diff": _snapshot_diff(prewrite_snapshot, postwrite_snapshot),
        "run": qp_summary,
        "docs_comparison": _doc_compare(golden_row, qp_summary["e_total_eh"]),
    }
    _write_json(summary_path, payload)
    return payload


def run_small_shared_seed_step_parity(
    *,
    outdir: Path,
    qp_root: Path | None = None,
    fixed_store_keep: int = 256,
    fixed_propagate_keep: int = 64,
    steps: int = 8,
    qp_mode: str = "baseline",
    m_cycle: int = SMALL_PARITY_THICK_MCYCLE,
    ncycles: int = SMALL_PARITY_THICK_NCYCLES,
    nroot: int = SMALL_PARITY_THICK_NROOT,
    trace_topn: int = 100000,
    force: bool = False,
) -> dict[str, object]:
    outdir = Path(outdir)
    summary_path = outdir / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    if not SMALL_SHARED_EZFIO.exists():
        raise FileNotFoundError(f"Missing validated shared small-seed EZFIO: {SMALL_SHARED_EZFIO}")
    if not SMALL_SHARED_FCIDUMP.exists():
        raise FileNotFoundError(f"Missing validated shared small-seed FCIDUMP: {SMALL_SHARED_FCIDUMP}")

    hdata = _load_fcidump_hamiltonian(SMALL_SHARED_FCIDUMP)
    active = load_prepared_active_space(SMALL_SHARED_EZFIO)
    nuclear_repulsion_eh = _read_ezfio_float(SMALL_SHARED_EZFIO, "nuclei/nuclear_repulsion")
    electronic_shift_eh = float(hdata["ecore"]) - float(nuclear_repulsion_eh)
    if qp_mode == "baseline":
        python_payload = {
            "steps": _run_python_small_step_trace(
                fcidump_path=SMALL_SHARED_FCIDUMP,
                outdir=outdir,
                steps=steps,
                fixed_store_keep=fixed_store_keep,
                fixed_propagate_keep=fixed_propagate_keep,
                electronic_shift_eh=electronic_shift_eh,
            ),
            "restart": None,
        }
    elif qp_mode == "thick":
        python_payload = _run_python_small_thick_trace(
            fcidump_path=SMALL_SHARED_FCIDUMP,
            outdir=outdir,
            fixed_store_keep=fixed_store_keep,
            fixed_propagate_keep=fixed_propagate_keep,
            m_cycle=m_cycle,
            ncycles=ncycles,
            nroot=nroot,
            electronic_shift_eh=electronic_shift_eh,
        )
    else:
        raise ValueError(f"Unknown small shared parity mode: {qp_mode!r}")

    qp_root_resolved = _resolve_qp_root_with_binary(qp_root)
    runtime_parent = outdir / "qp_runtime"
    runtime_ezfio = runtime_parent / SMALL_SHARED_EZFIO.name
    if runtime_parent.exists():
        shutil.rmtree(runtime_parent)
    runtime_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SMALL_SHARED_EZFIO, runtime_ezfio)
    ensure_ezfio_defaults(runtime_ezfio, qp_root=qp_root_resolved)
    qp_params = _make_small_parity_qp_params(
        steps=steps,
        fixed_store_keep=fixed_store_keep,
        fixed_propagate_keep=fixed_propagate_keep,
        qp_mode=qp_mode,
        m_cycle=m_cycle,
        ncycles=ncycles,
        nroot=nroot,
    )
    qp_params = replace(qp_params, ccik_reference_total_energy=float(hdata["exact_total_energy_eh"]))
    set_ccik_params(discover_qp_paths_from_root(qp_root_resolved), runtime_ezfio, qp_params)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace", True)
    write_ezfio_value(runtime_ezfio, "ccik/ccik_debug_trace_topn", int(trace_topn))
    write_ezfio_value(runtime_ezfio, "ccik/ccik_reference_total_energy", float(hdata["exact_total_energy_eh"]))
    write_ezfio_value(runtime_ezfio, "ccik/ccik_print_iteration_error", True)

    qp_run = _run_qp_ccik(
        qp_root=qp_root_resolved,
        runtime_parent=runtime_parent,
        ezfio=runtime_ezfio,
        log_dir=outdir / "qp_logs",
        omp_threads=1,
    )
    trace_path = outdir / "qp_logs" / "ccik_debug_trace.log"
    if not trace_path.exists():
        raise RuntimeError(f"QP small-seed parity run did not emit a debug trace: {trace_path}")
    parsed_qp = _parse_qp_trace_with_steps(trace_path, ncore=int(active.ncore), ncas=int(active.ncas))
    parsed_restart = _parse_qp_restart_trace(trace_path, ncore=int(active.ncore), ncas=int(active.ncas))
    qp_rows = [
        _qp_step_payload(step, parsed_qp[step], include_current_support=True)
        for step in sorted(parsed_qp)
        if "next_meta" in parsed_qp[step]
    ]
    qp_payload = {
        "oracle": f"qp_{qp_mode}",
        "candidate_mode": "connected_sd_fullq",
        "omp_threads": 1,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "m_cycle": int(m_cycle),
        "ncycles": int(ncycles),
        "nroot": int(nroot),
        "trace_path": str(trace_path),
        "run": qp_run,
        "params": asdict(qp_params),
        "steps": qp_rows,
    }
    _write_json(outdir / "qp_steps.json", qp_payload)
    qp_restart_payload = {
        "oracle": f"qp_{qp_mode}",
        "candidate_mode": "connected_sd_fullq",
        "omp_threads": 1,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "m_cycle": int(m_cycle),
        "ncycles": int(ncycles),
        "nroot": int(nroot),
        "trace_path": str(trace_path),
        "cycles": parsed_restart,
        "params": asdict(qp_params),
    }
    if qp_mode == "thick":
        _write_json(outdir / "qp_restart.json", qp_restart_payload)

    python_steps = list(python_payload["steps"]["steps"])
    step_diffs: list[dict[str, object]] = []
    for py_row, qp_row in zip(python_steps, qp_rows):
        diff_row = {
            "step": int(py_row["step"]),
            "propagation_support": _compare_stage_vectors(py_row, qp_row, "propagation_support"),
            "parent_support": _compare_stage_vectors(py_row, qp_row, "parent_support"),
            "projected_hq": _compare_stage_vectors(py_row, qp_row, "projected_hq"),
            "orthogonalized_residual": _compare_stage_vectors(py_row, qp_row, "orthogonalized_residual"),
            "q_next": _compare_stage_vectors(py_row, qp_row, "q_next"),
            "lowest_ritz_energy": _compare_scalar(
                py_row.get("lowest_ritz_electronic_energy"),
                qp_row.get("lowest_ritz_electronic_energy"),
                tol=1.0e-9,
            ),
        }
        step_diffs.append(diff_row)
    diff_summary = {
        "steps": step_diffs,
        "first_mismatch": _localize_first_mismatch(step_diffs),
    }
    _write_json(outdir / "step_diff.json", diff_summary)
    _write_json(outdir / "step_diff_summary.json", diff_summary)

    restart_diff = None
    if qp_mode == "thick":
        restart_diff = _compare_restart_cycles(
            list(python_payload["restart"]["cycles"]),
            list(qp_restart_payload["cycles"]),
        )
        _write_json(outdir / "restart_diff.json", restart_diff)

    boundary_diag = {
        "mode": qp_mode,
        "steps": [
            {
                "step": int(py_row["step"]),
                "python": _boundary_diag_for_step(
                    py_row,
                    fixed_store_keep=int(fixed_store_keep),
                    fixed_propagate_keep=int(fixed_propagate_keep),
                ),
                "qp": _boundary_diag_for_step(
                    qp_row,
                    fixed_store_keep=int(fixed_store_keep),
                    fixed_propagate_keep=int(fixed_propagate_keep),
                ),
            }
            for py_row, qp_row in zip(python_steps, qp_rows)
        ],
        "restart": [] if qp_mode != "thick" else [
            {
                "cycle": int(py_cycle["cycle"]),
                "roots": [
                    {
                        "root": int(py_root["root"]),
                        "python_raw_store_boundary": _boundary_cutoff_diag_from_items(
                            list(py_root["raw_restart_vector"]["top"]),
                            keep=int(fixed_store_keep),
                        ),
                        "qp_raw_store_boundary": _boundary_cutoff_diag_from_items(
                            list(qp_root["raw_restart_vector"]["top"]),
                            keep=int(fixed_store_keep),
                        ),
                    }
                    for py_root, qp_root in zip(py_cycle["roots"], qp_cycle["roots"])
                ],
            }
            for py_cycle, qp_cycle in zip(python_payload["restart"]["cycles"], qp_restart_payload["cycles"])
        ],
    }
    _write_json(outdir / "boundary_diag.json", boundary_diag)

    payload = {
        "mode": "small_shared_parity",
        "qp_mode": qp_mode,
        "seed_root": str(SMALL_SHARED_SEED),
        "seed_ezfio": str(SMALL_SHARED_EZFIO),
        "seed_fcidump": str(SMALL_SHARED_FCIDUMP),
        "qp_root": str(qp_root_resolved),
        "omp_threads": 1,
        "fixed_store_keep": int(fixed_store_keep),
        "fixed_propagate_keep": int(fixed_propagate_keep),
        "steps_requested": int(steps),
        "m_cycle": int(m_cycle),
        "ncycles": int(ncycles),
        "nroot": int(nroot),
        "trace_topn": int(trace_topn),
        "shared_active_space": asdict(active),
        "shared_hamiltonian": {
            "metadata": hdata["metadata"],
            "exact_total_energy_eh": float(hdata["exact_total_energy_eh"]),
            "nuclear_repulsion_eh": float(nuclear_repulsion_eh),
            "electronic_shift_eh": float(electronic_shift_eh),
        },
        "artifacts": {
            "python_steps": str(outdir / "python_steps.json"),
            "qp_steps": str(outdir / "qp_steps.json"),
            "step_diff": str(outdir / "step_diff.json"),
            "step_diff_summary": str(outdir / "step_diff_summary.json"),
            "python_restart": None if qp_mode != "thick" else str(outdir / "python_restart.json"),
            "qp_restart": None if qp_mode != "thick" else str(outdir / "qp_restart.json"),
            "restart_diff": None if qp_mode != "thick" else str(outdir / "restart_diff.json"),
            "boundary_diag": str(outdir / "boundary_diag.json"),
        },
        "first_mismatch": diff_summary["first_mismatch"] if diff_summary["first_mismatch"]["step"] is not None else (
            None if restart_diff is None else restart_diff["first_mismatch"]
        ),
        "qp_run": qp_run,
    }
    _write_json(summary_path, payload)
    return payload


def classify_drift(results: dict[str, dict[str, object]]) -> dict[str, object]:
    donor = results.get("donor_seeded_qp")
    python_seed = results.get("python_seed_fcidump")
    fresh = results.get("fresh_publication_setup_qp")

    donor_match = bool(donor and donor["docs_comparison"]["matches_docs"])
    python_match = bool(python_seed and python_seed["docs_comparison"]["matches_docs"])
    fresh_match = bool(fresh and fresh["docs_comparison"]["matches_docs"])

    if donor_match and fresh is not None and not fresh_match:
        classification = "setup_drift_between_donor_seeded_and_fresh_publication_setup"
    elif donor is not None and python_seed is not None and (not donor_match) and (not python_match):
        classification = "seed_or_fcidump_semantics_drift_relative_to_retained_docs_row"
    elif donor is not None and python_seed is not None and python_match and (not donor_match):
        classification = "qp_fixed_budget_path_drift_on_same_seed_fcidump"
    elif donor is not None and fresh is not None and (not donor_match) and (not fresh_match):
        donor_delta = donor["docs_comparison"]["delta_vs_docs_mEh"]
        fresh_delta = fresh["docs_comparison"]["delta_vs_docs_mEh"]
        if donor_delta is not None and fresh_delta is not None and not math.isclose(float(donor_delta), float(fresh_delta), abs_tol=DELTA_TOL_MEH):
            classification = "setup_drift_first_donor_seeded_and_fresh_publication_setup_diverge"
        else:
            classification = "inconclusive_shared_drift_against_retained_docs_row"
    else:
        classification = "partial_or_inconclusive"

    return {
        "classification": classification,
        "donor_seeded_qp_matches_docs": donor_match,
        "python_seed_fcidump_matches_docs": python_match,
        "fresh_publication_setup_qp_matches_docs": fresh_match,
    }


def run_regression(
    *,
    case_name: str = DEFAULT_CASE,
    phase: str = DEFAULT_PHASE,
    budget_value: int = DEFAULT_BUDGET,
    method: str = DEFAULT_METHOD,
    outdir: Path = DEFAULT_OUTDIR,
    qp_root: Path | None = None,
    omp_threads: int = 8,
    selected_paths: tuple[str, ...] = PATH_CHOICES,
    bridge_policy: str = "current",
    qp_mode: str = "thick",
    force: bool = False,
) -> dict[str, object]:
    case = resolve_publication_case(case_name)
    golden_row = load_golden_docs_row(case_name=case_name, phase=phase, budget_value=budget_value, method=method)
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_qp_root = None if qp_root is None else Path(qp_root).resolve()
    if any(path in selected_paths for path in ("donor_seeded_qp", "fresh_publication_setup_qp")):
        resolved_qp_root = _resolve_qp_root_with_binary(qp_root)

    results: dict[str, dict[str, object]] = {}
    if "donor_seeded_qp" in selected_paths:
        assert resolved_qp_root is not None
        results["donor_seeded_qp"] = run_donor_seeded_qp_reproduction(
            case=case,
            golden_row=golden_row,
            qp_root=resolved_qp_root,
            outdir=outdir,
            omp_threads=int(omp_threads),
            bridge_policy=bridge_policy,
            qp_mode=qp_mode,
            force=force,
        )
    if "python_seed_fcidump" in selected_paths:
        results["python_seed_fcidump"] = run_python_seed_fcidump_reproduction(
            case=case,
            golden_row=golden_row,
            outdir=outdir,
            force=force,
        )
    if "fresh_publication_setup_qp" in selected_paths:
        assert resolved_qp_root is not None
        results["fresh_publication_setup_qp"] = run_fresh_publication_setup_qp_reproduction(
            case=case,
            golden_row=golden_row,
            qp_root=resolved_qp_root,
            outdir=outdir,
            omp_threads=int(omp_threads),
            bridge_policy=bridge_policy,
            qp_mode=qp_mode,
            force=force,
        )

    payload = {
        "case": case.name,
        "title": case.title,
        "qp_root": None if resolved_qp_root is None else str(resolved_qp_root),
        "comparison_mode": "bridge-only historical replay",
        "omp_threads": int(omp_threads),
        "phase": phase,
        "budget_value": int(budget_value),
        "method": method,
        "bridge_policy": bridge_policy,
        "qp_mode": qp_mode,
        "golden_row": golden_row,
        "runtime_and_plugin_fingerprints": (
            None if resolved_qp_root is None else _runtime_and_plugin_fingerprints(resolved_qp_root)
        ),
        "results": results,
        "classification": classify_drift(results),
    }
    _write_json(outdir / "summary.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the retained N2 CAS(12,12) fixed-budget docs row and classify drift")
    p.add_argument("--mode", default="regression", choices=RUN_MODES)
    p.add_argument("--case", default=DEFAULT_CASE, choices=[DEFAULT_CASE])
    p.add_argument("--phase", default=DEFAULT_PHASE, choices=("kprop", "kstore"))
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--method", default=DEFAULT_METHOD, choices=("ccik_thick_qp",))
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--qp-root", type=Path, default=None)
    p.add_argument("--omp-threads", default="auto")
    p.add_argument("--bridge-policy", default="current", choices=BRIDGE_POLICIES)
    p.add_argument("--qp-mode", default="thick", choices=QP_MODES)
    p.add_argument(
        "--paths",
        default=",".join(PATH_CHOICES),
        help="Comma-separated subset of paths to run: donor_seeded_qp, python_seed_fcidump, fresh_publication_setup_qp",
    )
    p.add_argument("--parity-kstore", type=int, default=256)
    p.add_argument("--parity-kprop", type=int, default=64)
    p.add_argument("--parity-steps", type=int, default=8)
    p.add_argument("--parity-m-cycle", type=int, default=SMALL_PARITY_THICK_MCYCLE)
    p.add_argument("--parity-ncycles", type=int, default=SMALL_PARITY_THICK_NCYCLES)
    p.add_argument("--parity-nroot", type=int, default=SMALL_PARITY_THICK_NROOT)
    p.add_argument("--parity-trace-topn", type=int, default=100000)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def _parse_paths(text: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    unknown = sorted(set(parts) - set(PATH_CHOICES))
    if unknown:
        raise ValueError(f"Unknown regression paths requested: {unknown}")
    if not parts:
        raise ValueError("At least one regression path must be requested")
    return parts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "small_shared_parity":
        outdir = args.outdir if args.outdir != DEFAULT_OUTDIR else DEFAULT_PARITY_OUTDIR
        payload = run_small_shared_seed_step_parity(
            outdir=outdir,
            qp_root=args.qp_root,
            fixed_store_keep=int(args.parity_kstore),
            fixed_propagate_keep=int(args.parity_kprop),
            steps=int(args.parity_steps),
            qp_mode=str(args.qp_mode),
            m_cycle=int(args.parity_m_cycle),
            ncycles=int(args.parity_ncycles),
            nroot=int(args.parity_nroot),
            trace_topn=int(args.parity_trace_topn),
            force=args.force,
        )
        print(json.dumps(payload["first_mismatch"], indent=2, sort_keys=True))
        print(outdir / "summary.json")
        return 0

    selected_paths = _parse_paths(args.paths)
    omp_threads = pub.resolve_omp_threads(args.omp_threads)
    payload = run_regression(
        case_name=args.case,
        phase=args.phase,
        budget_value=int(args.budget),
        method=args.method,
        outdir=args.outdir,
        qp_root=args.qp_root,
        omp_threads=int(omp_threads),
        selected_paths=selected_paths,
        bridge_policy=args.bridge_policy,
        qp_mode=args.qp_mode,
        force=args.force,
    )
    print(json.dumps(payload["classification"], indent=2, sort_keys=True))
    print(args.outdir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
