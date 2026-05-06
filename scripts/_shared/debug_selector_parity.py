"""Deep parity/debug driver for the matvec-first selector scheme.

This script can run a single donor-backed parity case or a small N2 donor sweep.
The Python side always uses the same Hamiltonian exported from the prepared QP
EZFIO via FCIDUMP in order to compare the Python oracle and the QP plugin on
the same active-space problem.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLD_TEST_ROOT = _ROOT / "experiments" / "threshold_test"
_LOCAL_SEED_ROOT = _ROOT / "docs" / "assets" / "local_qp_seeds"
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import make_mol_pyscf  # noqa: E402
from ccik.params import CCIKParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from ccik.reference_debug import trace_ccik_run_with_energy  # noqa: E402
from n2_utils import make_n2_atom  # noqa: E402
from qp_ccik_common import (  # noqa: E402
    _with_qp_build_env,
    PreparedActiveSpace,
    QPPaths,
    bash,
    discover_qp_paths_from_root,
    detect_physical_cpu_count,
    ensure_ezfio_defaults,
    load_prepared_active_space,
    qp_parallel_env,
    rebuild_ccik_plugin,
    resolve_rebuildable_qp_root,
    resolve_omp_threads,
)


@dataclass(frozen=True)
class DonorCaseSpec:
    name: str
    donor_ezfio: Path
    cached_qp_fcidump: Path | None
    basis: str
    geometry_label: str
    r_ang: float
    ncore_hint: int
    ncas_hint: int
    mo_num_hint: int
    recommended_steps: int
    exact_hq_diagnostics: bool
    charge: int = 0
    spin: int = 0


@dataclass(frozen=True)
class N2ReferenceCase:
    h1: object
    eri8: object
    ecore: float
    e_nuc: float
    ncas: int
    nelec: tuple[int, int]
    mo_coeff: list[list[float]]
    mo_occ: list[float]
    ao_md5: str
    mo_label: str
    ncore: int
    mo_num: int


DONOR_N2_CASES: dict[str, DonorCaseSpec] = {
    "n2_cas10_10_R1p10": DonorCaseSpec(
        name="n2_cas10_10_R1p10",
        donor_ezfio=_THRESHOLD_TEST_ROOT / "n2_qp_ccik_simplified_smoke" / "n2_1p10.ezfio",
        cached_qp_fcidump=_THRESHOLD_TEST_ROOT / "threshold_mode_validation_20260325" / "zero_cas10" / "qp_active.FCIDUMP",
        basis="6-31g",
        geometry_label="R1p10",
        r_ang=1.10,
        ncore_hint=2,
        ncas_hint=10,
        mo_num_hint=18,
        recommended_steps=4,
        exact_hq_diagnostics=True,
    ),
    "n2_cas12_12_R2p50": DonorCaseSpec(
        name="n2_cas12_12_R2p50",
        donor_ezfio=(
            _LOCAL_SEED_ROOT / "n2_cas12_12_R2p50" / "n2.ezfio"
            if (_LOCAL_SEED_ROOT / "n2_cas12_12_R2p50" / "n2.ezfio").exists()
            else _THRESHOLD_TEST_ROOT / "n2_qp_ccik_cas12_12_R2p50_b" / "n2_2p50.ezfio"
        ),
        cached_qp_fcidump=(
            _LOCAL_SEED_ROOT / "n2_cas12_12_R2p50" / "qp_active.FCIDUMP"
            if (_LOCAL_SEED_ROOT / "n2_cas12_12_R2p50" / "qp_active.FCIDUMP").exists()
            else _THRESHOLD_TEST_ROOT / "threshold_mode_validation_20260325" / "zero_cas12" / "qp_active.FCIDUMP"
        ),
        basis="6-31g",
        geometry_label="R2p50",
        r_ang=2.50,
        ncore_hint=1,
        ncas_hint=12,
        mo_num_hint=18,
        recommended_steps=4,
        exact_hq_diagnostics=True,
    ),
    "n2_cas14_14_R2p50": DonorCaseSpec(
        name="n2_cas14_14_R2p50",
        donor_ezfio=(
            _LOCAL_SEED_ROOT / "n2_cas14_14_R2p50" / "n2.ezfio"
            if (_LOCAL_SEED_ROOT / "n2_cas14_14_R2p50" / "n2.ezfio").exists()
            else _THRESHOLD_TEST_ROOT / "n2_qp_ccik_cas14_14_hard_budget1e7" / "n2_2p50.ezfio"
        ),
        cached_qp_fcidump=(
            _LOCAL_SEED_ROOT / "n2_cas14_14_R2p50" / "qp_active.FCIDUMP"
            if (_LOCAL_SEED_ROOT / "n2_cas14_14_R2p50" / "qp_active.FCIDUMP").exists()
            else _THRESHOLD_TEST_ROOT / "donor_n2_parity_sweep_20260320_rerun" / "n2_cas14_14_R2p50" / "connected_sd_fullq" / "qp_active.FCIDUMP"
        ),
        basis="6-31g",
        geometry_label="R2p50",
        r_ang=2.50,
        ncore_hint=0,
        ncas_hint=14,
        mo_num_hint=18,
        recommended_steps=2,
        exact_hq_diagnostics=False,
    ),
}

DONOR_N2_SWEEP: tuple[str, ...] = (
    "n2_cas10_10_R1p10",
    "n2_cas12_12_R2p50",
    "n2_cas14_14_R2p50",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep selector parity/debug driver")
    p.add_argument("--outdir", type=Path, default=_THRESHOLD_TEST_ROOT / "selector_parity_debug")
    p.add_argument("--case", choices=sorted(DONOR_N2_CASES), default=None, help="Named donor-backed parity case")
    p.add_argument("--sweep", choices=["donor_n2"], default=None, help="Run the donor-backed N2 benchmark sweep")
    p.add_argument("--R", type=float, default=None, help="Backward-compatible alias for the 1.10 A CAS(10,10) donor case")
    p.add_argument("--steps", type=int, default=None, help="Override the per-case recommended step budget")
    p.add_argument("--selector-mode", default="abs_residual", choices=["pt2_residual", "pt2_legacy", "abs_hq", "abs_residual"])
    p.add_argument(
        "--matvec-backend",
        default="connected_sd_fullq",
        choices=["legacy_pt2", "connected_sd_fullq"],
        help="QP projected-matvec backend to exercise",
    )
    p.add_argument(
        "--compression-policy",
        default="fixed_budget",
        choices=["topk", "threshold", "fixed_budget"],
        help="Compression policy to use in the Python and QP CCIK runs",
    )
    p.add_argument(
        "--fixed-store-keep",
        type=int,
        default=1_000_000,
        help="Stored-vector top-K cap used when --compression-policy=fixed_budget",
    )
    p.add_argument(
        "--fixed-propagate-keep",
        type=int,
        default=10_000,
        help="Propagation-support top-K cap used when --compression-policy=fixed_budget",
    )
    p.add_argument(
        "--store-abs-tol",
        type=float,
        default=0.0,
        help="Absolute threshold for storing normalized Krylov-vector coefficients in threshold mode",
    )
    p.add_argument(
        "--propagate-abs-tol",
        type=float,
        default=0.0,
        help="Absolute threshold for using normalized q_k coefficients in Hq propagation in threshold mode",
    )
    p.add_argument("--trace-topn", type=int, default=32, help="Top-N coefficients to dump in Python/QP traces")
    p.add_argument(
        "--parity-basis-mode",
        default="qp_fcidump",
        choices=["qp_fcidump", "inject_pyscf"],
        help="Hamiltonian/basis source for Python-vs-QP parity",
    )
    p.add_argument("--with-qp", action="store_true", help="Also run the QP plugin if a QP root is available")
    p.add_argument("--qp-root", type=Path, default=None, help="Explicit Quantum Package root")
    p.add_argument(
        "--omp-threads",
        default="auto",
        help=f'OpenMP thread count for QP runs ("auto" uses physical cores, currently {detect_physical_cpu_count()})',
    )
    p.add_argument("--keep-ezfio", action="store_true")
    return p.parse_args(argv)


def _resolve_requested_cases(args: argparse.Namespace) -> list[DonorCaseSpec]:
    if args.sweep == "donor_n2":
        return [DONOR_N2_CASES[name] for name in DONOR_N2_SWEEP]
    if args.case is not None:
        return [DONOR_N2_CASES[args.case]]
    if args.R is not None:
        if abs(float(args.R) - 1.10) > 1e-10:
            raise ValueError("--R is only kept as a backward-compatible alias for the 1.10 A CAS(10,10) donor case")
        return [DONOR_N2_CASES["n2_cas10_10_R1p10"]]
    return [DONOR_N2_CASES["n2_cas10_10_R1p10"]]


def _effective_steps(args: argparse.Namespace, case: DonorCaseSpec) -> int:
    return int(case.recommended_steps if args.steps is None else args.steps)


def build_n2_case(case: DonorCaseSpec, active: PreparedActiveSpace | None = None) -> N2ReferenceCase:
    from pyscf import ao2mo, mcscf, scf

    ncore = case.ncore_hint if active is None else int(active.ncore)
    ncas = case.ncas_hint if active is None else int(active.ncas)
    if active is None:
        nelecas = int(2 * (7 - ncore))
        nelec = (nelecas // 2, nelecas - nelecas // 2)
    else:
        nelec = (int(active.nelecas // 2), int(active.nelecas - active.nelecas // 2))
        nelecas = int(active.nelecas)

    mol = make_mol_pyscf(
        atom=make_n2_atom(float(case.r_ang)),
        basis=case.basis,
        unit="Angstrom",
        charge=case.charge,
        spin=case.spin,
        verbose=0,
    )
    cas = CASSpec(ncas=ncas, nelecas=nelecas, ncore=ncore)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, cas.ncas, cas.nelecas)
    mc.ncore = cas.ncore
    mc.mo_coeff = mf.mo_coeff
    h1eff, ecore = mc.get_h1eff()
    eri8 = ao2mo.restore(8, mc.get_h2eff(), cas.ncas)
    ao_md5 = (case.donor_ezfio / "ao_basis" / "ao_md5").read_text(encoding="ascii").strip()

    return N2ReferenceCase(
        h1=h1eff,
        eri8=eri8,
        ecore=float(ecore),
        e_nuc=float(mol.energy_nuc()),
        ncas=cas.ncas,
        nelec=nelec,
        mo_coeff=mf.mo_coeff.tolist(),
        mo_occ=mf.mo_occ.tolist(),
        ao_md5=ao_md5,
        mo_label="PySCF_RHF_631G",
        ncore=cas.ncore,
        mo_num=int(mf.mo_coeff.shape[1]),
    )


def _ezfio_scalar_text(value: object) -> str:
    if isinstance(value, bool):
        return ("T" if value else "F") + "\n\n"
    if isinstance(value, int):
        return f"{value:20d}\n"
    if isinstance(value, float):
        return f"{value:24.15E}\n"
    return f"{value}\n"


def _write_ezfio_value(ezfio: Path, relpath: str, value: object) -> None:
    target = ezfio / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_ezfio_scalar_text(value), encoding="ascii")


def _set_ccik_params_direct(ezfio: Path, args: argparse.Namespace) -> None:
    basis_size = max(2, int(args.steps) + 1)
    values: dict[str, object] = {
        "ccik/ccik_m": basis_size,
        "ccik/ccik_basis_size": basis_size,
        "ccik/ccik_generator_keep": 256,
        "ccik/ccik_generator_fraction": 0.0,
        "ccik/ccik_generator_abs_coeff_cutoff": 0.0,
        "ccik/ccik_max_parents_for_expansion": 256,
        "ccik/ccik_include_current_support": True,
        "ccik/ccik_selector_keep": 2000,
        "ccik/ccik_external_keep": 4000,
        "ccik/ccik_external_candidate_factor": 4,
        "ccik/ccik_pt2_rank_weight": 1.0,
        "ccik/ccik_selector_mode": args.selector_mode,
        "ccik/ccik_matvec_backend": args.matvec_backend,
        "ccik/ccik_compression_policy": getattr(args, "compression_policy", "topk"),
        "ccik/ccik_fixed_store_keep": int(getattr(args, "fixed_store_keep", 1_000_000)),
        "ccik/ccik_fixed_propagate_keep": int(getattr(args, "fixed_propagate_keep", 10_000)),
        "ccik/ccik_store_abs_tol": float(getattr(args, "store_abs_tol", 0.0)),
        "ccik/ccik_propagate_abs_tol": float(getattr(args, "propagate_abs_tol", 0.0)),
        "ccik/ccik_nkeep": 12000,
        "ccik/ccik_tmp_keep_factor": 4,
        "ccik/ccik_preserve_prev_support": True,
        "ccik/ccik_preserve_prev_support_cap": 2048,
        "ccik/ccik_total_det_budget": 0,
        "ccik/ccik_conv_tol": 1e-12,
        "ccik/ccik_ritz_max_drop": 0.0,
        "ccik/ccik_save_final_wf": False,
        "ccik/ccik_use_existing_wf": False,
        "ccik/ccik_do_post_pt2": False,
        "ccik/ccik_debug_trace": True,
        "ccik/ccik_debug_trace_topn": int(args.trace_topn),
        "ccik/ccik_max_cycle": 1,
        "ccik/ccik_restart_m_cycle": 0,
        "ccik/ccik_restart_nroot": 2,
        "ccik/ccik_verbose": False,
    }
    for relpath, value in values.items():
        _write_ezfio_value(ezfio, relpath, value)


def _set_mo_class_direct(qp: QPPaths, ezfio: Path, mo_num: int, ncore: int, ncas: int) -> None:
    _write_ezfio_value(ezfio, "mo_basis/mo_num", mo_num)
    script = f"""
from ezfio import ezfio
ezfio.set_file(r"{ezfio}")
mo_num = int(ezfio.get_mo_basis_mo_num())
mo_class = []
for i in range(1, mo_num + 1):
    if i <= {int(ncore)}:
        mo_class.append("Core")
    elif i <= {int(ncore + ncas)}:
        mo_class.append("Active")
    else:
        mo_class.append("Deleted")
ezfio.set_mo_basis_mo_class(mo_class)
ezfio.set_mo_basis_mo_symmetry([1] * mo_num)
"""
    bash(
        f"source {qp.quantum_package_rc} && python3 - <<'PY'\n{script}\nPY",
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
    )


def _inject_reference_basis(qp: QPPaths, ezfio: Path, case: N2ReferenceCase, payload_path: Path) -> None:
    payload = {
        "ezfio": str(ezfio),
        "mo_num": int(case.mo_num),
        "mo_label": case.mo_label,
        "ao_md5": case.ao_md5,
        "mo_occ": [float(x) for x in case.mo_occ],
        "mo_coef": [[float(x) for x in row] for row in case.mo_coeff],
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    script = f"""
import json
from pathlib import Path
from ezfio import ezfio

payload = json.loads(Path(r"{payload_path}").read_text())
ezfio.set_file(payload["ezfio"])
mo_coef = [list(col) for col in zip(*payload["mo_coef"])]
ezfio.set_mo_basis_mo_num(int(payload["mo_num"]))
ezfio.set_mo_basis_mo_label(str(payload["mo_label"]))
ezfio.set_mo_basis_ao_md5(str(payload["ao_md5"]))
ezfio.set_mo_basis_mo_occ([float(x) for x in payload["mo_occ"]])
ezfio.set_mo_basis_mo_coef(mo_coef)
"""
    bash(
        f"source {qp.quantum_package_rc} && python3 - <<'PY'\n{script}\nPY",
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
    )


def _force_integral_rebuild(qp: QPPaths, ezfio: Path) -> None:
    _write_ezfio_value(ezfio, "ao_two_e_ints/io_ao_two_e_integrals", "Write")
    _write_ezfio_value(ezfio, "mo_two_e_ints/io_mo_two_e_integrals", "Write")
    _write_ezfio_value(ezfio, "mo_one_e_ints/io_mo_one_e_integrals", "None")


def _write_prepared_metadata(path: Path, case: DonorCaseSpec) -> None:
    payload = {
        "case": case.name,
        "mo_num": int(case.mo_num_hint),
        "ncore": int(case.ncore_hint),
        "ncas": int(case.ncas_hint),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_qp_debug_trace(trace_path: Path) -> dict[int, dict[str, object]]:
    parsed: dict[int, dict[str, object]] = {}
    current_step = 0
    current_vec: str | None = None
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "CCIK_TRACE_HQ":
            current_step = int(parts[1])
            step = parsed.setdefault(current_step, {"vectors": {}})
            current_vec = None
            if len(parts) >= 14:
                meta = {
                    "selector_mode": parts[2],
                    "matvec_backend": parts[3],
                    "compression_policy": parts[4],
                    "fullq_support_size": int(parts[5]),
                    "parent_count": int(parts[6]),
                    "generated_before_dedup": int(parts[7]),
                    "generated_unique": int(parts[8]),
                    "rayleigh": float(parts[9]),
                    "source_used_fallback": parts[10] == "T",
                    "source_dropped_norm": float(parts[11]),
                    "projected_full_norm": float(parts[12]),
                    "projected_final_norm": float(parts[13]),
                }
            else:
                meta = {
                    "selector_mode": parts[2],
                    "matvec_backend": parts[3],
                    "compression_policy": "topk",
                    "fullq_support_size": int(parts[4]),
                    "parent_count": int(parts[5]),
                    "generated_before_dedup": int(parts[6]),
                    "generated_unique": int(parts[7]),
                    "rayleigh": float(parts[8]),
                    "source_used_fallback": False,
                    "source_dropped_norm": 0.0,
                    "projected_full_norm": float(parts[9]),
                    "projected_final_norm": float(parts[10]),
                }
            step["hq_meta"] = meta
        elif tag == "CCIK_TRACE_NEXT":
            current_step = int(parts[1])
            step = parsed.setdefault(current_step, {"vectors": {}})
            current_vec = None
            if len(parts) >= 15:
                step["next_meta"] = {
                    "selector_mode": parts[2],
                    "compression_policy": parts[3],
                    "tmp_keep": int(parts[4]),
                    "final_keep": int(parts[5]),
                    "preserve_cap": int(parts[6]),
                    "store_used_fallback": parts[7] == "T",
                    "q_norm": float(parts[8]),
                    "projected_norm": float(parts[9]),
                    "v_temp_norm": float(parts[10]),
                    "discarded_weight": float(parts[11]),
                    "orthogonalized_residual_norm": float(parts[12]),
                    "post_compression_residual_norm": float(parts[13]),
                    "store_dropped_norm": float(parts[14]),
                }
            else:
                step["next_meta"] = {
                    "selector_mode": parts[2],
                    "compression_policy": "topk",
                    "tmp_keep": int(parts[3]),
                    "final_keep": int(parts[4]),
                    "preserve_cap": int(parts[5]),
                    "store_used_fallback": False,
                    "q_norm": float(parts[6]),
                    "projected_norm": float(parts[7]),
                    "v_temp_norm": float(parts[8]),
                    "discarded_weight": float(parts[9]),
                    "orthogonalized_residual_norm": float(parts[10]),
                    "post_compression_residual_norm": float(parts[11]),
                    "store_dropped_norm": 0.0,
                }
        elif tag == "CCIK_TRACE_VEC":
            current_vec = parts[1]
            parsed.setdefault(current_step, {"vectors": {}}).setdefault("vectors", {}).setdefault(current_vec, [])
        elif tag == "CCIK_TRACE_COEF" and current_vec is not None:
            vec_name = parts[1]
            vectors = parsed.setdefault(current_step, {"vectors": {}}).setdefault("vectors", {})
            if vec_name != current_vec:
                current_vec = vec_name
                vectors.setdefault(current_vec, [])
            vectors[current_vec].append(
                {
                    "rank": int(parts[2]),
                    "stra": int(parts[3]),
                    "strb": int(parts[4]),
                    "coef": float(parts[5]),
                    "abs_coef": float(parts[6]),
                }
            )
    return parsed


def _trace_exposes_compression_policy(trace_path: Path) -> dict[str, bool]:
    found = {"hq": False, "next": False}
    known_policies = {"topk", "threshold", "fixed_budget"}
    if not trace_path.exists():
        return found
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "CCIK_TRACE_HQ" and len(parts) >= 5 and parts[4] in known_policies:
            found["hq"] = True
        if parts[0] == "CCIK_TRACE_NEXT" and len(parts) >= 4 and parts[3] in known_policies:
            found["next"] = True
    return found


def _vector_l2_diff(py_items: list[dict[str, object]], qp_items: list[dict[str, object]]) -> dict[str, object]:
    py_map = {(int(it["stra"]), int(it["strb"])): float(it["coef"]) for it in py_items}
    qp_map = {(int(it["stra"]), int(it["strb"])): float(it["coef"]) for it in qp_items}
    keys = set(py_map) | set(qp_map)
    l2 = sum((py_map.get(k, 0.0) - qp_map.get(k, 0.0)) ** 2 for k in keys) ** 0.5
    overlap = len(set(py_map) & set(qp_map))
    return {
        "l2_diff_on_traced_union": float(l2),
        "support_overlap": int(overlap),
        "support_union": int(len(keys)),
        "python_size": int(len(py_map)),
        "qp_size": int(len(qp_map)),
    }


def _map_qp_det_to_active(det: int, *, ncore: int, ncas: int) -> int:
    out = 0
    for iact in range(int(ncas)):
        full_bit = int(ncore) + iact
        if int(det) & (1 << full_bit):
            out |= 1 << iact
    return out


def _compare_python_qp_vectors(py_trace_path: Path, qp_trace_path: Path, *, ncore: int, ncas: int) -> dict[str, object]:
    py_rows = [json.loads(line) for line in py_trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qp = _parse_qp_debug_trace(qp_trace_path)
    vec_map = {
        "projected_hq": "projected_top",
        "score_vec": "score_vector_top",
        "q_next": "q_next_top",
    }
    out: dict[str, object] = {"steps": {}}
    for idx, row in enumerate(py_rows, start=1):
        if idx not in qp:
            continue
        step_out: dict[str, object] = {}
        for qp_name, py_name in vec_map.items():
            py_items = row["truncated"][py_name]
            qp_items = [
                {
                    **it,
                    "stra": _map_qp_det_to_active(int(it["stra"]), ncore=ncore, ncas=ncas),
                    "strb": _map_qp_det_to_active(int(it["strb"]), ncore=ncore, ncas=ncas),
                }
                for it in qp[idx].get("vectors", {}).get(qp_name, [])
            ]
            step_out[qp_name] = _vector_l2_diff(py_items, qp_items)
        if "hq_meta" in qp[idx]:
            step_out["hq_meta"] = qp[idx]["hq_meta"]
        if "next_meta" in qp[idx]:
            step_out["next_meta"] = qp[idx]["next_meta"]
        out["steps"][str(idx)] = step_out
    return out


def python_trace_from_data(
    args: argparse.Namespace,
    outdir: Path,
    *,
    h1,
    eri8,
    ncas: int,
    nelec: tuple[int, int],
    mo_label: str,
    hamiltonian_source: str,
    exact_hq_diagnostics: bool,
) -> dict[str, object]:
    params = CCIKParams(
        m=max(2, int(args.steps) + 1),
        nadd=2000,
        nkeep=12000,
        Kv=4000,
        compression_policy=args.compression_policy,
        fixed_store_keep=int(args.fixed_store_keep),
        fixed_propagate_keep=int(args.fixed_propagate_keep),
        store_abs_tol=float(args.store_abs_tol),
        propagate_abs_tol=float(args.propagate_abs_tol),
        candidate_mode="connected_sd",
        selector_mode=args.selector_mode,
        selector_depth=1,
        generator_keep=256,
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=256,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=2048,
        exact_hq_diagnostics=exact_hq_diagnostics,
        selector_ci_coeff_cutoff=5e-4,
        selector_cutoff=5e-4,
        orth_tol=1e-12,
        verbose=False,
    )
    trace_path = outdir / "python_trace.jsonl"
    t0 = time.perf_counter()
    trace_payload = trace_ccik_run_with_energy(
        h1,
        eri8,
        ncas,
        nelec,
        params=params,
        max_steps=int(args.steps),
        topn=int(args.trace_topn),
        output_path=trace_path,
    )
    wall_time_s = time.perf_counter() - t0
    traces = trace_payload["traces"]
    summary = trace_payload["summary"]
    payload = {
        "selector_mode": args.selector_mode,
        "steps": len(traces),
        "mo_label": mo_label,
        "hamiltonian_source": hamiltonian_source,
        "exact_hq_diagnostics": bool(exact_hq_diagnostics),
        "summary": summary,
        "trace_path": str(trace_path),
        "wall_time_s": float(wall_time_s),
    }
    (outdir / "python_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _extract_final_scalar(stdout: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)} =\s*([-+0-9.eE]+)", stdout)
    if m is None:
        return None
    return float(m.group(1))


def _apply_constant_shift_to_fixed_nelec_h1(h1, *, nelec: tuple[int, int], shift: float):
    import numpy as np

    out = np.array(h1, copy=True)
    nelec_total = int(nelec[0] + nelec[1])
    if nelec_total <= 0:
        return out
    delta = float(shift) / float(nelec_total)
    idx = np.diag_indices_from(out)
    out[idx] += delta
    return out


def _case_nuclear_repulsion(case: DonorCaseSpec) -> float:
    mol = make_mol_pyscf(
        atom=make_n2_atom(float(case.r_ang)),
        basis=case.basis,
        unit="Angstrom",
        charge=case.charge,
        spin=case.spin,
        verbose=0,
    )
    return float(mol.energy_nuc())


def discover_qp_paths(qp_root: Path) -> QPPaths:
    return discover_qp_paths_from_root(qp_root)


def resolve_qp_root(explicit: Path | None) -> Path | None:
    return resolve_rebuildable_qp_root(explicit)


def _prepare_qp_workdir(
    args: argparse.Namespace,
    qp: QPPaths,
    *,
    case_spec: DonorCaseSpec,
    inject_pyscf_basis: bool,
    reference_case: N2ReferenceCase | None = None,
) -> tuple[Path, Path, Path]:
    if not case_spec.donor_ezfio.exists():
        raise RuntimeError(f"Missing donor EZFIO for {case_spec.name}: {case_spec.donor_ezfio}")

    workdir = Path(tempfile.mkdtemp(prefix=f"{case_spec.name}_", dir="/tmp"))
    ezfio = workdir / f"{case_spec.name}.ezfio"
    shutil.copytree(case_spec.donor_ezfio, ezfio)
    ensure_ezfio_defaults(ezfio)

    metadata_path = workdir / "prepared_metadata.json"
    _write_prepared_metadata(metadata_path, case_spec)

    if inject_pyscf_basis:
        if reference_case is None:
            raise RuntimeError("inject_pyscf mode requires a prepared Python reference case")
        _inject_reference_basis(qp, ezfio, reference_case, workdir / "pyscf_mos.json")

    _set_mo_class_direct(qp, ezfio, case_spec.mo_num_hint, case_spec.ncore_hint, case_spec.ncas_hint)
    _force_integral_rebuild(qp, ezfio)
    _set_ccik_params_direct(ezfio, args)
    return workdir, ezfio, metadata_path


def _generate_qp_fcidump(qp: QPPaths, ezfio: Path, outdir: Path, *, case_spec: DonorCaseSpec | None = None) -> Path:
    from pyscf.tools import fcidump as pyscf_fcidump

    copied = outdir / "qp_active.FCIDUMP"
    cached = None if case_spec is None else case_spec.cached_qp_fcidump
    if cached is not None and cached.exists():
        shutil.copy2(cached, copied)
        source_kind = "cached_case_fcidump"
        raw = cached
    else:
        fcidump_binary = qp.qp_root / "src" / "tools" / "fcidump"
        quoted_rc = shlex.quote(str(qp.quantum_package_rc))
        quoted_binary = shlex.quote(str(fcidump_binary))
        short_ezfio = shlex.quote(ezfio.name)
        tmpdir = Path(tempfile.mkdtemp(prefix="qp_fcidump_", dir="/tmp"))
        try:
            run = subprocess.run(
                ["/bin/bash", "-lc", f"source {quoted_rc} && {quoted_binary} {short_ezfio}"],
                check=False,
                text=True,
                cwd=ezfio.parent,
                capture_output=True,
                env=_with_qp_build_env({**os.environ, "QP_ROOT": str(qp.qp_root), "TMPDIR": str(tmpdir)}),
            )
            (outdir / "fcidump_stdout.log").write_text(run.stdout, encoding="utf-8")
            (outdir / "fcidump_stderr.log").write_text(run.stderr, encoding="utf-8")
            if run.returncode != 0:
                raise RuntimeError(
                    "QP FCIDUMP export failed "
                    f"(returncode={run.returncode}); see "
                    f"{outdir / 'fcidump_stdout.log'} and {outdir / 'fcidump_stderr.log'}"
                )
            raw = ezfio.parent / f"{ezfio.name}.FCIDUMP"
            shutil.copy2(raw, copied)
            source_kind = "generated_from_qp"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    data = pyscf_fcidump.read(str(copied), verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec + ms2) // 2
    nelecb = nelec - neleca
    payload = {
        "source_kind": source_kind,
        "source_path": str(raw),
        "fcidump_path": str(copied),
        "norb": norb,
        "nelec": [int(neleca), int(nelecb)],
        "ecore": float(data.get("ECORE", 0.0)),
    }
    (outdir / "qp_fcidump_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return copied


def _load_qp_fcidump_hamiltonian(fcidump_path: Path) -> tuple[object, object, int, tuple[int, int], float]:
    from pyscf import ao2mo
    from pyscf.tools import fcidump as pyscf_fcidump

    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec + ms2) // 2
    nelecb = nelec - neleca
    return data["H1"], ao2mo.restore(8, data["H2"], norb), norb, (int(neleca), int(nelecb)), float(data.get("ECORE", 0.0))


def run_qp_trace(
    args: argparse.Namespace,
    outdir: Path,
    *,
    prepared: tuple[Path, Path],
    timeout_s: int | None = None,
) -> dict[str, object]:
    qp_root = resolve_qp_root(args.qp_root)
    if qp_root is None:
        raise RuntimeError("No Quantum Package root found. Use --qp-root or source quantum_package.rc.")

    qp = discover_qp_paths(qp_root)
    workdir, ezfio = prepared
    workdir = Path(workdir).resolve()
    ezfio = Path(ezfio).resolve()
    omp_threads = resolve_omp_threads(getattr(args, "omp_threads", "auto"))
    quoted_rc = shlex.quote(str(qp.quantum_package_rc))
    quoted_ccik = shlex.quote(str(qp.ccik_binary))
    quoted_ezfio = shlex.quote(str(ezfio))
    t0 = time.perf_counter()
    stdout_path = outdir / "ccik_stdout.log"
    stderr_path = outdir / "ccik_stderr.log"
    raw_trace_path = workdir / "ccik_debug_trace.log"
    trace_path = outdir / "ccik_debug_trace.log"
    tmpdir = workdir / ".tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    def _as_text(payload: object) -> str:
        if payload is None:
            return ""
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return str(payload)

    try:
        run = subprocess.run(
            ["/bin/bash", "-lc", f"source {quoted_rc} && {quoted_ccik} {quoted_ezfio}"],
            check=False,
            text=True,
            cwd=workdir,
            capture_output=True,
            timeout=timeout_s,
            env=_with_qp_build_env({
                **os.environ,
                "QP_ROOT": str(qp_root),
                "TMPDIR": str(tmpdir),
                "TMP": str(tmpdir),
                "TEMP": str(tmpdir),
                **qp_parallel_env(omp_threads),
            }),
        )
        wall_time_s = time.perf_counter() - t0
        stdout_path.write_text(run.stdout, encoding="utf-8")
        stderr_path.write_text(run.stderr, encoding="utf-8")
        if raw_trace_path.exists():
            shutil.copy2(raw_trace_path, trace_path)
    except subprocess.TimeoutExpired as exc:
        wall_time_s = time.perf_counter() - t0
        stdout_path.write_text(_as_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_as_text(exc.stderr), encoding="utf-8")
        if raw_trace_path.exists():
            shutil.copy2(raw_trace_path, trace_path)
        raise RuntimeError(
            "QP CCIK run timed out.\n"
            f"timeout_s={timeout_s}\n"
            f"wall_time_s={wall_time_s}\n"
            f"stdout_path={stdout_path}\n"
            f"stderr_path={stderr_path}\n"
            f"trace_path={trace_path}"
        ) from exc

    if run.returncode != 0:
        raise RuntimeError(
            "QP CCIK run failed.\n"
            f"returncode={run.returncode}\n"
            f"stdout_path={stdout_path}\n"
            f"stderr_path={stderr_path}\n"
            f"trace_path={trace_path}\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}"
        )

    expected_markers = [
        "ccik_generator_abs_coeff_cutoff",
        "ccik_selector_mode",
        "ccik_matvec_backend",
        "ccik_compression_policy",
        "ccik_store_abs_tol",
        "ccik_propagate_abs_tol",
        "ccik_tmp_keep_factor",
        "ccik_debug_trace",
    ]
    observed_markers = {marker: (marker in run.stdout) for marker in expected_markers}
    trace_layout = _trace_exposes_compression_policy(trace_path)
    mo_label_match = re.search(r"\*\s+mo_label\s+([A-Za-z0-9_\\-]+)", run.stdout)
    runtime_mo_label = mo_label_match.group(1) if mo_label_match else None

    hq_steps = _parse_qp_debug_trace(trace_path) if trace_path.exists() else {}
    payload = {
        "qp_root": str(qp_root),
        "workdir": str(workdir),
        "ezfio": str(ezfio),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "trace_path": str(trace_path),
        "trace_exists": trace_path.exists(),
        "requested_basis_mode": args.parity_basis_mode,
        "requested_matvec_backend": args.matvec_backend,
        "runtime_mo_label": runtime_mo_label,
        "final_electronic_energy": _extract_final_scalar(run.stdout, "CCIK_FINAL_ELECTRONIC_ENERGY"),
        "final_total_energy": _extract_final_scalar(run.stdout, "CCIK_FINAL_TOTAL_ENERGY"),
        "final_basis": _extract_final_scalar(run.stdout, "CCIK_FINAL_BASIS"),
        "observed_markers": observed_markers,
        "trace_layout": trace_layout,
        "likely_stale_generated_binary": (not trace_path.exists()) or (not all(observed_markers.values())) or (not all(trace_layout.values())),
        "wall_time_s": float(wall_time_s),
        "omp_threads": int(omp_threads),
        "hq_steps": hq_steps,
    }
    (outdir / "qp_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["likely_stale_generated_binary"]:
        raise RuntimeError(
            "QP parity run detected a stale or incomplete binary. "
            f"Observed markers: {observed_markers}. Trace policy markers: {trace_layout}. "
            f"See {outdir / 'qp_summary.json'} and {stdout_path}."
        )
    return payload


def _case_args(args: argparse.Namespace, *, case: DonorCaseSpec, backend: str | None = None) -> argparse.Namespace:
    data = vars(args).copy()
    data["case"] = case.name
    data["R"] = case.r_ang
    data["steps"] = _effective_steps(args, case)
    if backend is not None:
        data["matvec_backend"] = backend
    return argparse.Namespace(**data)


def _run_single_case_backend(args: argparse.Namespace, case: DonorCaseSpec, backend: str | None = None, *, base_outdir: Path) -> dict[str, object]:
    case_args = _case_args(args, case=case, backend=backend)
    outdir = base_outdir
    outdir.mkdir(parents=True, exist_ok=True)
    run_t0 = time.perf_counter()

    qp_root = resolve_qp_root(case_args.qp_root)
    prepared_meta: PreparedActiveSpace | None = None
    prep_dir: Path | None = None
    prep_ezfio: Path | None = None
    prep_meta_path: Path | None = None
    py_summary: dict[str, object]
    compare: dict[str, object] | None = None
    qp_summary: dict[str, object] | None = None

    reference_case_for_injection: N2ReferenceCase | None = None
    if case_args.parity_basis_mode == "inject_pyscf":
        reference_case_for_injection = build_n2_case(case)

    if case_args.with_qp and case_args.parity_basis_mode == "qp_fcidump":
        if qp_root is None:
            raise RuntimeError("No Quantum Package root found. Use --qp-root or source quantum_package.rc.")
        qp = discover_qp_paths(qp_root)
        prep_dir, prep_ezfio, prep_meta_path = _prepare_qp_workdir(
            case_args,
            qp,
            case_spec=case,
            inject_pyscf_basis=False,
            reference_case=None,
        )
        prepared_meta = load_prepared_active_space(prep_ezfio, fallback_metadata=prep_meta_path)
        fcidump_path = _generate_qp_fcidump(qp, prep_ezfio, outdir, case_spec=case)
        h1, eri8, ncas, nelec, ecore = _load_qp_fcidump_hamiltonian(fcidump_path)
        electronic_shift = float(ecore - _case_nuclear_repulsion(case))
        h1 = _apply_constant_shift_to_fixed_nelec_h1(h1, nelec=nelec, shift=electronic_shift)
        py_summary = python_trace_from_data(
            case_args,
            outdir,
            h1=h1,
            eri8=eri8,
            ncas=ncas,
            nelec=nelec,
            mo_label="QP_FCIDUMP_ELECTRONIC",
            hamiltonian_source="qp_fcidump_electronic",
            exact_hq_diagnostics=case.exact_hq_diagnostics,
        )
        qp_summary = run_qp_trace(case_args, outdir, prepared=(prep_dir, prep_ezfio))
        trace_path = outdir / "ccik_debug_trace.log"
        if trace_path.exists():
            compare = _compare_python_qp_vectors(
                outdir / "python_trace.jsonl",
                trace_path,
                ncore=prepared_meta.ncore,
                ncas=ncas,
            )
            (outdir / "python_qp_compare.json").write_text(json.dumps(compare, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        active = PreparedActiveSpace(
            mo_num=case.mo_num_hint,
            ncore=case.ncore_hint,
            ncas=case.ncas_hint,
            neleca=7,
            nelecb=7,
            nelecas=14 - 2 * case.ncore_hint,
            source="case_hint",
        )
        reference_case = build_n2_case(case, active=active)
        py_summary = python_trace_from_data(
            case_args,
            outdir,
            h1=reference_case.h1,
            eri8=reference_case.eri8,
            ncas=reference_case.ncas,
            nelec=reference_case.nelec,
            mo_label=reference_case.mo_label,
            hamiltonian_source="pyscf_rhf",
            exact_hq_diagnostics=case.exact_hq_diagnostics,
        )
        if case_args.with_qp:
            if qp_root is None:
                raise RuntimeError("No Quantum Package root found. Use --qp-root or source quantum_package.rc.")
            qp = discover_qp_paths(qp_root)
            prep_dir, prep_ezfio, prep_meta_path = _prepare_qp_workdir(
                case_args,
                qp,
                case_spec=case,
                inject_pyscf_basis=True,
                reference_case=reference_case_for_injection or reference_case,
            )
            prepared_meta = load_prepared_active_space(prep_ezfio, fallback_metadata=prep_meta_path)
            qp_summary = run_qp_trace(case_args, outdir, prepared=(prep_dir, prep_ezfio))
            trace_path = outdir / "ccik_debug_trace.log"
            if trace_path.exists():
                compare = _compare_python_qp_vectors(
                    outdir / "python_trace.jsonl",
                    trace_path,
                    ncore=prepared_meta.ncore,
                    ncas=reference_case.ncas,
                )
                (outdir / "python_qp_compare.json").write_text(json.dumps(compare, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if prep_ezfio is not None and not case_args.keep_ezfio:
        if prep_ezfio.exists():
            shutil.rmtree(prep_ezfio)
        if prep_dir is not None:
            shutil.rmtree(prep_dir, ignore_errors=True)

    final_python = py_summary["summary"]["ritz_energy"]
    final_qp = None if qp_summary is None else qp_summary["final_electronic_energy"]
    final_gap_mEh = None if final_qp is None else float((final_qp - final_python) * 1000.0)
    final_step_key = None
    if compare is not None and compare.get("steps"):
        final_step_key = str(max(int(key) for key in compare["steps"]))

    summary = {
        "case": case.name,
        "geometry_label": case.geometry_label,
        "basis": case.basis,
        "steps": int(case_args.steps),
        "selector_mode": case_args.selector_mode,
        "matvec_backend": case_args.matvec_backend,
        "exact_hq_diagnostics": bool(case.exact_hq_diagnostics),
        "python_ritz_energy": float(final_python),
        "qp_final_electronic_energy": None if final_qp is None else float(final_qp),
        "energy_gap_mEh": final_gap_mEh,
        "wall_time_s": float(time.perf_counter() - run_t0),
        "python_wall_time_s": float(py_summary["wall_time_s"]),
        "qp_wall_time_s": None if qp_summary is None else float(qp_summary["wall_time_s"]),
        "final_step": final_step_key,
        "final_step_qnext_l2": None if final_step_key is None else compare["steps"][final_step_key]["q_next"]["l2_diff_on_traced_union"],
        "final_step_projected_l2": None if final_step_key is None else compare["steps"][final_step_key]["projected_hq"]["l2_diff_on_traced_union"],
        "final_step_score_l2": None if final_step_key is None else compare["steps"][final_step_key]["score_vec"]["l2_diff_on_traced_union"],
        "prepared_active_space": None if prepared_meta is None else {
            "mo_num": prepared_meta.mo_num,
            "ncore": prepared_meta.ncore,
            "ncas": prepared_meta.ncas,
            "neleca": prepared_meta.neleca,
            "nelecb": prepared_meta.nelecb,
            "nelecas": prepared_meta.nelecas,
            "source": prepared_meta.source,
        },
        "hq_steps": [] if qp_summary is None else [
            {"step": int(step), **step_data["hq_meta"], **({"next_meta": step_data["next_meta"]} if "next_meta" in step_data else {})}
            for step, step_data in sorted(qp_summary["hq_steps"].items())
            if "hq_meta" in step_data
        ],
        "artifacts": {
            "python_summary": str(outdir / "python_summary.json"),
            "qp_summary": None if qp_summary is None else str(outdir / "qp_summary.json"),
            "compare": None if compare is None else str(outdir / "python_qp_compare.json"),
        },
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _sweep_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    flat: list[dict[str, object]] = []
    for row in rows:
        flat.append(
            {
                "case": row["case"],
                "backend": row["matvec_backend"],
                "steps": row["steps"],
                "exact_hq_diagnostics": row["exact_hq_diagnostics"],
                "python_ritz_energy": row["python_ritz_energy"],
                "qp_final_electronic_energy": row["qp_final_electronic_energy"],
                "energy_gap_mEh": row["energy_gap_mEh"],
                "wall_time_s": row["wall_time_s"],
                "qp_wall_time_s": row["qp_wall_time_s"],
                "final_step": row["final_step"],
                "final_step_qnext_l2": row["final_step_qnext_l2"],
                "final_step_projected_l2": row["final_step_projected_l2"],
                "final_step_score_l2": row["final_step_score_l2"],
            }
        )
    return flat


def _evaluate_release_gates(rows: list[dict[str, object]]) -> dict[str, object]:
    by_case_backend = {(row["case"], row["matvec_backend"]): row for row in rows}
    parity_row = by_case_backend.get(("n2_cas10_10_R1p10", "connected_sd_fullq"))
    parity_pass = (
        parity_row is not None
        and parity_row["final_step"] == "4"
        and float(parity_row["final_step_qnext_l2"]) <= 1.0e-5
        and abs(float(parity_row["energy_gap_mEh"])) <= 0.2
    )

    donor_energy_gate = True
    donor_runtime_gate = True
    case_checks: list[dict[str, object]] = []
    for case_name in DONOR_N2_SWEEP:
        legacy = by_case_backend.get((case_name, "legacy_pt2"))
        fullq = by_case_backend.get((case_name, "connected_sd_fullq"))
        if legacy is None or fullq is None:
            donor_energy_gate = False
            donor_runtime_gate = False
            case_checks.append({"case": case_name, "status": "missing"})
            continue
        energy_ok = float(fullq["qp_final_electronic_energy"]) <= float(legacy["qp_final_electronic_energy"]) + 1.0e-10
        runtime_ok = float(fullq["wall_time_s"]) <= 2.5 * float(legacy["wall_time_s"])
        donor_energy_gate = donor_energy_gate and energy_ok
        donor_runtime_gate = donor_runtime_gate and runtime_ok
        case_checks.append(
            {
                "case": case_name,
                "legacy_energy": legacy["qp_final_electronic_energy"],
                "connected_sd_fullq_energy": fullq["qp_final_electronic_energy"],
                "energy_ok": energy_ok,
                "legacy_wall_time_s": legacy["wall_time_s"],
                "connected_sd_fullq_wall_time_s": fullq["wall_time_s"],
                "runtime_ok": runtime_ok,
            }
        )

    return {
        "parity_gate": {
            "passed": bool(parity_pass),
            "required_case": "n2_cas10_10_R1p10",
            "max_step4_qnext_l2": 1.0e-5,
            "max_abs_energy_gap_mEh": 0.2,
            "observed": None if parity_row is None else {
                "final_step": parity_row["final_step"],
                "final_step_qnext_l2": parity_row["final_step_qnext_l2"],
                "energy_gap_mEh": parity_row["energy_gap_mEh"],
            },
        },
        "donor_energy_gate": {"passed": bool(donor_energy_gate), "cases": case_checks},
        "runtime_gate": {"passed": bool(donor_runtime_gate), "max_ratio": 2.5, "cases": case_checks},
        "flip_default": bool(parity_pass and donor_energy_gate and donor_runtime_gate),
    }


def _write_sweep_outputs(outdir: Path, rows: list[dict[str, object]], gates: dict[str, object]) -> None:
    flat_rows = _sweep_summary_rows(rows)
    csv_path = outdir / "results.csv"
    if flat_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    gate_path = outdir / "release_gate.json"
    gate_path.write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_lines = [
        "# Donor-Backed QP Parity Sweep",
        "",
        f"- Cases: {', '.join(DONOR_N2_SWEEP)}",
        f"- Default flip approved: {gates['flip_default']}",
        f"- Parity gate passed: {gates['parity_gate']['passed']}",
        f"- Donor energy gate passed: {gates['donor_energy_gate']['passed']}",
        f"- Runtime gate passed: {gates['runtime_gate']['passed']}",
        "",
        "## Runs",
        "",
    ]
    for row in flat_rows:
        summary_lines.append(
            f"- `{row['case']}` / `{row['backend']}`: "
            f"E_qp = {row['qp_final_electronic_energy']}, "
            f"gap = {row['energy_gap_mEh']} mEh, "
            f"wall = {row['wall_time_s']:.3f} s, "
            f"step-{row['final_step']} q_next L2 = {row['final_step_qnext_l2']}"
        )
    (outdir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def run_donor_sweep(args: argparse.Namespace, *, outdir: Path) -> dict[str, object]:
    if not args.with_qp:
        args = argparse.Namespace(**{**vars(args), "with_qp": True, "parity_basis_mode": "qp_fcidump"})

    rows: list[dict[str, object]] = []
    for case_name in DONOR_N2_SWEEP:
        case = DONOR_N2_CASES[case_name]
        for backend in ("legacy_pt2", "connected_sd_fullq"):
            run_dir = outdir / case.name / backend
            rows.append(_run_single_case_backend(args, case, backend, base_outdir=run_dir))

    gates = _evaluate_release_gates(rows)
    _write_sweep_outputs(outdir, rows, gates)
    return {"rows": rows, "release_gates": gates}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.with_qp:
        qp_root = resolve_qp_root(args.qp_root)
        if qp_root is None:
            raise RuntimeError("No Quantum Package root found. Use --qp-root or source quantum_package.rc.")
        qp = discover_qp_paths(qp_root)
        rebuild_summary = rebuild_ccik_plugin(qp, clean=True, sync_workspace=True)
        (outdir / "qp_plugin_rebuild.json").write_text(
            json.dumps(
                {
                    "plugin_dir": rebuild_summary["plugin_dir"],
                    "command": rebuild_summary["command"],
                    "returncode": rebuild_summary["returncode"],
                    "binary_path": rebuild_summary["binary_path"],
                    "binary_exists": rebuild_summary["binary_exists"],
                    "sync_summary": rebuild_summary["sync_summary"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.sweep == "donor_n2":
        payload = run_donor_sweep(args, outdir=outdir)
        print(json.dumps(payload["release_gates"], indent=2))
        return 0

    case = _resolve_requested_cases(args)[0]
    if args.with_qp and args.parity_basis_mode == "qp_fcidump":
        args = argparse.Namespace(**{**vars(args), "with_qp": True})
    summary = _run_single_case_backend(args, case, args.matvec_backend, base_outdir=outdir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
