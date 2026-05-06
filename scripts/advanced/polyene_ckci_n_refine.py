#!/usr/bin/env python3
"""Offline CKCI-n refinement for a finished cached-reference polyene CK run.

This script reads the final CK determinant pool from a saved support snapshot,
reuses the cached DMRG reference and FCIDUMP Hamiltonian, and builds the
determinant-count trajectory

    CK -> CKCI-0 -> CKCI-1 -> CKCI-2

inside a per-case ``ckci_n/`` sidecar directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
SHARED = SCRIPTS / "_shared"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import debug_selector_parity as dsp  # noqa: E402
import n2_thick_iteration_study as n2_study  # noqa: E402
from ccik.sparse_ci import SparseCIVector, normalize_sparse, sparse_from_mapping  # noqa: E402
from ccik.thick_restart import (  # noqa: E402
    _exactize_support_union,
    _one_shell_support_from_parents,
    _top_parent_detkeys_from_exactized,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_qp_python_imports(qp_root: Path):
    candidates = [
        qp_root / "external" / "ezfio" / "Python",
        qp_root / "external" / "Python",
        qp_root / "scripts",
        qp_root / "scripts" / "ezfio_interface",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import ezfio  # type: ignore

    return ezfio.ezfio


def _runtime_ezfio_path(case_dir: Path, run_name: str, ezfio_name: str) -> Path:
    return case_dir / run_name / "runtime" / ezfio_name


def _support_snapshot_path(runtime_ezfio: Path) -> Path:
    return runtime_ezfio / "work" / "ccik_final_support_union.txt"


def _pack_det_chunks(chunks: list[int], *, bits_per_chunk: int) -> int:
    packed = 0
    mask = (1 << bits_per_chunk) - 1
    for idx, raw in enumerate(chunks):
        chunk = int(raw) & mask
        packed |= chunk << (bits_per_chunk * idx)
    return int(packed)


def _load_saved_wavefunction(runtime_ezfio: Path, *, qp_root: Path, state_index: int = 0) -> tuple[SparseCIVector, int]:
    ezfio = _ensure_qp_python_imports(qp_root)
    ezfio.set_file(str(runtime_ezfio))
    n_det = int(ezfio.get_determinants_n_det())
    n_states = int(ezfio.get_determinants_n_states())
    if state_index < 0 or state_index >= n_states:
        raise IndexError(f"Requested state_index={state_index} but runtime EZFIO only has {n_states} states")
    if not bool(ezfio.has_determinants_psi_det()) or not bool(ezfio.has_determinants_psi_coef()):
        raise FileNotFoundError(
            "Runtime EZFIO does not contain a saved final wavefunction yet. "
            f"Missing determinants/psi_det.gz or determinants/psi_coef.gz in {runtime_ezfio}"
        )
    n_int = int(ezfio.get_determinants_n_int())
    bytes_per_chunk = int(ezfio.get_determinants_bit_kind())
    bits_per_chunk = 8 * bytes_per_chunk
    psi_det = ezfio.get_determinants_psi_det()
    psi_coef = ezfio.get_determinants_psi_coef()
    coeffs: dict[tuple[int, int], float] = {}
    for idx in range(n_det):
        alpha_chunks = [int(x) for x in psi_det[idx][0][:n_int]]
        beta_chunks = [int(x) for x in psi_det[idx][1][:n_int]]
        coeff = float(psi_coef[state_index][idx])
        if abs(coeff) <= 0.0:
            continue
        coeffs[(
            _pack_det_chunks(alpha_chunks, bits_per_chunk=bits_per_chunk),
            _pack_det_chunks(beta_chunks, bits_per_chunk=bits_per_chunk),
        )] = coeff
    return normalize_sparse(sparse_from_mapping(coeffs)), n_det


def _load_support_snapshot(path: Path) -> list[tuple[int, int]]:
    header, *body = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header_parts = header.split()
    if len(header_parts) != 2:
        raise RuntimeError(f"Malformed support snapshot header in {path}: {header!r}")
    n_int = int(header_parts[0])
    support_size = int(header_parts[1])
    keys: list[tuple[int, int]] = []
    bits_per_chunk = 64
    for idx, line in enumerate(body, start=1):
        pieces = [int(piece) for piece in line.split()]
        if len(pieces) != 2 * n_int:
            raise RuntimeError(
                f"Malformed support snapshot line {idx} in {path}: expected {2 * n_int} integers, got {len(pieces)}"
            )
        alpha = _pack_det_chunks(pieces[:n_int], bits_per_chunk=bits_per_chunk)
        beta = _pack_det_chunks(pieces[n_int:], bits_per_chunk=bits_per_chunk)
        keys.append((alpha, beta))
    if len(keys) != support_size:
        raise RuntimeError(
            f"Support snapshot size mismatch in {path}: header says {support_size}, parsed {len(keys)} determinants"
        )
    return sorted(keys, key=lambda key: (int(key[0]), int(key[1])))


def _build_initial_guess(vec: SparseCIVector, support_keys: list[tuple[int, int]]) -> np.ndarray | None:
    if not vec.coeffs or not support_keys:
        return None
    guess = np.asarray([float(vec.coeffs.get(key, 0.0)) for key in support_keys], dtype=float)
    norm = float(np.linalg.norm(guess))
    if norm <= 0.0:
        return None
    return guess / norm


def _project_exactized_guess(
    *,
    previous_support: list[tuple[int, int]],
    previous_coeffs: np.ndarray,
    expanded_support: list[tuple[int, int]],
) -> np.ndarray | None:
    if len(previous_support) != int(previous_coeffs.shape[0]):
        raise ValueError("Previous exactized coefficient vector does not match support size")
    coeff_map = {
        (int(key[0]), int(key[1])): float(previous_coeffs[idx])
        for idx, key in enumerate(previous_support)
    }
    guess = np.asarray([float(coeff_map.get(key, 0.0)) for key in expanded_support], dtype=float)
    norm = float(np.linalg.norm(guess))
    if norm <= 0.0:
        return None
    return guess / norm


def _expand_support_once(
    *,
    support_keys: list[tuple[int, int]],
    exactized_coeffs: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    topn: int,
) -> tuple[list[tuple[int, int]], int, int]:
    parents = _top_parent_detkeys_from_exactized(
        support_keys=support_keys,
        coeffs=np.asarray(exactized_coeffs, dtype=float),
        topn=topn,
    )
    shell_support = _one_shell_support_from_parents(
        parents=parents,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
    )
    base_support = {(int(stra), int(strb)) for stra, strb in support_keys}
    expanded_support = sorted(base_support | shell_support, key=lambda key: (int(key[0]), int(key[1])))
    shell_added = int(len(set(expanded_support) - base_support))
    return expanded_support, int(len(parents)), shell_added


def _stage_row(
    *,
    stage_label: str,
    determinant_count: int,
    total_energy_eh: float,
    reference_total_eh: float,
    solver_kind: str,
    hamiltonian_nnz: int | None,
    parent_count: int | None,
    shell_added: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta_mha = float(1000.0 * (float(total_energy_eh) - float(reference_total_eh)))
    row = {
        "stage_label": str(stage_label),
        "determinant_count": int(determinant_count),
        "total_energy_eh": float(total_energy_eh),
        "delta_mHa": float(delta_mha),
        "abs_delta_mHa": float(abs(delta_mha)),
        "solver_kind": str(solver_kind),
        "hamiltonian_nnz": None if hamiltonian_nnz is None else int(hamiltonian_nnz),
        "parent_count": None if parent_count is None else int(parent_count),
        "shell_added": None if shell_added is None else int(shell_added),
    }
    if extra:
        row.update(extra)
    return row


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stage_label",
        "determinant_count",
        "total_energy_eh",
        "delta_mHa",
        "abs_delta_mHa",
        "solver_kind",
        "hamiltonian_nnz",
        "parent_count",
        "shell_added",
        "support_source",
        "trace_support_union_size",
        "saved_wf_n_det",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_caption(path: Path, *, case_label: str, topn: int, max_n: int) -> None:
    text = (
        f"CKCI-$n$ refinement for {case_label}. The black CK point is the final thick-restart "
        "compressed Krylov determinant pool, while the colored CKCI-$n$ points are obtained by "
        "exact diagonalization in that pool (CKCI-0) followed by one-shell expansions from the "
        f"top {topn} amplitudes and re-exactization through CKCI-{max_n}. The horizontal axis "
        "reports the number of determinants used at each stage."
    )
    path.write_text(text + "\n", encoding="utf-8")


def _plot_stages(path_png: Path, path_pdf: Path, *, case_label: str, rows: list[dict[str, Any]]) -> None:
    xs = [max(1, int(row["determinant_count"])) for row in rows]
    ys = [max(1.0e-12, float(row["abs_delta_mHa"])) for row in rows]
    labels = [str(row["stage_label"]) for row in rows]
    colors = ["#111827", "#d97706", "#dc2626", "#2563eb"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(xs, ys, color="#4b5563", linewidth=1.8, alpha=0.75, zorder=1)
    for idx, row in enumerate(rows):
        below_reference = row.get("delta_mHa") is not None and float(row["delta_mHa"]) < -1.0e-9
        ax.scatter(
            [xs[idx]],
            [ys[idx]],
            s=82,
            marker="x" if below_reference else markers[idx % len(markers)],
            color="#b91c1c" if below_reference else colors[idx % len(colors)],
            edgecolor="none" if below_reference else "white",
            linewidth=1.4 if below_reference else 0.8,
            zorder=3,
            label=labels[idx],
        )
        ax.annotate(
            labels[idx],
            (xs[idx], ys[idx]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color="#b91c1c" if below_reference else colors[idx % len(colors)],
            weight="semibold",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Determinants")
    ax.set_ylabel(r"$|\Delta E|$ (mHa)")
    ax.set_title(case_label)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    handles, handle_labels = ax.get_legend_handles_labels()
    dedup_handles: list[Any] = []
    dedup_labels: list[str] = []
    for handle, label in zip(handles, handle_labels, strict=False):
        if label in dedup_labels:
            continue
        dedup_handles.append(handle)
        dedup_labels.append(label)
    ax.legend(dedup_handles, dedup_labels, frameon=False, loc="best")
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)


def run_refinement(case_dir: Path, *, run_name: str, topn: int, max_n: int) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    input_path = case_dir / "input.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input.json in {case_dir}")
    spec = n2_study.load_study_input(input_path)
    system_setup = _load_json(case_dir / "system_setup" / "system_setup.json")
    reference_choice = _load_json(case_dir / "system_setup" / "reference_choice.json")

    qp_root = Path(str(spec["study"]["qp_root"])).expanduser().resolve()
    runtime_ezfio = _runtime_ezfio_path(case_dir, run_name, Path(str(system_setup["ezfio_path"])).name)
    if not runtime_ezfio.exists():
        raise FileNotFoundError(f"Missing runtime EZFIO for run {run_name}: {runtime_ezfio}")

    run_dir = case_dir / run_name
    _, trace_json = n2_study._trace_paths(run_dir)
    rows = n2_study.load_trace_rows_from_json(trace_json)
    if not rows:
        raise FileNotFoundError(f"Missing iteration trace for run {run_name} in {case_dir}")
    final_ck = dict(rows[-1])
    trace_support_union_size = (
        None if final_ck.get("support_union_size") is None else int(final_ck["support_union_size"])
    )

    saved_wf, saved_wf_n_det = _load_saved_wavefunction(runtime_ezfio, qp_root=qp_root, state_index=0)
    support_snapshot = _support_snapshot_path(runtime_ezfio)
    support_source = "saved_final_wf"
    if support_snapshot.exists():
        support_keys = _load_support_snapshot(support_snapshot)
        support_source = "final_support_snapshot"
    else:
        support_keys = sorted(saved_wf.coeffs, key=lambda key: (int(key[0]), int(key[1])))
    if not support_keys:
        raise RuntimeError(f"Resolved an empty final CK support for {run_name} in {case_dir}")

    if trace_support_union_size is not None and support_source == "final_support_snapshot":
        if int(trace_support_union_size) != int(len(support_keys)):
            raise RuntimeError(
                "Final support snapshot size does not match the last recorded support_union_size. "
                f"trace={trace_support_union_size} snapshot={len(support_keys)}"
            )

    fcidump_path = Path(str(system_setup["fcidump_path"]))
    h1, eri8, norb, nelec, _ecore = dsp._load_qp_fcidump_hamiltonian(fcidump_path)
    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri8, int(norb), tuple(nelec), fac=0.5)
    reference_total_eh = float(reference_choice["reference_total_energy_eh"])

    ck_rows: list[dict[str, Any]] = [
        _stage_row(
            stage_label="CK",
            determinant_count=(
                int(trace_support_union_size)
                if trace_support_union_size is not None and support_source == "final_support_snapshot"
                else int(len(support_keys))
            ),
            total_energy_eh=float(final_ck["total_energy_eh"]),
            reference_total_eh=reference_total_eh,
            solver_kind="qp_ccik",
            hamiltonian_nnz=None,
            parent_count=None,
            shell_added=None,
            extra={
                "support_source": support_source,
                "trace_support_union_size": trace_support_union_size,
                "saved_wf_n_det": int(saved_wf_n_det),
            },
        )
    ]

    initial_guess = _build_initial_guess(saved_wf, support_keys)
    stage0 = _exactize_support_union(
        support_keys=support_keys,
        h2eff=h2eff,
        norb=int(norb),
        nelec=(int(nelec[0]), int(nelec[1])),
        initial_guess=initial_guess,
    )
    ck_rows.append(
        _stage_row(
            stage_label="CKCI-0",
            determinant_count=int(len(support_keys)),
            total_energy_eh=float(stage0["electronic_energy_eh"]) + float(system_setup["ecore_eh"]),
            reference_total_eh=reference_total_eh,
            solver_kind=str(stage0["solver_kind"]),
            hamiltonian_nnz=None if stage0["hamiltonian_nnz"] is None else int(stage0["hamiltonian_nnz"]),
            parent_count=None,
            shell_added=None,
            extra={
                "support_source": support_source,
                "trace_support_union_size": trace_support_union_size,
                "saved_wf_n_det": int(saved_wf_n_det),
            },
        )
    )

    previous_support = list(support_keys)
    previous_coeffs = np.asarray(stage0["lowest_eigenvector"], dtype=float)
    for level in range(1, int(max_n) + 1):
        expanded_support, parent_count, shell_added = _expand_support_once(
            support_keys=previous_support,
            exactized_coeffs=previous_coeffs,
            h2eff=h2eff,
            norb=int(norb),
            nelec=(int(nelec[0]), int(nelec[1])),
            topn=int(topn),
        )
        guess = _project_exactized_guess(
            previous_support=previous_support,
            previous_coeffs=previous_coeffs,
            expanded_support=expanded_support,
        )
        stage = _exactize_support_union(
            support_keys=expanded_support,
            h2eff=h2eff,
            norb=int(norb),
            nelec=(int(nelec[0]), int(nelec[1])),
            initial_guess=guess,
        )
        ck_rows.append(
            _stage_row(
                stage_label=f"CKCI-{level}",
                determinant_count=int(len(expanded_support)),
                total_energy_eh=float(stage["electronic_energy_eh"]) + float(system_setup["ecore_eh"]),
                reference_total_eh=reference_total_eh,
                solver_kind=str(stage["solver_kind"]),
                hamiltonian_nnz=None if stage["hamiltonian_nnz"] is None else int(stage["hamiltonian_nnz"]),
                parent_count=int(parent_count),
                shell_added=int(shell_added),
                extra={
                    "support_source": support_source,
                    "trace_support_union_size": trace_support_union_size,
                    "saved_wf_n_det": int(saved_wf_n_det),
                },
            )
        )
        previous_support = list(expanded_support)
        previous_coeffs = np.asarray(stage["lowest_eigenvector"], dtype=float)

    output_dir = case_dir / "ckci_n"
    _write_summary_csv(output_dir / "ckci_n_summary.csv", ck_rows)
    summary_payload = {
        "case_dir": str(case_dir),
        "input_path": str(input_path),
        "run_name": str(run_name),
        "case_label": str(spec["label"]),
        "reference_total_energy_eh": float(reference_total_eh),
        "topn": int(topn),
        "max_n": int(max_n),
        "runtime_ezfio": str(runtime_ezfio),
        "support_snapshot_path": None if not support_snapshot.exists() else str(support_snapshot),
        "support_source": support_source,
        "trace_support_union_size": trace_support_union_size,
        "saved_wf_n_det": int(saved_wf_n_det),
        "stages": ck_rows,
    }
    _write_summary_json(output_dir / "ckci_n_summary.json", summary_payload)
    _plot_stages(
        output_dir / "ckci_n_plot.png",
        output_dir / "ckci_n_plot.pdf",
        case_label=str(spec["label"]),
        rows=ck_rows,
    )
    _write_caption(
        output_dir / "ckci_n_caption.tex",
        case_label=str(spec["label"]),
        topn=int(topn),
        max_n=int(max_n),
    )
    return summary_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build offline CKCI-n sidecars for a finished polyene CK run")
    parser.add_argument("--case", type=Path, required=True, help="Case directory containing input.json")
    parser.add_argument("--run", type=str, required=True, help="Run directory name, e.g. baseline_m031_ck")
    parser.add_argument("--topn", type=int, default=512, help="Number of top parents used per shell expansion")
    parser.add_argument("--max-n", type=int, default=2, help="Maximum CKCI order beyond CKCI-0")
    args = parser.parse_args(argv)

    payload = run_refinement(args.case, run_name=str(args.run), topn=int(args.topn), max_n=int(args.max_n))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
