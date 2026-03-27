"""Large CCIK / CCIK-thick sweep on the donor-backed N2 CAS(12,12) FCIDUMP.

This script uses the maintained Python implementation on the exact same
Hamiltonian already exported from Quantum Package for the N2 CAS(12,12) point.
It is intended to get the best practical CCIK numbers for this case without
waiting on a buffered long QP plugin run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump as pyscf_fcidump  # noqa: E402

from ccik import ccik_ground_energy_dense, ccik_ground_energy_dense_thick_restart  # noqa: E402
from ccik.params import CCIKParams, CCIKThickRestartParams  # noqa: E402


DEFAULT_EXPERIMENT = _ROOT / "experiments" / "n2_cas12_method_compare_20260324"
DEFAULT_FCIDUMP = DEFAULT_EXPERIMENT / "qp_active.FCIDUMP"
DEFAULT_EXACT = DEFAULT_EXPERIMENT / "exact_fci_summary.json"
DEFAULT_OUTDIR = _ROOT / "experiments" / "n2_cas12_ccik_python_sweep_20260324"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a large Python CCIK sweep on N2 CAS(12,12)")
    p.add_argument("--fcidump", type=Path, default=DEFAULT_FCIDUMP)
    p.add_argument("--exact-summary", type=Path, default=DEFAULT_EXACT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--dense-label", type=str, default=None)
    p.add_argument("--dense-offset", type=int, default=0)
    p.add_argument("--dense-rungs", type=int, default=3)
    p.add_argument("--thick-label", type=str, default=None)
    p.add_argument("--thick-offset", type=int, default=0)
    p.add_argument("--thick-rungs", type=int, default=3)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_hamiltonian(fcidump_path: Path) -> tuple[Any, Any, int, tuple[int, int], float]:
    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec_tot = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    neleca = (nelec_tot + ms2) // 2
    nelecb = nelec_tot - neleca
    ecore = float(data.get("ECORE", 0.0))
    h1 = data["H1"]
    eri8 = ao2mo.restore(8, data["H2"], norb)
    return h1, eri8, norb, (int(neleca), int(nelecb)), ecore


def _load_exact_total(exact_summary: Path) -> float:
    payload = json.loads(exact_summary.read_text(encoding="utf-8"))
    return float(payload["total_energy_eh"])


def _dense_ladder(verbose: bool) -> list[tuple[str, CCIKParams]]:
    common = dict(
        candidate_mode="full",
        selector_mode="abs_residual",
        selector_depth=1,
        generator_keep=256,
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=256,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=4096,
        exact_hq_diagnostics=False,
        orth_tol=1e-12,
        verbose=verbose,
    )
    weak = dict(common)
    weak["tmp_keep_factor"] = 2
    weak["preserve_prev_support_cap"] = 16384
    return [
        ("dense_full_m15_weak", CCIKParams(m=15, nadd=50000, nkeep=120000, Kv=30000, **weak)),
        ("dense_full_m40_weak", CCIKParams(m=40, nadd=40000, nkeep=80000, Kv=20000, **weak)),
        ("dense_full_m60_vweak", CCIKParams(m=60, nadd=60000, nkeep=120000, Kv=30000, **weak)),
        ("dense_full_m80_ultra", CCIKParams(m=80, nadd=80000, nkeep=160000, Kv=40000, **weak)),
        ("dense_full_r1", CCIKParams(m=24, nadd=8000, nkeep=16000, Kv=4000, **common)),
        ("dense_full_r2", CCIKParams(m=32, nadd=12000, nkeep=24000, Kv=8000, **common)),
        ("dense_full_r3", CCIKParams(m=40, nadd=20000, nkeep=40000, Kv=12000, **common)),
        ("dense_full_r4", CCIKParams(m=48, nadd=40000, nkeep=80000, Kv=20000, **common)),
    ]


def _thick_ladder(verbose: bool) -> list[tuple[str, CCIKThickRestartParams]]:
    common = dict(
        candidate_mode="full",
        selector_mode="abs_residual",
        selector_depth=1,
        generator_keep=256,
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=256,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=4096,
        exact_hq_diagnostics=False,
        orth_tol=1e-12,
        verbose=verbose,
        tol=1e-7,
    )
    weak = dict(common)
    weak["tmp_keep_factor"] = 2
    weak["preserve_prev_support_cap"] = 16384
    return [
        ("thick_full_m15_weak", CCIKThickRestartParams(m_cycle=15, ncycles=20, nroot=4, nadd=50000, nkeep=120000, Kv=30000, **weak)),
        ("thick_full_m30_vweak", CCIKThickRestartParams(m_cycle=30, ncycles=30, nroot=6, nadd=60000, nkeep=120000, Kv=30000, **weak)),
        ("thick_full_m40_ultra", CCIKThickRestartParams(m_cycle=40, ncycles=40, nroot=8, nadd=80000, nkeep=160000, Kv=40000, **weak)),
        ("thick_full_r1", CCIKThickRestartParams(m_cycle=10, ncycles=16, nroot=3, nadd=8000, nkeep=16000, Kv=4000, **common)),
        ("thick_full_r2", CCIKThickRestartParams(m_cycle=16, ncycles=20, nroot=4, nadd=12000, nkeep=24000, Kv=8000, **common)),
        ("thick_full_r3", CCIKThickRestartParams(m_cycle=20, ncycles=24, nroot=6, nadd=20000, nkeep=40000, Kv=12000, **common)),
        ("thick_full_r4", CCIKThickRestartParams(m_cycle=24, ncycles=28, nroot=8, nadd=40000, nkeep=80000, Kv=20000, **common)),
    ]


def _run_dense(
    *,
    label: str,
    params: CCIKParams,
    h1: Any,
    eri8: Any,
    norb: int,
    nelec: tuple[int, int],
    ecore: float,
    exact_total: float,
) -> dict[str, object]:
    stats: dict[str, object] = {}
    t0 = time.perf_counter()
    e_cas = ccik_ground_energy_dense(h1, eri8, norb, nelec, params=params, stats=stats)
    wall_s = float(time.perf_counter() - t0)
    e_total = float(ecore + e_cas)
    return {
        "method": "ccik",
        "rung": label,
        "e_total": e_total,
        "abs_err_eh": abs(e_total - exact_total),
        "delta_vs_exact_mEh": 1000.0 * (e_total - exact_total),
        "wall_s": wall_s,
        "params": asdict(params),
        "stats": stats,
    }


def _run_thick(
    *,
    label: str,
    params: CCIKThickRestartParams,
    h1: Any,
    eri8: Any,
    norb: int,
    nelec: tuple[int, int],
    ecore: float,
    exact_total: float,
) -> dict[str, object]:
    stats: dict[str, object] = {}
    t0 = time.perf_counter()
    e_cas = ccik_ground_energy_dense_thick_restart(h1, eri8, norb, nelec, params=params, stats=stats)
    wall_s = float(time.perf_counter() - t0)
    e_total = float(ecore + e_cas)
    return {
        "method": "ccik_thick",
        "rung": label,
        "e_total": e_total,
        "abs_err_eh": abs(e_total - exact_total),
        "delta_vs_exact_mEh": 1000.0 * (e_total - exact_total),
        "wall_s": wall_s,
        "params": asdict(params),
        "stats": stats,
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.fcidump.exists():
        raise FileNotFoundError(args.fcidump)
    if not args.exact_summary.exists():
        raise FileNotFoundError(args.exact_summary)

    h1, eri8, norb, nelec, ecore = _load_hamiltonian(args.fcidump)
    exact_total = _load_exact_total(args.exact_summary)

    results: dict[str, Any] = {
        "fcidump_path": str(args.fcidump.resolve()),
        "exact_summary_path": str(args.exact_summary.resolve()),
        "norb": int(norb),
        "nelec": [int(nelec[0]), int(nelec[1])],
        "ecore_eh": float(ecore),
        "exact_total_energy_eh": float(exact_total),
        "dense": [],
        "thick": [],
    }

    dense_all = _dense_ladder(args.verbose)
    thick_all = _thick_ladder(args.verbose)
    dense_map = dict(dense_all)
    thick_map = dict(thick_all)
    if args.dense_label is not None:
        if args.dense_label not in dense_map:
            raise KeyError(f"Unknown dense rung {args.dense_label!r}")
        dense_ladder = [(args.dense_label, dense_map[args.dense_label])]
    else:
        dense_ladder = dense_all[max(0, int(args.dense_offset)) : max(0, int(args.dense_offset)) + max(0, int(args.dense_rungs))]
    if args.thick_label is not None:
        if args.thick_label not in thick_map:
            raise KeyError(f"Unknown thick rung {args.thick_label!r}")
        thick_ladder = [(args.thick_label, thick_map[args.thick_label])]
    else:
        thick_ladder = thick_all[max(0, int(args.thick_offset)) : max(0, int(args.thick_offset)) + max(0, int(args.thick_rungs))]

    for label, params in dense_ladder:
        print(f"[dense] starting {label}", flush=True)
        result = _run_dense(
            label=label,
            params=params,
            h1=h1,
            eri8=eri8,
            norb=norb,
            nelec=nelec,
            ecore=ecore,
            exact_total=exact_total,
        )
        results["dense"].append(result)
        _write_json(args.outdir / "results_partial.json", results)
        print(
            f"[dense] finished {label}: E = {result['e_total']:.12f}  "
            f"delta = {result['delta_vs_exact_mEh']:.6f} mEh  wall = {result['wall_s']:.1f} s",
            flush=True,
        )

    for label, params in thick_ladder:
        print(f"[thick] starting {label}", flush=True)
        result = _run_thick(
            label=label,
            params=params,
            h1=h1,
            eri8=eri8,
            norb=norb,
            nelec=nelec,
            ecore=ecore,
            exact_total=exact_total,
        )
        results["thick"].append(result)
        _write_json(args.outdir / "results_partial.json", results)
        print(
            f"[thick] finished {label}: E = {result['e_total']:.12f}  "
            f"delta = {result['delta_vs_exact_mEh']:.6f} mEh  wall = {result['wall_s']:.1f} s",
            flush=True,
        )

    flat_rows: list[dict[str, object]] = []
    for family in ("dense", "thick"):
        for row in results[family]:
            flat_rows.append(
                {
                    "method": row["method"],
                    "rung": row["rung"],
                    "e_total": row["e_total"],
                    "abs_err_eh": row["abs_err_eh"],
                    "delta_vs_exact_mEh": row["delta_vs_exact_mEh"],
                    "wall_s": row["wall_s"],
                    "m_eff": row["stats"].get("m_eff") if isinstance(row.get("stats"), dict) else None,
                    "ndet_union": row["stats"].get("ndet_union") if isinstance(row.get("stats"), dict) else None,
                    "ndet_sum": row["stats"].get("ndet_sum") if isinstance(row.get("stats"), dict) else None,
                }
            )

    best_dense = min(results["dense"], key=lambda row: float(row["e_total"])) if results["dense"] else None
    best_thick = min(results["thick"], key=lambda row: float(row["e_total"])) if results["thick"] else None
    best_overall = min(
        [row for row in (best_dense, best_thick) if row is not None],
        key=lambda row: float(row["e_total"]),
    ) if (best_dense is not None or best_thick is not None) else None

    summary = {
        "exact_total_energy_eh": float(exact_total),
        "best_dense": best_dense,
        "best_thick": best_thick,
        "best_overall": best_overall,
    }
    _write_json(args.outdir / "results.json", results)
    _write_json(args.outdir / "summary.json", summary)
    _write_csv(args.outdir / "summary.csv", flat_rows)

    lines = [
        "# N2 CAS(12,12) Python CCIK Sweep",
        "",
        f"- FCIDUMP: `{args.fcidump}`",
        f"- Exact total energy: `{exact_total:.12f} Eh`",
        "",
        "## Rungs",
        "",
    ]
    for row in flat_rows:
        lines.append(
            f"- `{row['method']}` `{row['rung']}`: "
            f"`E = {float(row['e_total']):.12f} Eh`, "
            f"`Delta_exact = {float(row['delta_vs_exact_mEh']):+.6f} mEh`, "
            f"`wall = {float(row['wall_s']):.1f} s`"
        )
    if best_overall is not None:
        lines.extend(
            [
                "",
                "## Best",
                "",
                f"- `{best_overall['method']}` `{best_overall['rung']}`: "
                f"`E = {float(best_overall['e_total']):.12f} Eh`, "
                f"`Delta_exact = {float(best_overall['delta_vs_exact_mEh']):+.6f} mEh`",
            ]
        )
    (args.outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
