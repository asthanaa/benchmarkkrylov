"""Shared large-reference backends for QP study runners.

The study drivers use these helpers to choose an exact FCI, DMRG, or bounded
selected-CI reference without duplicating the disk-guard and logging logic in
every script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from pyscf.tools import fcidump as pyscf_fcidump

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qp_ccik_common import discover_qp_paths_from_root, ensure_ezfio_defaults, qp_parallel_env


# The large-space studies default to the local reproducible DMRG env created by
# `scripts/bootstrap_dmrg_env.sh`. Inputs may override this interpreter, but
# they do not need to hard-code a user-specific path.
DEFAULT_DMRG_PYTHON = ROOT / ".conda" / "cka-dmrg-py312" / "bin" / "python"
DEFAULT_DMRG_SCRATCH_ROOT = Path("/tmp/cka_dmrg")
DEFAULT_CIPSI_SCRATCH_ROOT = Path("/tmp/cka_cipsi")
DEFAULT_MAX_FILE_MB = 500
DEFAULT_TOTAL_SCRATCH_GB = 45.0
DEFAULT_CIPSI_RUNGS: tuple[int, ...] = (100_000, 300_000, 1_000_000, 2_000_000)
DEFAULT_PYSCF_CIPSI_CUTOFFS: tuple[float, ...] = (5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5, 2.0e-5)
DEFAULT_DMRG_BOND_DIMS: tuple[int, ...] = (500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6000)
DEFAULT_DMRG_NOISES: tuple[float, ...] = (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 0.0, 0.0, 0.0)
DEFAULT_DMRG_DAV_THRDS: tuple[float, ...] = (1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9, 1e-10, 1e-10)
DEFAULT_DMRG_PREFLIGHT_BOND_DIMS: tuple[int, ...] = (200, 200)
DEFAULT_DMRG_PREFLIGHT_NOISES: tuple[float, ...] = (1e-5, 0.0)
DEFAULT_DMRG_PREFLIGHT_DAV_THRDS: tuple[float, ...] = (1e-6, 1e-7)


def _fit_schedule_length(values: tuple[int | float, ...], target_len: int) -> list[int | float]:
    if target_len <= 0:
        raise ValueError("DMRG schedule must contain at least one stage")
    seq = list(values)
    if not seq:
        raise ValueError("DMRG schedule helper received an empty template")
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [seq[-1]] * (target_len - len(seq))


def normalize_reference_summary(summary: dict[str, object]) -> dict[str, object]:
    normalized = dict(summary)
    reference_kind = str(normalized.get("reference_kind") or normalized.get("method") or "").strip().lower()
    if reference_kind in {"", "dmrg"}:
        normalized["reference_kind"] = "dmrg"
        normalized.setdefault("reference_label", "DMRG")
    elif reference_kind in {"exact_fci", "fci"}:
        normalized["reference_kind"] = "exact_fci"
        normalized.setdefault("reference_label", "Exact FCI")
    elif reference_kind in {"cipsi", "variational_cipsi"}:
        normalized["reference_kind"] = "cipsi"
        normalized.setdefault("reference_label", "CIPSI variational")
    else:
        normalized.setdefault("reference_kind", reference_kind)
        normalized.setdefault("reference_label", str(normalized["reference_kind"]))
    return normalized


def hf_occ_vector(norb: int, nelec_tot: int) -> list[float]:
    if norb <= 0:
        return []
    occs = [0.0] * int(norb)
    ndoubly = max(0, int(nelec_tot) // 2)
    for i in range(min(int(norb), ndoubly)):
        occs[i] = 2.0
    if int(nelec_tot) % 2 == 1 and ndoubly < int(norb):
        occs[ndoubly] = 1.0
    return occs


def _normalized_c1_isym(value: object) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 1
    return 1 if parsed == 0 else parsed


def _dmrg_reference_script() -> str:
    return r"""
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from pyblock2.gsdmrg import GSDMRG
from pyscf.tools import fcidump as pyscf_fcidump

fcidump_path = Path(os.environ["CKA_FCIDUMP"])
scratch_dir = Path(os.environ["CKA_SCRATCH"])
log_path = Path(os.environ["CKA_LOG"])
memory_gb = int(os.environ["CKA_MEMORY_GB"])
threads = int(os.environ["CKA_THREADS"])
conv_tol = float(os.environ["CKA_CONV_TOL"])
bond_dims = json.loads(os.environ["CKA_BOND_DIMS"])
noises = json.loads(os.environ["CKA_NOISES"])
dav_thrds = json.loads(os.environ["CKA_DAV_THRDS"])
n_steps = int(os.environ["CKA_N_STEPS"])
hf_occs = json.loads(os.environ["CKA_OCCS_JSON"])
preflight_bond_dims = json.loads(os.environ["CKA_PREFLIGHT_BOND_DIMS"])
preflight_noises = json.loads(os.environ["CKA_PREFLIGHT_NOISES"])
preflight_dav_thrds = json.loads(os.environ["CKA_PREFLIGHT_DAV_THRDS"])
preflight_n_steps = int(os.environ["CKA_PREFLIGHT_N_STEPS"])

data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
norb = int(data["NORB"])
nelec = int(data["NELEC"])
ms2 = int(data["MS2"])
orb_sym = [int(x) for x in data.get("ORBSYM", [1] * norb)]
isym = int(os.environ["CKA_ISYM"])
ecore = float(data.get("ECORE", 0.0))
t0 = time.perf_counter()
init_api = "explicit_hamiltonian"
init_mode = "fci_init"

result = {
    "reference_kind": "dmrg",
    "reference_label": "DMRG",
    "success": False,
    "init_api": init_api,
    "init_mode": init_mode,
    "norb": norb,
    "nelec": nelec,
    "ms2": ms2,
    "isym": isym,
    "orb_sym": orb_sym,
    "ecore_eh": ecore,
    "fcidump_path": str(fcidump_path),
    "log_path": str(log_path),
    "threads": threads,
    "memory_gb": memory_gb,
    "conv_tol": conv_tol,
    "bond_dims": bond_dims,
    "noises": noises,
    "dav_thrds": dav_thrds,
    "n_steps": n_steps,
    "preflight": {"success": False, "attempts": []},
}

def run_dmrg(occs, bias, label, stage_bond_dims, stage_noises, stage_dav_thrds, stage_n_steps):
    local_scratch = scratch_dir / label
    local_scratch.mkdir(parents=True, exist_ok=True)
    dmrg = GSDMRG(
        scratch=str(local_scratch),
        memory=float(memory_gb) * 1e9,
        omp_threads=int(threads),
        verbose=3,
        print_statistics=True,
    )
    # Use the explicit Hamiltonian init path. This lets us normalize the
    # FCIDUMP symmetry metadata ourselves and avoids the zero-norm initial MPS
    # failure that appeared on large N2 cases.
    dmrg.init_hamiltonian(
        "c1",
        n_sites=norb,
        n_elec=nelec,
        twos=ms2,
        isym=isym,
        orb_sym=orb_sym,
        e_core=ecore,
        h1e=data["H1"],
        g2e=data["H2"],
    )
    return float(
        dmrg.dmrg(
            bond_dims=stage_bond_dims,
            noises=stage_noises,
            dav_thrds=stage_dav_thrds,
            occs=occs,
            bias=bias,
            n_steps=stage_n_steps,
            conv_tol=conv_tol,
            cutoff=1e-14,
        )
    )

def record_attempt(label, occs, bias, stage_bond_dims, stage_noises, stage_dav_thrds, stage_n_steps):
    attempt = {
        "label": label,
        "bond_dims": stage_bond_dims,
        "n_steps": stage_n_steps,
    }
    try:
        energy = run_dmrg(occs, bias, label, stage_bond_dims, stage_noises, stage_dav_thrds, stage_n_steps)
        attempt["success"] = True
        attempt["energy_eh"] = float(energy)
        result["preflight"]["attempts"].append(attempt)
        return energy
    except Exception as exc:
        attempt["success"] = False
        attempt["error"] = repr(exc)
        attempt["traceback"] = traceback.format_exc()
        result["preflight"]["attempts"].append(attempt)
        raise

old_tmpdir = os.environ.get("TMPDIR")
os.environ["TMPDIR"] = str(scratch_dir)
failure_stage = "preflight"
try:
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                try:
                    record_attempt(
                        "preflight_fci_init",
                        None,
                        1.0,
                        preflight_bond_dims,
                        preflight_noises,
                        preflight_dav_thrds,
                        preflight_n_steps,
                    )
                    result["preflight"]["success"] = True
                    result["preflight"]["init_mode"] = "fci_init"
                    init_mode = "fci_init"
                    full_occs = None
                    full_bias = 1.0
                    full_label = "fci_init"
                except Exception as exc:
                    if "Cannot generate initial guess 0" not in str(exc):
                        raise
                    print("Retrying DMRG with HF occupation-number initialized MPS")
                    record_attempt(
                        "preflight_hf_occ",
                        hf_occs,
                        0.0,
                        preflight_bond_dims,
                        preflight_noises,
                        preflight_dav_thrds,
                        preflight_n_steps,
                    )
                    result["preflight"]["success"] = True
                    result["preflight"]["init_mode"] = "hf_occ_fallback"
                    init_mode = "hf_occ_fallback"
                    full_occs = hf_occs
                    full_bias = 0.0
                    full_label = "hf_occ_init"
                failure_stage = "full_run"
                e_total = run_dmrg(full_occs, full_bias, full_label, bond_dims, noises, dav_thrds, n_steps)
    except Exception as exc:
        result.update(
            {
                "success": False,
                "init_mode": init_mode,
                "failure": repr(exc),
                "stop_reason": f"{failure_stage}_failed",
                "wall_s": float(time.perf_counter() - t0),
                "first_failure": None
                if not result["preflight"]["attempts"]
                else result["preflight"]["attempts"][-1],
            }
        )
        print(json.dumps(result))
        sys.exit(2)
finally:
    if old_tmpdir is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = old_tmpdir

result.update(
    {
        "success": True,
        "init_mode": init_mode,
        "total_energy_eh": float(e_total),
        "electronic_energy_eh": float(e_total - ecore),
        "wall_s": float(time.perf_counter() - t0),
    }
)
print(json.dumps(result))
"""


def parse_cipsi_summaries(text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    current: dict[str, float | int] | None = None
    for line in text.splitlines():
        m = re.search(r"Summary at N_det =\s*(\d+)", line)
        if m:
            current = {"n_det": int(m.group(1))}
            rows.append(current)
            continue
        if current is None:
            continue
        m = re.search(r"# E\s+(-?\d+\.\d+)", line)
        if m and "E+PT2" not in line:
            current["e_var"] = float(m.group(1))
        m = re.search(r"# E\+PT2\s+(-?\d+\.\d+)", line)
        if m:
            current["e_pt2"] = float(m.group(1))
    return rows


def _largest_file_across(roots: list[Path]) -> tuple[Path | None, int]:
    largest_path: Path | None = None
    largest_size = 0
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            if size > largest_size:
                largest_path = candidate
                largest_size = size
    return largest_path, largest_size


def _total_size_across(roots: list[Path]) -> int:
    total_size = 0
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                total_size += int(candidate.stat().st_size)
            except OSError:
                continue
    return total_size


def run_capped_command(
    *,
    cmd: list[str],
    cwd: Path | None,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    watch_roots: list[Path],
    max_file_mb: int,
    max_total_scratch_gb: float,
    poll_s: float = 5.0,
) -> dict[str, object]:
    # Large references can create many scratch files quickly. We guard both the
    # single largest file and the total watched scratch size so a failed
    # reference attempt cannot quietly consume tens of GB on disk.
    max_file_bytes = int(max_file_mb) * 1024 * 1024
    max_total_bytes = int(float(max_total_scratch_gb) * 1024.0 * 1024.0 * 1024.0)
    largest_path: Path | None = None
    largest_size = 0
    peak_total_size = 0
    stop_reason: str | None = None
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=None if cwd is None else str(cwd),
            env=env,
            text=True,
            stdout=stdout_fh,
            stderr=stderr_fh,
        )
        while True:
            current_path, current_size = _largest_file_across(watch_roots)
            current_total_size = _total_size_across(watch_roots)
            if current_size > largest_size:
                largest_path = current_path
                largest_size = current_size
            if current_total_size > peak_total_size:
                peak_total_size = current_total_size
            if current_size > max_file_bytes:
                stop_reason = "max_file_mb_exceeded"
                proc.kill()
                proc.wait()
                break
            if current_total_size > max_total_bytes:
                stop_reason = "total_scratch_cap_exceeded"
                proc.kill()
                proc.wait()
                break
            if proc.poll() is not None:
                break
            time.sleep(poll_s)
        returncode = int(proc.wait())
    current_path, current_size = _largest_file_across(watch_roots)
    current_total_size = _total_size_across(watch_roots)
    if current_size > largest_size:
        largest_path = current_path
        largest_size = current_size
    if current_total_size > peak_total_size:
        peak_total_size = current_total_size
    return {
        "returncode": returncode,
        "stop_reason": stop_reason,
        "largest_file_path": None if largest_path is None else str(largest_path),
        "largest_file_mb": float(largest_size) / (1024.0 * 1024.0),
        "peak_total_scratch_gb": float(peak_total_size) / (1024.0 * 1024.0 * 1024.0),
        "max_file_mb": int(max_file_mb),
        "max_total_scratch_gb": float(max_total_scratch_gb),
    }


def load_or_compute_dmrg_reference_from_fcidump(
    *,
    fcidump_path: Path,
    outdir: Path,
    force: bool,
    memory_gb: int,
    threads: int,
    conv_tol: float,
    dmrg_python: Path,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
    max_total_scratch_gb: float = DEFAULT_TOTAL_SCRATCH_GB,
    scratch_root: Path = DEFAULT_DMRG_SCRATCH_ROOT,
    bond_dims: tuple[int, ...] | None = None,
    noises: tuple[float, ...] | None = None,
    dav_thrds: tuple[float, ...] | None = None,
    n_steps: int = 40,
) -> dict[str, object]:
    path = outdir / "reference_dmrg_summary.json"
    attempt_path = outdir / "reference_dmrg_attempt.json"
    if path.exists() and not force:
        payload = normalize_reference_summary(json.loads(path.read_text(encoding="utf-8")))
        payload["success"] = True
        return payload
    if not dmrg_python.exists():
        payload = {
            "success": False,
            "reference_kind": "dmrg",
            "reference_label": "DMRG",
            "failure": f"DMRG Python interpreter not found: {dmrg_python}",
            "stop_reason": "missing_dmrg_python",
            "max_file_mb": int(max_file_mb),
            "max_total_scratch_gb": float(max_total_scratch_gb),
        }
        attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    runtime_dir = outdir / "reference_dmrg_runtime"
    scratch_root.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(tempfile.mkdtemp(prefix=f"{fcidump_path.stem}_dmrg_", dir=str(scratch_root)))
    log_path = runtime_dir / "dmrg.log"
    stdout_path = runtime_dir / "subprocess.stdout"
    stderr_path = runtime_dir / "subprocess.stderr"
    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data["MS2"])
    ecore = float(data.get("ECORE", 0.0))
    orb_sym = [int(x) for x in data.get("ORBSYM", [1] * norb)]
    isym = _normalized_c1_isym(data.get("ISYM", 0))
    hf_occs = hf_occ_vector(norb, nelec)

    requested_bond_dims = (
        [int(x) for x in bond_dims]
        if bond_dims is not None
        else [int(x) for x in DEFAULT_DMRG_BOND_DIMS]
    )
    if not requested_bond_dims:
        raise ValueError("DMRG reference requires at least one bond dimension")
    requested_noises = (
        [float(x) for x in noises]
        if noises is not None
        else [float(x) for x in _fit_schedule_length(DEFAULT_DMRG_NOISES, len(requested_bond_dims))]
    )
    requested_dav_thrds = (
        [float(x) for x in dav_thrds]
        if dav_thrds is not None
        else [float(x) for x in _fit_schedule_length(DEFAULT_DMRG_DAV_THRDS, len(requested_bond_dims))]
    )
    if len(requested_noises) != len(requested_bond_dims):
        raise ValueError("DMRG noise schedule must match the bond-dimension schedule length")
    if len(requested_dav_thrds) != len(requested_bond_dims):
        raise ValueError("DMRG Davidson-threshold schedule must match the bond-dimension schedule length")
    preflight_bond_dims = [int(x) for x in DEFAULT_DMRG_PREFLIGHT_BOND_DIMS]
    preflight_noises = [float(x) for x in DEFAULT_DMRG_PREFLIGHT_NOISES]
    preflight_dav_thrds = [float(x) for x in DEFAULT_DMRG_PREFLIGHT_DAV_THRDS]
    script = _dmrg_reference_script()
    env = os.environ.copy()
    env.update(
        {
            "CKA_FCIDUMP": str(fcidump_path),
            "CKA_SCRATCH": str(scratch_dir),
            "CKA_LOG": str(log_path),
            "CKA_MEMORY_GB": str(int(memory_gb)),
            "CKA_THREADS": str(int(threads)),
            "CKA_CONV_TOL": str(float(conv_tol)),
            "CKA_BOND_DIMS": json.dumps(requested_bond_dims),
            "CKA_NOISES": json.dumps(requested_noises),
            "CKA_DAV_THRDS": json.dumps(requested_dav_thrds),
            "CKA_N_STEPS": str(int(n_steps)),
            "CKA_OCCS_JSON": json.dumps(hf_occs),
            "CKA_ISYM": str(isym),
            "CKA_ORBSYM_JSON": json.dumps(orb_sym),
            "CKA_PREFLIGHT_BOND_DIMS": json.dumps(preflight_bond_dims),
            "CKA_PREFLIGHT_NOISES": json.dumps(preflight_noises),
            "CKA_PREFLIGHT_DAV_THRDS": json.dumps(preflight_dav_thrds),
            "CKA_PREFLIGHT_N_STEPS": "2",
        }
    )
    # The short preflight inside `_dmrg_reference_script()` is meant to fail
    # fast on broken symmetry/initialization settings before we start the full
    # bond-dimension ladder.
    capped = run_capped_command(
        cmd=[str(dmrg_python), "-c", script],
        cwd=None,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        watch_roots=[runtime_dir, scratch_dir],
        max_file_mb=int(max_file_mb),
        max_total_scratch_gb=float(max_total_scratch_gb),
    )
    try:
        if capped["stop_reason"] is not None:
            payload = {
                "success": False,
                "reference_kind": "dmrg",
                "reference_label": "DMRG",
                "failure": (
                    f"DMRG scratch guard triggered: {capped['stop_reason']}"
                ),
                "stop_reason": capped["stop_reason"],
                "largest_file_path": capped["largest_file_path"],
                "largest_file_mb": capped["largest_file_mb"],
                "peak_total_scratch_gb": capped["peak_total_scratch_gb"],
                "max_file_mb": int(max_file_mb),
                "max_total_scratch_gb": float(max_total_scratch_gb),
                "log_path": str(log_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return payload
        if int(capped["returncode"]) != 0:
            payload: dict[str, object]
            stdout_text = stdout_path.read_text(encoding="utf-8").strip() if stdout_path.exists() else ""
            try:
                parsed = json.loads(stdout_text) if stdout_text else None
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                payload = normalize_reference_summary(parsed)
                payload.update(
                    {
                        "success": False,
                        "largest_file_path": capped["largest_file_path"],
                        "largest_file_mb": capped["largest_file_mb"],
                        "peak_total_scratch_gb": capped["peak_total_scratch_gb"],
                        "max_file_mb": int(max_file_mb),
                        "max_total_scratch_gb": float(max_total_scratch_gb),
                        "log_path": str(log_path),
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                    }
                )
                payload.setdefault(
                    "failure",
                    "DMRG reference subprocess failed; inspect structured preflight payload",
                )
                payload.setdefault("stop_reason", "subprocess_failed")
            else:
                payload = {
                    "success": False,
                    "reference_kind": "dmrg",
                    "reference_label": "DMRG",
                    "failure": (
                        "DMRG reference subprocess failed with rc="
                        f"{int(capped['returncode'])}; see {log_path}, {stdout_path}, and {stderr_path}"
                    ),
                    "stop_reason": "subprocess_failed",
                    "largest_file_path": capped["largest_file_path"],
                    "largest_file_mb": capped["largest_file_mb"],
                    "peak_total_scratch_gb": capped["peak_total_scratch_gb"],
                    "max_file_mb": int(max_file_mb),
                    "max_total_scratch_gb": float(max_total_scratch_gb),
                    "log_path": str(log_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "init_api": "explicit_hamiltonian",
                    "isym": isym,
                    "ms2": ms2,
                    "norb": norb,
                    "nelec": nelec,
                    "orb_sym": orb_sym,
                }
            attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return payload
        payload = normalize_reference_summary(json.loads(stdout_path.read_text(encoding="utf-8").strip()))
        payload.update(
            {
                "success": True,
                "largest_file_path": capped["largest_file_path"],
                "largest_file_mb": capped["largest_file_mb"],
                "peak_total_scratch_gb": capped["peak_total_scratch_gb"],
                "max_file_mb": int(max_file_mb),
                "max_total_scratch_gb": float(max_total_scratch_gb),
                "scratch_root": str(scratch_dir),
            }
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def load_or_compute_pyscf_selected_ci_reference_from_fcidump(
    *,
    fcidump_path: Path,
    outdir: Path,
    force: bool,
    memory_gb: int,
    conv_tol: float,
    cutoff_ladder: tuple[float, ...] = DEFAULT_PYSCF_CIPSI_CUTOFFS,
    max_cycle: int = 40,
) -> dict[str, object]:
    from pyscf.fci import selected_ci

    summary_path = outdir / "reference_cipsi_summary.json"
    attempt_path = outdir / "reference_cipsi_attempt.json"
    runtime_dir = outdir / "reference_cipsi_runtime"
    if summary_path.exists() and not force:
        payload = normalize_reference_summary(json.loads(summary_path.read_text(encoding="utf-8")))
        payload["success"] = True
        return payload

    runtime_dir.mkdir(parents=True, exist_ok=True)
    data = pyscf_fcidump.read(str(fcidump_path), verbose=False)
    h1 = data["H1"]
    h2 = data["H2"]
    norb = int(data["NORB"])
    nelec_tot = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    na = (nelec_tot + ms2) // 2
    nb = nelec_tot - na
    nelec = (int(na), int(nb))
    ecore = float(data.get("ECORE", 0.0))

    rung_rows: list[dict[str, object]] = []
    ci0 = None
    t0 = time.perf_counter()
    try:
        for cutoff in cutoff_ladder:
            myci = selected_ci.SelectedCI()
            myci.select_cutoff = float(cutoff)
            myci.ci_coeff_cutoff = float(cutoff)
            myci.max_memory = int(memory_gb) * 1024
            myci.max_cycle = int(max_cycle)
            myci.conv_tol = float(conv_tol)
            rung_t0 = time.perf_counter()
            total_energy, ci0 = myci.kernel(h1, h2, norb, nelec, ci0=ci0, ecore=ecore)
            ndet = int(len(ci0._strs[0]) * len(ci0._strs[1]))
            rung_rows.append(
                {
                    "cutoff": float(cutoff),
                    "total_energy_eh": float(total_energy),
                    "electronic_energy_eh": float(total_energy - ecore),
                    "selected_n_det": ndet,
                    "wall_s": float(time.perf_counter() - rung_t0),
                }
            )
    except Exception as exc:
        payload = {
            "success": False,
            "reference_kind": "cipsi",
            "reference_label": "CIPSI variational",
            "reference_backend": "pyscf_selected_ci",
            "failure": f"{type(exc).__name__}: {exc}",
            "stop_reason": "exception",
            "memory_gb": int(memory_gb),
            "conv_tol": float(conv_tol),
            "cutoff_ladder": [float(x) for x in cutoff_ladder],
            "completed_rungs": rung_rows,
        }
        attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    if not rung_rows:
        payload = {
            "success": False,
            "reference_kind": "cipsi",
            "reference_label": "CIPSI variational",
            "reference_backend": "pyscf_selected_ci",
            "failure": "No selected-CI rung completed",
            "stop_reason": "no_completed_rungs",
            "memory_gb": int(memory_gb),
            "conv_tol": float(conv_tol),
            "cutoff_ladder": [float(x) for x in cutoff_ladder],
            "completed_rungs": [],
        }
        attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    chosen = rung_rows[-1]
    payload = {
        "success": True,
        "reference_kind": "cipsi",
        "reference_label": "CIPSI variational",
        "reference_backend": "pyscf_selected_ci",
        "total_energy_eh": float(chosen["total_energy_eh"]),
        "electronic_energy_eh": float(chosen["electronic_energy_eh"]),
        "selected_n_det": int(chosen["selected_n_det"]),
        "selected_cutoff": float(chosen["cutoff"]),
        "completed_rungs": rung_rows,
        "memory_gb": int(memory_gb),
        "conv_tol": float(conv_tol),
        "cutoff_ladder": [float(x) for x in cutoff_ladder],
        "wall_s": float(time.perf_counter() - t0),
        "fcidump_path": str(fcidump_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _run_cipsi_setup_command(
    *,
    qp_root: Path,
    runtime_ezfio: Path,
    n_det_max: int,
    read_wf: bool,
) -> None:
    qp = discover_qp_paths_from_root(qp_root)
    cmd = (
        f"source '{qp.quantum_package_rc}' >/dev/null 2>&1 && "
        f"qp set_file '{runtime_ezfio}' >/dev/null && "
        f"qp set determinants n_det_max {int(n_det_max)} && "
        f"qp set determinants read_wf {'true' if read_wf else 'false'}"
    )
    subprocess.run(
        ["/bin/bash", "-lc", cmd],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
    )


def load_or_compute_cipsi_reference_from_ezfio(
    *,
    label: str,
    qp_root: Path,
    base_ezfio: Path,
    outdir: Path,
    force: bool,
    omp_threads: int,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
    max_total_scratch_gb: float = DEFAULT_TOTAL_SCRATCH_GB,
    scratch_root: Path = DEFAULT_CIPSI_SCRATCH_ROOT,
    cipsi_rungs: tuple[int, ...] = DEFAULT_CIPSI_RUNGS,
) -> dict[str, object]:
    summary_path = outdir / "reference_cipsi_summary.json"
    attempt_path = outdir / "reference_cipsi_attempt.json"
    runtime_dir = outdir / "reference_cipsi_runtime"
    if summary_path.exists() and not force:
        payload = normalize_reference_summary(json.loads(summary_path.read_text(encoding="utf-8")))
        payload["success"] = True
        return payload

    qp = discover_qp_paths_from_root(qp_root)
    fci_binary = qp.qp_root / "src" / "fci" / "fci"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix=f"{label}_cipsi_", dir=str(scratch_root)))
    completed_rungs: list[dict[str, object]] = []
    previous_seed: Path | None = None
    largest_file_path: str | None = None
    largest_file_mb = 0.0
    peak_total_scratch_gb = 0.0

    try:
        for n_det_max in cipsi_rungs:
            rung_dir = work_root / f"cipsi_{int(n_det_max)}"
            runtime_ezfio = rung_dir / f"{label}.ezfio"
            if rung_dir.exists():
                shutil.rmtree(rung_dir)
            rung_dir.mkdir(parents=True, exist_ok=True)
            source = base_ezfio if previous_seed is None else previous_seed
            shutil.copytree(source, runtime_ezfio)
            ensure_ezfio_defaults(runtime_ezfio, qp_root=qp.qp_root)
            _run_cipsi_setup_command(
                qp_root=qp_root,
                runtime_ezfio=runtime_ezfio,
                n_det_max=int(n_det_max),
                read_wf=previous_seed is not None,
            )
            stdout_path = runtime_dir / f"cipsi_{int(n_det_max)}.log"
            stderr_path = runtime_dir / f"cipsi_{int(n_det_max)}.stderr"
            local_scratch = Path(tempfile.mkdtemp(prefix=f"cipsi_{int(n_det_max)}_", dir=str(work_root)))
            capped = run_capped_command(
                cmd=[
                    "/bin/bash",
                    "-lc",
                    (
                        f"source '{qp.quantum_package_rc}' >/dev/null 2>&1 && "
                        f"'{fci_binary}' '{runtime_ezfio}'"
                    ),
                ],
                cwd=None,
                env={**os.environ, "QP_ROOT": str(qp.qp_root), "TMPDIR": str(local_scratch), **qp_parallel_env(int(omp_threads))},
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                watch_roots=[runtime_dir, work_root],
                max_file_mb=int(max_file_mb),
                max_total_scratch_gb=float(max_total_scratch_gb),
            )
            largest_file_mb = max(largest_file_mb, float(capped["largest_file_mb"]))
            peak_total_scratch_gb = max(peak_total_scratch_gb, float(capped["peak_total_scratch_gb"]))
            if capped["largest_file_path"] is not None and float(capped["largest_file_mb"]) >= largest_file_mb:
                largest_file_path = str(capped["largest_file_path"])
            shutil.rmtree(local_scratch, ignore_errors=True)
            if capped["stop_reason"] is not None or int(capped["returncode"]) != 0:
                stop_text = (
                    str(capped["stop_reason"])
                    if capped["stop_reason"] is not None
                    else f"returncode={int(capped['returncode'])}"
                )
                payload = {
                    "success": False,
                    "reference_kind": "cipsi",
                    "reference_label": "CIPSI variational",
                    "failure": (
                        f"CIPSI failed at n_det_max={int(n_det_max)} with "
                        f"{stop_text}"
                    ),
                    "stop_reason": capped["stop_reason"] or "subprocess_failed",
                    "largest_file_path": largest_file_path,
                    "largest_file_mb": largest_file_mb,
                    "peak_total_scratch_gb": peak_total_scratch_gb,
                    "max_file_mb": int(max_file_mb),
                    "max_total_scratch_gb": float(max_total_scratch_gb),
                    "completed_rungs": completed_rungs,
                }
                attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return payload
            parsed = parse_cipsi_summaries(stdout_path.read_text(errors="ignore"))
            if not parsed:
                payload = {
                    "success": False,
                    "reference_kind": "cipsi",
                    "reference_label": "CIPSI variational",
                    "failure": f"No CIPSI summaries found at n_det_max={int(n_det_max)}",
                    "stop_reason": "missing_cipsi_summary",
                    "largest_file_path": largest_file_path,
                    "largest_file_mb": largest_file_mb,
                    "peak_total_scratch_gb": peak_total_scratch_gb,
                    "max_file_mb": int(max_file_mb),
                    "max_total_scratch_gb": float(max_total_scratch_gb),
                    "completed_rungs": completed_rungs,
                }
                attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return payload
            best = parsed[-1]
            completed_rungs.append(
                {
                    "n_det": int(best["n_det"]),
                    "e_var": float(best["e_var"]),
                    "e_pt2": float(best["e_pt2"]),
                    "log_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "largest_file_mb": float(capped["largest_file_mb"]),
                    "peak_total_scratch_gb": float(capped["peak_total_scratch_gb"]),
                    "largest_file_path": capped["largest_file_path"],
                    "requested_n_det_max": int(n_det_max),
                    "seed_kind": "previous_rung" if previous_seed is not None else "fresh_setup",
                }
            )
            previous_seed = runtime_ezfio

        if not completed_rungs:
            payload = {
                "success": False,
                "reference_kind": "cipsi",
                "reference_label": "CIPSI variational",
                "failure": "No bounded CIPSI rung completed successfully",
                "stop_reason": "no_completed_rungs",
                "largest_file_path": largest_file_path,
                "largest_file_mb": largest_file_mb,
                "peak_total_scratch_gb": peak_total_scratch_gb,
                "max_file_mb": int(max_file_mb),
                "max_total_scratch_gb": float(max_total_scratch_gb),
                "completed_rungs": [],
            }
            attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return payload
        chosen = completed_rungs[-1]
        payload = {
            "success": True,
            "reference_kind": "cipsi",
            "reference_label": "CIPSI variational",
            "total_energy_eh": float(chosen["e_var"]),
            "cipsi_pt2_total_energy_eh": float(chosen["e_pt2"]),
            "selected_n_det": int(chosen["n_det"]),
            "selected_requested_n_det_max": int(chosen["requested_n_det_max"]),
            "completed_rungs": completed_rungs,
            "largest_file_path": largest_file_path,
            "largest_file_mb": largest_file_mb,
            "peak_total_scratch_gb": peak_total_scratch_gb,
            "max_file_mb": int(max_file_mb),
            "max_total_scratch_gb": float(max_total_scratch_gb),
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        attempt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
