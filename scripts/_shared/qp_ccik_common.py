"""Shared helpers for Quantum Package CCIK driver scripts.

These helpers keep the QP setup/run logic in one place so the N2 and Cr2
single-point/scan drivers do not drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EZFIO_DONOR = ROOT / "docs" / "assets" / "local_qp_seeds" / "n2_cas14_14_R2p50" / "n2.ezfio"
DEFAULT_SMOKE_EZFIO_DONOR = ROOT / "docs" / "assets" / "local_qp_seeds" / "n2_cas12_12_R2p50" / "n2.ezfio"
WORKSPACE_CCIK_PLUGIN = ROOT / "ccik_qp" / "stable" / "ccik"
DEFAULT_QP_REBUILD_MANIFEST = (
    ROOT / "experiments" / "threshold_test" / "threshold_mode_validation_20260325" / "qp_plugin_rebuild.json"
)
DEFAULT_QP_FORBIDDEN_PREFIXES = (Path("/private/tmp/qp2_fresh_20260320"),)
DEFAULT_MACOS_DIAGNOSTIC_REPORTS_DIR = Path.home() / "Library" / "Logs" / "DiagnosticReports"


@dataclass(frozen=True)
class QPPaths:
    qp_root: Path
    quantum_package_rc: Path
    qp_set_mo_class: Path
    qp_create_ezfio: str
    scf_binary: Path
    ccik_binary: Path
    pt2_binary: Path
    orb_opt_binary: Path


@dataclass(frozen=True)
class PreparedActiveSpace:
    mo_num: int
    ncore: int
    ncas: int
    neleca: int
    nelecb: int
    nelecas: int
    source: str


@dataclass(frozen=True)
class CCIKParams:
    ccik_basis_size: int = 32
    ccik_generator_keep: int = 256
    ccik_generator_fraction: float = 0.0
    ccik_generator_abs_coeff_cutoff: float = 0.0
    ccik_max_parents_for_expansion: int = 0
    ccik_include_current_support: bool = True
    ccik_selector_keep: int = 2048
    ccik_external_keep: int = 2048
    ccik_external_candidate_factor: int = 4
    ccik_pt2_rank_weight: float = 1.0
    ccik_selector_mode: str = "abs_residual"
    ccik_matvec_backend: str = "connected_sd_fullq"
    ccik_compression_policy: str = "fixed_budget"
    ccik_fixed_store_keep: int = 65_536
    ccik_fixed_propagate_keep: int = 512
    ccik_store_abs_tol: float = 0.0
    ccik_propagate_abs_tol: float = 0.0
    ccik_nkeep: int = 2048
    ccik_tmp_keep_factor: int = 4
    ccik_preserve_prev_support: bool = True
    ccik_preserve_prev_support_cap: int = 0
    ccik_total_det_budget: int = 0
    ccik_conv_tol: float = 1e-6
    ccik_ritz_overlap_tol: float = 1e-8
    ccik_ritz_max_drop: float = 0.0
    ccik_save_final_wf: bool = True
    ccik_use_existing_wf: bool = False
    ccik_do_post_pt2: bool = False
    ccik_max_cycle: int = 1
    ccik_restart_m_cycle: int = 0
    ccik_restart_nroot: int = 2
    ccik_verbose: bool = False


def bash(cmd: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-lc", cmd],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def discover_qp_paths_from_root(root: Path) -> QPPaths:
    root = Path(root).resolve()
    qp_create_ezfio_path = root / "ocaml" / "qp_create_ezfio"
    return QPPaths(
        qp_root=root,
        quantum_package_rc=root / "quantum_package.rc",
        qp_set_mo_class=root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio=str(qp_create_ezfio_path) if qp_create_ezfio_path.exists() else "qp_create_ezfio",
        scf_binary=root / "src" / "hartree_fock" / "scf",
        ccik_binary=root / "plugins" / "ccik_qp" / "stable" / "ccik" / "ccik",
        pt2_binary=root / "src" / "fci" / "pt2",
        orb_opt_binary=root / "src" / "mo_optimization" / "orb_opt",
    )


def discover_qp_paths() -> QPPaths:
    qp_root = os.environ.get("QP_ROOT")
    if not qp_root:
        raise RuntimeError("QP_ROOT is not set. Source quantum_package.rc first.")
    return discover_qp_paths_from_root(Path(qp_root))


def validate_rebuildable_qp_root(root: Path) -> dict[str, object]:
    candidate = Path(root).resolve()
    plugin_dir = candidate / "plugins" / "ccik_qp" / "stable" / "ccik"
    payload = {
        "qp_root": str(candidate),
        "root_exists": candidate.exists(),
        "quantum_package_rc": str(candidate / "quantum_package.rc"),
        "quantum_package_rc_exists": (candidate / "quantum_package.rc").exists(),
        "build_ninja": str(candidate / "config" / "build.ninja"),
        "build_ninja_exists": (candidate / "config" / "build.ninja").exists(),
        "plugin_dir": str(plugin_dir),
        "plugin_dir_exists": plugin_dir.exists(),
        "is_rebuildable": False,
    }
    payload["is_rebuildable"] = bool(
        payload["root_exists"]
        and payload["quantum_package_rc_exists"]
        and payload["build_ninja_exists"]
        and payload["plugin_dir_exists"]
    )
    return payload


def _allowed_nonroot_dependency_prefixes() -> tuple[str, ...]:
    return (
        "/System/Library/",
        "/usr/lib/",
        "/opt/homebrew/",
        "/usr/local/",
        "/Library/Developer/CommandLineTools/",
        "@rpath/",
        "@loader_path/",
        "@executable_path/",
    )


def _is_allowed_dependency_path(dep: str, qp_root: Path, *, allowed_nonroot_prefixes: tuple[str, ...]) -> bool:
    root_text = str(qp_root.resolve())
    if dep.startswith(root_text):
        return True
    return any(dep.startswith(prefix) for prefix in allowed_nonroot_prefixes)


def collect_qp_macho_validation(
    qp: QPPaths | Path,
    *,
    forbidden_prefixes: tuple[Path | str, ...] = DEFAULT_QP_FORBIDDEN_PREFIXES,
) -> dict[str, object]:
    qp_paths = qp if isinstance(qp, QPPaths) else discover_qp_paths_from_root(Path(qp))
    qp_root = qp_paths.qp_root.resolve()
    lib_dir = (qp_root / "lib").resolve()
    dylibs = sorted(path for path in lib_dir.glob("*.dylib") if path.is_file())
    allowed_nonroot_prefixes = _allowed_nonroot_dependency_prefixes()
    forbidden_texts = tuple(str(Path(prefix).resolve()) for prefix in forbidden_prefixes)

    dylib_rows: list[dict[str, object]] = []
    dylib_issues: list[str] = []
    for dylib in dylibs:
        install_id = _mach_o_id(dylib)
        issues: list[str] = []
        if install_id is None:
            issues.append("missing_install_id")
        elif not install_id.startswith(str(lib_dir)):
            issues.append(f"install_id_outside_root:{install_id}")
        for prefix in forbidden_texts:
            if install_id is not None and install_id.startswith(prefix):
                issues.append(f"install_id_forbidden_prefix:{prefix}")
        if issues:
            dylib_issues.append(f"{dylib.name}: " + ", ".join(issues))
        dylib_rows.append(
            {
                "path": str(dylib),
                "install_id": install_id,
                "issues": issues,
            }
        )

    runtime_targets = {
        "ccik": qp_paths.ccik_binary.resolve(),
        "fcidump": (qp_root / "src" / "tools" / "fcidump").resolve(),
    }
    target_rows: list[dict[str, object]] = []
    target_issues: list[str] = []
    for label, target in runtime_targets.items():
        deps = _mach_o_deps(target)
        issues: list[str] = []
        for dep in deps:
            if not _is_allowed_dependency_path(dep, qp_root, allowed_nonroot_prefixes=allowed_nonroot_prefixes):
                issues.append(f"unexpected_dependency:{dep}")
            for prefix in forbidden_texts:
                if dep.startswith(prefix):
                    issues.append(f"forbidden_dependency:{dep}")
        if issues:
            target_issues.append(f"{label}: " + ", ".join(issues))
        target_rows.append(
            {
                "label": label,
                "path": str(target),
                "dependencies": deps,
                "issues": issues,
            }
        )

    all_issues = dylib_issues + target_issues
    return {
        "qp_root": str(qp_root),
        "lib_dir": str(lib_dir),
        "forbidden_prefixes": list(forbidden_texts),
        "dylibs": dylib_rows,
        "runtime_targets": target_rows,
        "issues": all_issues,
        "passed": not all_issues,
    }


def _snapshot_diagnostic_reports(
    reports_dir: Path = DEFAULT_MACOS_DIAGNOSTIC_REPORTS_DIR,
    *,
    binary_names: tuple[str, ...] = ("ccik", "fcidump"),
) -> dict[str, float]:
    if not reports_dir.exists():
        return {}
    snapshot: dict[str, float] = {}
    for report in reports_dir.glob("*.ips"):
        for binary in binary_names:
            if report.name.startswith(f"{binary}-") or report.name.startswith(f"{binary}."):
                try:
                    snapshot[str(report.resolve())] = report.stat().st_mtime
                except FileNotFoundError:
                    continue
                break
    return snapshot


def _new_diagnostic_reports(before: dict[str, float], after: dict[str, float]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for path, mtime in sorted(after.items()):
        if before.get(path) == mtime:
            continue
        report = Path(path)
        try:
            text = report.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        payload.append(
            {
                "path": path,
                "mtime": mtime,
                "contains_invalid_page": "Invalid Page" in text,
                "contains_codesigning": "CODESIGNING" in text,
            }
        )
    return payload


def _copy_ezfio_tree(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _smoke_ccik_params() -> CCIKParams:
    return CCIKParams(
        ccik_basis_size=2,
        ccik_generator_keep=32,
        ccik_selector_keep=64,
        ccik_external_keep=64,
        ccik_external_candidate_factor=2,
        ccik_nkeep=64,
        ccik_tmp_keep_factor=2,
        ccik_preserve_prev_support=False,
        ccik_total_det_budget=0,
        ccik_conv_tol=1e-6,
        ccik_ritz_max_drop=0.0,
        ccik_save_final_wf=False,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=False,
        ccik_max_cycle=1,
        ccik_restart_m_cycle=0,
        ccik_restart_nroot=2,
        ccik_verbose=False,
    )


def _run_qp_runtime_smoke(
    qp: QPPaths,
    *,
    outdir: Path,
    donor_ezfio: Path | None = None,
    timeout_s: int = 180,
) -> dict[str, object]:
    donor = DEFAULT_SMOKE_EZFIO_DONOR if donor_ezfio is None else Path(donor_ezfio)
    if not donor.exists():
        raise FileNotFoundError(f"Smoke donor EZFIO not found: {donor}")

    reports_before = _snapshot_diagnostic_reports()
    workdir = outdir / "runtime_smoke"
    workdir.mkdir(parents=True, exist_ok=True)
    smoke_ezfio = _copy_ezfio_tree(donor, workdir / "smoke.ezfio")
    ensure_ezfio_defaults(smoke_ezfio, qp_root=qp.qp_root)

    env = {**os.environ, "QP_ROOT": str(qp.qp_root), **qp_parallel_env(1)}
    quoted_rc = shlex.quote(str(qp.quantum_package_rc))
    short_ezfio = shlex.quote(smoke_ezfio.name)
    fcidump_binary = shlex.quote(str((qp.qp_root / "src" / "tools" / "fcidump").resolve()))
    fcidump_stdout = workdir / "fcidump_stdout.log"
    fcidump_stderr = workdir / "fcidump_stderr.log"
    run_fcidump = subprocess.run(
        ["/bin/bash", "-lc", f"source {quoted_rc} >/dev/null 2>&1 && {fcidump_binary} {short_ezfio}"],
        cwd=str(workdir),
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    fcidump_stdout.write_text(run_fcidump.stdout, encoding="utf-8")
    fcidump_stderr.write_text(run_fcidump.stderr, encoding="utf-8")

    write_ezfio_value(smoke_ezfio, "ccik/ccik_debug_trace", True)
    write_ezfio_value(smoke_ezfio, "ccik/ccik_debug_trace_topn", 8)
    set_ccik_params(qp, smoke_ezfio, _smoke_ccik_params())

    ccik_stdout = workdir / "ccik_stdout.log"
    ccik_stderr = workdir / "ccik_stderr.log"
    quoted_ccik = shlex.quote(str(qp.ccik_binary.resolve()))
    run_ccik_binary = subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"source {quoted_rc} >/dev/null 2>&1 && qp set_file {short_ezfio} >/dev/null && {quoted_ccik} {short_ezfio}",
        ],
        cwd=str(workdir),
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    ccik_stdout.write_text(run_ccik_binary.stdout, encoding="utf-8")
    ccik_stderr.write_text(run_ccik_binary.stderr, encoding="utf-8")

    trace_path = workdir / "ccik_debug_trace.log"
    reports_after = _snapshot_diagnostic_reports()
    new_reports = _new_diagnostic_reports(reports_before, reports_after)
    passed = (
        run_fcidump.returncode == 0
        and run_ccik_binary.returncode == 0
        and trace_path.exists()
        and not new_reports
    )
    return {
        "donor_ezfio": str(donor.resolve()),
        "workdir": str(workdir),
        "fcidump": {
            "returncode": int(run_fcidump.returncode),
            "stdout_path": str(fcidump_stdout),
            "stderr_path": str(fcidump_stderr),
            "output_path": str(workdir / "FCIDUMP"),
            "output_exists": (workdir / "FCIDUMP").exists(),
        },
        "ccik": {
            "returncode": int(run_ccik_binary.returncode),
            "stdout_path": str(ccik_stdout),
            "stderr_path": str(ccik_stderr),
            "trace_path": str(trace_path),
            "trace_exists": trace_path.exists(),
        },
        "new_diagnostic_reports": new_reports,
        "passed": passed,
    }


def validate_fresh_qp_root(
    root: Path,
    *,
    outdir: Path | None = None,
    donor_ezfio: Path | None = None,
    forbidden_prefixes: tuple[Path | str, ...] = DEFAULT_QP_FORBIDDEN_PREFIXES,
    smoke_timeout_s: int = 180,
) -> dict[str, object]:
    structural = validate_rebuildable_qp_root(root)
    payload: dict[str, object] = {
        "qp_root": str(Path(root).resolve()),
        "structural": structural,
        "macho": None,
        "runtime_smoke": None,
        "passed": False,
    }
    if not structural["is_rebuildable"]:
        return payload

    qp = discover_qp_paths_from_root(Path(root))
    macho = collect_qp_macho_validation(qp, forbidden_prefixes=forbidden_prefixes)
    payload["macho"] = macho
    if not macho["passed"]:
        return payload

    smoke_outdir = Path(tempfile.mkdtemp(prefix="qp_root_validation_")) if outdir is None else Path(outdir)
    smoke = _run_qp_runtime_smoke(qp, outdir=smoke_outdir, donor_ezfio=donor_ezfio, timeout_s=smoke_timeout_s)
    payload["runtime_smoke"] = smoke
    payload["passed"] = bool(structural["is_rebuildable"] and macho["passed"] and smoke["passed"])
    return payload


def _infer_qp_root_from_plugin_path(path: Path) -> Path | None:
    target = Path(path).resolve()
    parts = list(target.parts)
    try:
        idx = parts.index("plugins")
    except ValueError:
        return None
    if idx <= 0:
        return None
    return Path(*parts[:idx])


def infer_qp_root_from_rebuild_manifest(manifest_path: Path | None = None) -> Path | None:
    manifest = DEFAULT_QP_REBUILD_MANIFEST if manifest_path is None else Path(manifest_path)
    if not manifest.exists():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("plugin_dir", "binary_path"):
        value = payload.get(key)
        if not value:
            continue
        inferred = _infer_qp_root_from_plugin_path(Path(value))
        if inferred is not None:
            return inferred.resolve()
    sync_summary = payload.get("sync_summary")
    if isinstance(sync_summary, dict):
        value = sync_summary.get("installed_plugin_dir")
        if value:
            inferred = _infer_qp_root_from_plugin_path(Path(value))
            if inferred is not None:
                return inferred.resolve()
    return None


def resolve_rebuildable_qp_root(
    explicit: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    rebuild_manifest: Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_map = os.environ if env is None else env
    env_root = env_map.get("QP_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    manifest_root = infer_qp_root_from_rebuild_manifest(rebuild_manifest)
    if manifest_root is not None:
        candidates.append(manifest_root)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if validate_rebuildable_qp_root(resolved)["is_rebuildable"]:
            return resolved
    return None


def detect_physical_cpu_count() -> int:
    try:
        run = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            check=True,
            text=True,
            capture_output=True,
        )
        count = int(run.stdout.strip())
        if count > 0:
            return count
    except Exception:
        pass
    return max(1, int(os.cpu_count() or 1))


def resolve_omp_threads(value: str | int | None) -> int:
    if value is None:
        return detect_physical_cpu_count()
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "auto"}:
            return detect_physical_cpu_count()
        threads = int(stripped)
    else:
        threads = int(value)
    if threads <= 0:
        raise ValueError(f"OMP thread count must be positive, got {threads}")
    return threads


def qp_parallel_env(threads: str | int | None = None) -> dict[str, str]:
    resolved = resolve_omp_threads(threads)
    return {
        "OMP_NUM_THREADS": str(resolved),
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def resolve_workspace_ccik_plugin_dir(path: Path | None = None) -> Path:
    plugin_dir = (WORKSPACE_CCIK_PLUGIN if path is None else Path(path)).resolve()
    if not plugin_dir.exists():
        raise RuntimeError(f"Workspace CCIK plugin directory not found: {plugin_dir}")
    return plugin_dir


def resolve_installed_ccik_plugin_dir(qp: QPPaths | Path, *, create: bool = False) -> Path:
    qp_root = qp.qp_root if isinstance(qp, QPPaths) else Path(qp)
    plugin_dir = (Path(qp_root) / "plugins" / "ccik_qp" / "stable" / "ccik").resolve()
    if create:
        plugin_dir.mkdir(parents=True, exist_ok=True)
    if not plugin_dir.exists():
        raise RuntimeError(f"Installed QP CCIK plugin directory not found: {plugin_dir}")
    return plugin_dir


def resolve_qp_build_ninja(qp: QPPaths | Path) -> Path:
    qp_root = qp.qp_root if isinstance(qp, QPPaths) else Path(qp)
    build_ninja = (Path(qp_root) / "config" / "build.ninja").resolve()
    if not build_ninja.exists():
        raise RuntimeError(f"Installed QP build.ninja not found: {build_ninja}")
    return build_ninja


def resolve_plugin_build_ninja(qp: QPPaths | Path) -> Path:
    plugin_dir = resolve_installed_ccik_plugin_dir(qp)
    candidates = (
        (plugin_dir / "IRPF90_temp" / "build.ninja").resolve(),
        (plugin_dir / "build.ninja").resolve(),
    )
    for build_ninja in candidates:
        if build_ninja.exists():
            return build_ninja
    raise RuntimeError(
        "Installed plugin build.ninja not found in either the plugin root or IRPF90_temp.\n"
        f"plugin_dir={plugin_dir}"
    )


def _mach_o_id(path: Path) -> str | None:
    run = subprocess.run(
        ["otool", "-D", str(path)],
        check=False,
        text=True,
        capture_output=True,
    )
    if run.returncode != 0:
        return None
    lines = [line.strip() for line in run.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    return lines[1]


def _mach_o_deps(path: Path) -> list[str]:
    run = subprocess.run(
        ["otool", "-L", str(path)],
        check=False,
        text=True,
        capture_output=True,
    )
    if run.returncode != 0:
        return []
    deps: list[str] = []
    for line in run.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        dep = stripped.split(" (compatibility version", 1)[0].strip()
        deps.append(dep)
    return deps


def _candidate_runtime_targets(qp: QPPaths) -> list[Path]:
    targets = [
        qp.ccik_binary,
        qp.pt2_binary,
        qp.orb_opt_binary,
        qp.qp_root / "src" / "tools" / "fcidump",
        qp.scf_binary,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for target in targets:
        resolved = Path(target).resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def repair_qp_runtime_install_names(qp: QPPaths) -> dict[str, object]:
    lib_dir = (qp.qp_root / "lib").resolve()
    if not lib_dir.exists():
        raise RuntimeError(f"QP lib directory not found: {lib_dir}")

    canonical_libs: dict[str, Path] = {}
    for candidate in sorted(lib_dir.iterdir()):
        if ".dylib" not in candidate.name:
            continue
        if not candidate.exists():
            continue
        canonical_libs[candidate.name] = candidate.resolve()

    id_repairs: list[dict[str, str]] = []
    dep_repairs: list[dict[str, object]] = []

    for lib_path in sorted(set(canonical_libs.values())):
        current_id = _mach_o_id(lib_path)
        desired_id = str(lib_path)
        if current_id and current_id != desired_id:
            run = subprocess.run(
                ["install_name_tool", "-id", desired_id, str(lib_path)],
                check=False,
                text=True,
                capture_output=True,
            )
            if run.returncode != 0:
                raise RuntimeError(
                    "Failed to repair QP dylib install name.\n"
                    f"library={lib_path}\n"
                    f"stdout:\n{run.stdout}\n"
                    f"stderr:\n{run.stderr}"
                )
            id_repairs.append({"library": str(lib_path), "old_id": current_id, "new_id": desired_id})

    patch_targets = _candidate_runtime_targets(qp) + sorted(set(canonical_libs.values()))
    for target in patch_targets:
        changes: list[tuple[str, str]] = []
        for dep in _mach_o_deps(target):
            dep_path = Path(dep)
            mapped = canonical_libs.get(dep_path.name)
            if mapped is None:
                continue
            desired_dep = str(mapped)
            if dep != desired_dep:
                changes.append((dep, desired_dep))
        if not changes:
            continue
        cmd: list[str] = ["install_name_tool"]
        for old_dep, new_dep in changes:
            cmd.extend(["-change", old_dep, new_dep])
        cmd.append(str(target))
        run = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
        if run.returncode != 0:
            raise RuntimeError(
                "Failed to repair QP binary runtime dependencies.\n"
                f"target={target}\n"
                f"stdout:\n{run.stdout}\n"
                f"stderr:\n{run.stderr}"
            )
        dep_repairs.append(
            {
                "target": str(target),
                "changes": [{"old": old_dep, "new": new_dep} for old_dep, new_dep in changes],
            }
        )

    return {
        "lib_dir": str(lib_dir),
        "id_repairs": id_repairs,
        "dependency_repairs": dep_repairs,
    }


def sync_workspace_ccik_plugin(
    qp: QPPaths | Path,
    *,
    workspace_plugin_dir: Path | None = None,
) -> dict[str, object]:
    src_root = resolve_workspace_ccik_plugin_dir(workspace_plugin_dir)
    dst_root = resolve_installed_ccik_plugin_dir(qp, create=True)
    copied_files: list[str] = []

    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        shutil.copy2(src, dst)
        copied_files.append(rel.as_posix())

    return {
        "workspace_plugin_dir": str(src_root),
        "installed_plugin_dir": str(dst_root),
        "copied_count": int(len(copied_files)),
        "copied_files": copied_files,
    }


def ensure_ccik_plugin_installed(qp: QPPaths | Path) -> dict[str, object]:
    qp_paths = qp if isinstance(qp, QPPaths) else discover_qp_paths_from_root(Path(qp))
    plugin_dir = resolve_installed_ccik_plugin_dir(qp_paths, create=True)
    src_link = (qp_paths.qp_root / "src" / "ccik").resolve(strict=False)
    desired_target = plugin_dir.resolve()

    if src_link.exists() and src_link == desired_target:
        return {
            "already_installed": True,
            "src_link": str(qp_paths.qp_root / "src" / "ccik"),
            "plugin_dir": str(plugin_dir),
            "command": None,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }

    src_path = qp_paths.qp_root / "src" / "ccik"
    if src_path.exists() and src_link != desired_target:
        raise RuntimeError(
            "Fresh-root CCIK installation is ambiguous: src/ccik already exists but does not point "
            f"to the installed plugin directory.\n"
            f"src/ccik={src_path}\n"
            f"resolved={src_link}\n"
            f"expected={desired_target}"
        )

    quoted_rc = shlex.quote(str(qp_paths.quantum_package_rc))
    cmd = f"source {quoted_rc} >/dev/null 2>&1 && qp_plugins install ccik"
    run = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        check=False,
        text=True,
        capture_output=True,
        cwd=str(qp_paths.qp_root),
        env={**os.environ, "QP_ROOT": str(qp_paths.qp_root)},
    )
    if run.returncode != 0:
        raise RuntimeError(
            "Failed to install the CCIK plugin into the fresh QP root.\n"
            f"Command: {cmd}\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}"
        )
    if not src_path.is_symlink() or src_path.resolve() != desired_target:
        raise RuntimeError(
            "qp_plugins install ccik completed but src/ccik was not created correctly.\n"
            f"src/ccik={src_path}\n"
            f"expected_target={desired_target}"
        )
    return {
        "already_installed": False,
        "src_link": str(src_path),
        "plugin_dir": str(plugin_dir),
        "command": cmd,
        "returncode": int(run.returncode),
        "stdout": run.stdout,
        "stderr": run.stderr,
    }


def rebuild_ccik_plugin(
    qp: QPPaths,
    *,
    workspace_plugin_dir: Path | None = None,
    sync_workspace: bool = True,
    clean: bool = True,
    repair_runtime_install_names: bool = False,
) -> dict[str, object]:
    sync_summary = None
    if sync_workspace:
        sync_summary = sync_workspace_ccik_plugin(qp, workspace_plugin_dir=workspace_plugin_dir)
    install_summary = ensure_ccik_plugin_installed(qp)

    plugin_dir = resolve_installed_ccik_plugin_dir(qp)
    quoted_rc = shlex.quote(str(qp.quantum_package_rc))
    quoted_plugin = shlex.quote(str(plugin_dir))
    quoted_target = shlex.quote(str(qp.ccik_binary))
    quoted_plugin_target = shlex.quote(qp.ccik_binary.name)
    quoted_qp_create_ninja = shlex.quote(str((qp.qp_root / "scripts" / "compilation" / "qp_create_ninja").resolve()))
    build_ninja_kind = "top_level"
    bootstrap_summary = None
    force_refresh_ninja = bool(sync_summary and int(sync_summary.get("copied_count", 0)) > 0)
    force_refresh_ninja = force_refresh_ninja or (not bool(install_summary.get("already_installed", False)))
    plugin_irpf90_temp = plugin_dir / "IRPF90_temp"
    if force_refresh_ninja and plugin_irpf90_temp.exists():
        shutil.rmtree(plugin_irpf90_temp)
    try:
        if force_refresh_ninja:
            raise RuntimeError("plugin_ninja_refresh_required")
        build_ninja = resolve_plugin_build_ninja(qp)
        build_ninja_kind = "plugin_local"
    except RuntimeError:
        bootstrap_cmd = f"source {quoted_rc} >/dev/null 2>&1 && {quoted_qp_create_ninja} update"
        bootstrap_run = subprocess.run(
            ["/bin/bash", "-lc", bootstrap_cmd],
            check=False,
            text=True,
            capture_output=True,
            cwd=str(qp.qp_root),
            env={**os.environ, "QP_ROOT": str(qp.qp_root)},
        )
        bootstrap_summary = {
            "command": bootstrap_cmd,
            "returncode": int(bootstrap_run.returncode),
            "stdout": bootstrap_run.stdout,
            "stderr": bootstrap_run.stderr,
        }
        if bootstrap_run.returncode != 0:
            raise RuntimeError(
                "Failed to refresh QP ninja metadata after syncing the installed CCIK plugin.\n"
                f"Command: {bootstrap_cmd}\n"
                f"stdout:\n{bootstrap_run.stdout}\n"
                f"stderr:\n{bootstrap_run.stderr}"
            )
        try:
            build_ninja = resolve_plugin_build_ninja(qp)
            build_ninja_kind = "plugin_local"
        except RuntimeError:
            build_ninja = resolve_qp_build_ninja(qp)
            build_ninja_kind = "top_level"

    quoted_build_ninja = shlex.quote(str(build_ninja))
    run_cwd = plugin_dir if build_ninja_kind == "plugin_local" else qp.qp_root
    if build_ninja_kind == "plugin_local":
        plugin_wrapper_ninja = build_ninja.parent.resolve() == plugin_dir.resolve()
        if plugin_wrapper_ninja:
            clean_cmd = f"ninja -f {quoted_build_ninja} clean && "
            build_cmd = f"ninja -f {quoted_build_ninja} local"
        else:
            clean_cmd = f"ninja -f {quoted_build_ninja} -t clean && "
            build_cmd = f"ninja -f {quoted_build_ninja} {quoted_target}"
        if clean:
            cmd = (
                f"source {quoted_rc} >/dev/null 2>&1 && "
                f"{clean_cmd}"
                f"{build_cmd}"
            )
        else:
            cmd = f"source {quoted_rc} >/dev/null 2>&1 && {build_cmd}"
    else:
        has_plugin_makefile = (plugin_dir / "Makefile").exists()
        if clean and has_plugin_makefile:
            cmd = (
                f"source {quoted_rc} >/dev/null 2>&1 && "
                f"make -C {quoted_plugin} veryclean && "
                f"ninja -f {quoted_build_ninja} {quoted_target}"
            )
        else:
            cmd = f"source {quoted_rc} >/dev/null 2>&1 && ninja -f {quoted_build_ninja} {quoted_target}"
    run = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        check=False,
        text=True,
        capture_output=True,
        cwd=str(run_cwd),
        env={**os.environ, "QP_ROOT": str(qp.qp_root)},
    )

    runtime_patch_summary = None
    if repair_runtime_install_names:
        runtime_patch_summary = repair_qp_runtime_install_names(qp)

    payload = {
        "plugin_dir": str(plugin_dir),
        "build_ninja": str(build_ninja),
        "build_ninja_kind": build_ninja_kind,
        "install_summary": install_summary,
        "bootstrap_summary": bootstrap_summary,
        "command": cmd,
        "returncode": int(run.returncode),
        "stdout": run.stdout,
        "stderr": run.stderr,
        "sync_summary": sync_summary,
        "runtime_patch_summary": runtime_patch_summary,
        "binary_path": str(qp.ccik_binary),
        "binary_exists": bool(qp.ccik_binary.exists()),
    }
    if run.returncode != 0:
        raise RuntimeError(
            "Failed to rebuild the installed QP CCIK plugin.\n"
            f"Command: {cmd}\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}"
        )
    if not qp.ccik_binary.exists():
        raise RuntimeError(f"QP rebuild completed but the CCIK binary is missing: {qp.ccik_binary}")
    return payload


def format_orbital_range(start: int, stop: int) -> str:
    if start > stop:
        return "[]"
    return f"[{start}-{stop}]"


def _read_ezfio_text(ezfio: Path, relpath: str) -> str | None:
    target = ezfio / relpath
    if not target.exists():
        return None
    return target.read_text(encoding="ascii").strip()


def _read_ezfio_gzip_lines(ezfio: Path, relpath: str) -> list[str] | None:
    target = ezfio / relpath
    if not target.exists():
        return None
    with gzip.open(target, "rt", encoding="ascii") as fh:
        return [line.strip() for line in fh if line.strip()]


def _read_ezfio_int(ezfio: Path, relpath: str) -> int | None:
    text = _read_ezfio_text(ezfio, relpath)
    if text is None:
        return None
    return int(text)


def load_prepared_active_space(ezfio: Path, *, fallback_metadata: Path | None = None) -> PreparedActiveSpace:
    mo_class_lines = _read_ezfio_gzip_lines(ezfio, "mo_basis/mo_class.gz")
    mo_num: int | None = None
    ncore: int | None = None
    ncas: int | None = None
    source = "mo_class"

    if mo_class_lines is not None and len(mo_class_lines) >= 3:
        try:
            mo_num = int(mo_class_lines[1])
        except ValueError:
            mo_num = None
        classes = mo_class_lines[2:]
        if mo_num is None:
            mo_num = len(classes)
        classes = classes[:mo_num]
        ncore = sum(1 for label in classes if label == "Core")
        ncas = sum(1 for label in classes if label == "Active")

    if (mo_num is None or ncore is None or ncas is None) and fallback_metadata is not None and fallback_metadata.exists():
        payload = json.loads(fallback_metadata.read_text(encoding="utf-8"))
        mo_num = int(payload["mo_num"])
        ncore = int(payload["ncore"])
        ncas = int(payload["ncas"])
        source = "prepared_metadata"

    if mo_num is None or ncore is None or ncas is None:
        raise RuntimeError(f"Unable to derive active-space metadata from {ezfio}")

    neleca = _read_ezfio_int(ezfio, "electrons/elec_alpha_num")
    nelecb = _read_ezfio_int(ezfio, "electrons/elec_beta_num")
    if neleca is None or nelecb is None:
        raise RuntimeError(f"Missing electron counts in {ezfio}")
    nelecas = int(neleca + nelecb - 2 * ncore)

    return PreparedActiveSpace(
        mo_num=int(mo_num),
        ncore=int(ncore),
        ncas=int(ncas),
        neleca=int(neleca),
        nelecb=int(nelecb),
        nelecas=int(nelecas),
        source=source,
    )


def _coerce_qp_default_token(token: str) -> object:
    stripped = token.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _seed_defaults_from_qp_root(ezfio: Path, qp_root: Path) -> None:
    defaults_dir = Path(qp_root) / "data" / "ezfio_defaults"
    if not defaults_dir.exists():
        return
    for default_file in sorted(defaults_dir.glob("*.ezfio_interface_default")):
        try:
            lines = default_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        module = None
        for raw in lines:
            line = raw.rstrip()
            if not line.strip():
                continue
            if module is None:
                module = line.strip()
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pieces = stripped.split(None, 1)
            if len(pieces) != 2:
                continue
            key, value = pieces
            target = ezfio / module / key
            if target.exists():
                continue
            write_ezfio_value(ezfio, f"{module}/{key}", _coerce_qp_default_token(value))


def ensure_ezfio_defaults(ezfio: Path, *, qp_root: Path | None = None) -> None:
    defaults = {
        "basis/primitives_normalized": "T\n\n",
        "basis/ao_normalized": "F\n\n",
        "nuclei/point_charges": "F\n\n",
        "nuclei/is_periodic": "F\n\n",
        "nuclei/io_nuclear_repulsion": "None\n",
        "determinants/read_wf": "F\n\n",
        "determinants/n_det": "                   1\n",
        "determinants/n_det_max": "             1000000\n",
        "determinants/n_det_qp_edit": "                   1\n",
        "determinants/n_det_print_wf": "               10000\n",
        "determinants/n_states": "                   1\n",
        "determinants/pruning": "   0.000000000000000e+00\n",
        "determinants/pseudo_sym": "F\n\n",
        "determinants/s2_eig": "T\n\n",
        "determinants/save_threshold": "   1.000000000000000e-14\n",
        "determinants/selection_factor": "   1.000000000000000e+00\n",
        "determinants/target_energy": "   0.000000000000000e+00\n",
        "determinants/thresh_sym": "   1.000000000000000e-15\n",
        "determinants/threshold_generators": "   9.990000000000000e-01\n",
        "determinants/weight_one_e_dm": "                   2\n",
        "determinants/weight_selection": "                   1\n",
        "scf_utils/do_mom": "F\n\n",
        "scf_utils/frozen_orb_scf": "F\n\n",
        "scf_utils/level_shift": "   0.000000000000000e+00\n",
        "scf_utils/max_dim_diis": "                  15\n",
        "scf_utils/mo_guess_type": "Huckel\n",
        "scf_utils/n_it_scf_max": "                 500\n",
        "scf_utils/no_oa_or_av_opt": "F\n\n",
        "scf_utils/rohf_parameters": "Roothaan\n",
        "scf_utils/scf_algorithm": "DIIS\n",
        "scf_utils/threshold_diis": "   0.000000000000000e+00\n",
        "scf_utils/use_optimal_damping": "F\n\n",
        "pseudo/do_pseudo": "F\n\n",
        "pseudo/pseudo_grid_rmax": "   1.000000000000000E+01\n",
        "pseudo/pseudo_grid_size": "                1000\n",
        "ccik/ccik_m": "                  32\n",
        "ccik/ccik_score_shift": "   1.000000000000000E-12\n",
        "ccik/ccik_external_candidate_factor": "                   4\n",
        "ccik/ccik_pt2_rank_weight": "   1.000000000000000E+00\n",
        "ccik/ccik_matvec_backend": "connected_sd_fullq\n",
        "ccik/ccik_compression_policy": "fixed_budget\n",
        "ccik/ccik_fixed_store_keep": "               65536\n",
        "ccik/ccik_fixed_propagate_keep": "                 512\n",
        "ccik/ccik_store_abs_tol": "   0.000000000000000E+00\n",
        "ccik/ccik_propagate_abs_tol": "   0.000000000000000E+00\n",
        "ccik/ccik_ritz_overlap_tol": "   1.000000000000000E-08\n",
        "ccik/ccik_ritz_max_drop": "   0.000000000000000E+00\n",
        "ccik/ccik_debug_trace": "F\n\n",
        "ccik/ccik_debug_trace_topn": "                   8\n",
        "ccik/ccik_save_final_wf": "T\n\n",
        "ccik/ccik_use_existing_wf": "F\n\n",
        "ccik/ccik_do_post_pt2": "F\n\n",
        "ezfio_files/ezfio_convention": "            20210101\n",
        "json/empty": "F\n",
        "ao_two_e_ints/io_ao_two_e_integrals": "Write\n",
        "ao_two_e_ints/io_ao_cholesky": "None\n",
        "ao_two_e_ints/threshold_ao": "   1.000000000000000e-15\n",
        "ao_two_e_ints/ao_cholesky_threshold": "   1.000000000000000e-12\n",
        "ao_two_e_ints/do_ao_cholesky": "F\n\n",
        "ao_two_e_ints/io_ao_two_e_integrals_erf": "None\n",
        "ao_two_e_ints/use_only_lr": "F\n\n",
        "ao_one_e_ints/io_ao_integrals_kinetic": "None\n",
        "ao_one_e_ints/io_ao_integrals_n_e": "None\n",
        "ao_one_e_ints/io_ao_integrals_overlap": "None\n",
        "ao_one_e_ints/io_ao_integrals_pseudo": "None\n",
        "ao_one_e_ints/io_ao_one_e_integrals": "None\n",
        "ao_one_e_ints/lin_dep_cutoff": "   1.000000000000000e-06\n",
        "ao_one_e_ints/threshold_ao": "   1.000000000000000e-15\n",
        "mo_two_e_ints/do_mo_cholesky": "F\n\n",
        "mo_two_e_ints/io_mo_cholesky": "None\n",
        "mo_two_e_ints/io_mo_two_e_integrals": "Write\n",
        "mo_two_e_ints/io_mo_two_e_integrals_erf": "None\n",
        "mo_two_e_ints/mo_cholesky_double": "T\n\n",
        "mo_two_e_ints/mo_integrals_cache_shift": "                   7\n",
        "mo_two_e_ints/threshold_mo": "   1.000000000000000e-15\n",
        "mo_one_e_ints/io_mo_integrals_kinetic": "None\n",
        "mo_one_e_ints/io_mo_integrals_n_e": "None\n",
        "mo_one_e_ints/io_mo_integrals_pseudo": "None\n",
        "mo_one_e_ints/io_mo_one_e_integrals": "None\n",
    }
    for relpath, text in defaults.items():
        target = ezfio / relpath
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="ascii")

    resolved_qp_root = qp_root
    if resolved_qp_root is None:
        env_qp_root = os.environ.get("QP_ROOT")
        if env_qp_root:
            resolved_qp_root = Path(env_qp_root)
    if resolved_qp_root is not None:
        _seed_defaults_from_qp_root(ezfio, Path(resolved_qp_root))

    if DEFAULT_EZFIO_DONOR.exists():
        allowed_prefixes = (
            "casscf_cipsi/",
            "ccik/",
            "cipsi/",
            "density_for_dft/",
            "determinants/",
            "dft_keywords/",
            "hamiltonian/",
            "json/",
            "mol_properties/",
            "nuclei/",
            "scf_utils/",
            "trexio/",
            "utils_cc/",
        )
        for source in DEFAULT_EZFIO_DONOR.rglob("*"):
            if not source.is_file():
                continue
            rel = source.relative_to(DEFAULT_EZFIO_DONOR)
            rel_str = rel.as_posix()
            if not rel_str.startswith(allowed_prefixes):
                continue
            target = ezfio / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())


def ezfio_scalar_text(value: object) -> str:
    if isinstance(value, bool):
        return ("T" if value else "F") + "\n\n"
    if isinstance(value, int):
        return f"{value:20d}\n"
    if isinstance(value, float):
        return f"{value:24.15E}\n"
    return f"{value}\n"


def write_ezfio_value(ezfio: Path, relpath: str, value: object) -> None:
    target = ezfio / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ezfio_scalar_text(value), encoding="ascii")


def run_qp_setup(
    qp: QPPaths,
    *,
    basis: str,
    ezfio: Path,
    xyz: Path,
    ncore: int,
    ncas: int,
    nmo_total: int,
) -> None:
    bash(
        f"source {qp.quantum_package_rc} && "
        f"{qp.qp_create_ezfio} -b {basis} -m 1 -o {ezfio} {xyz}"
    )
    ensure_ezfio_defaults(ezfio, qp_root=qp.qp_root)
    bash(
        f"source {qp.quantum_package_rc} && "
        f"qp set_file {ezfio} && "
        "qp set scf_utils thresh_scf 1.e-10 && "
        "qp set ao_two_e_ints io_ao_two_e_integrals Write && "
        "qp set mo_two_e_ints io_mo_two_e_integrals Write"
    )
    ensure_ezfio_defaults(ezfio, qp_root=qp.qp_root)
    bash(
        f"source {qp.quantum_package_rc} && "
        f"{qp.scf_binary} {ezfio}"
    )

    first_act = ncore + 1
    last_act = ncore + ncas
    last_del = nmo_total
    pieces = [f"source {qp.quantum_package_rc} && {qp.qp_set_mo_class}"]
    if ncore > 0:
        pieces.append(f'--core="{format_orbital_range(1, ncore)}"')
    pieces.append(f'--act="{format_orbital_range(first_act, last_act)}"')
    if last_act < last_del:
        pieces.append(f'--del="{format_orbital_range(last_act + 1, last_del)}"')
    pieces.append(str(ezfio))
    bash(" ".join(pieces))


PARAM_ORDER = (
    "ccik_basis_size",
    "ccik_generator_keep",
    "ccik_generator_fraction",
    "ccik_generator_abs_coeff_cutoff",
    "ccik_max_parents_for_expansion",
    "ccik_include_current_support",
    "ccik_selector_keep",
    "ccik_external_keep",
    "ccik_external_candidate_factor",
    "ccik_pt2_rank_weight",
    "ccik_selector_mode",
    "ccik_matvec_backend",
    "ccik_compression_policy",
    "ccik_fixed_store_keep",
    "ccik_fixed_propagate_keep",
    "ccik_store_abs_tol",
    "ccik_propagate_abs_tol",
    "ccik_nkeep",
    "ccik_tmp_keep_factor",
    "ccik_preserve_prev_support",
    "ccik_preserve_prev_support_cap",
    "ccik_total_det_budget",
    "ccik_conv_tol",
    "ccik_ritz_overlap_tol",
    "ccik_ritz_max_drop",
    "ccik_save_final_wf",
    "ccik_use_existing_wf",
    "ccik_do_post_pt2",
    "ccik_max_cycle",
    "ccik_restart_m_cycle",
    "ccik_restart_nroot",
    "ccik_verbose",
)


def _qp_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def set_ccik_params(qp: QPPaths, ezfio: Path, params: CCIKParams) -> None:
    ensure_ezfio_defaults(ezfio, qp_root=qp.qp_root)
    for name in PARAM_ORDER:
        write_ezfio_value(ezfio, f"ccik/{name}", getattr(params, name))


def _extract_value(text: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)} =\s*([-+0-9.eE]+)", text)
    if not m:
        raise RuntimeError(f"Could not find `{key}` in plugin output")
    return float(m.group(1))


def run_ccik(qp: QPPaths, ezfio: Path) -> tuple[float, float, int, str]:
    quoted_rc = shlex.quote(str(qp.quantum_package_rc))
    quoted_ezfio = shlex.quote(str(ezfio))
    quoted_ccik = shlex.quote(str(qp.ccik_binary))
    out = bash(
        f"source {quoted_rc} && "
        f"qp set_file {quoted_ezfio} >/dev/null && "
        f"{quoted_ccik} {quoted_ezfio}"
    ).stdout
    e_elec = _extract_value(out, "CCIK_FINAL_ELECTRONIC_ENERGY")
    e_tot = _extract_value(out, "CCIK_FINAL_TOTAL_ENERGY")
    basis = int(_extract_value(out, "CCIK_FINAL_BASIS"))
    return e_elec, e_tot, basis, out


def run_qp_binary(qp: QPPaths, ezfio: Path, binary: Path, *, extra_setup: list[str] | None = None) -> str:
    setup = extra_setup or []
    quoted_rc = shlex.quote(str(qp.quantum_package_rc))
    quoted_ezfio = shlex.quote(str(ezfio))
    quoted_binary = shlex.quote(str(binary))
    cmd = (
        f"source {quoted_rc} && "
        f"qp set_file {quoted_ezfio} >/dev/null && "
        + " && ".join(setup + [f"{quoted_binary} {quoted_ezfio}"])
    )
    return bash(cmd).stdout


def read_ezfio_scalar_array(path: Path) -> list[float]:
    text = path.read_text(encoding="ascii", errors="ignore").strip()
    if not text:
        return []
    text = text.replace("[", " ").replace("]", " ").replace(",", " ")
    vals: list[float] = []
    for tok in text.split():
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals


def run_pt2(qp: QPPaths, ezfio: Path, *, relative_error: float = 0.05) -> dict[str, object]:
    out = run_qp_binary(
        qp,
        ezfio,
        qp.pt2_binary,
        extra_setup=[
            "qp set perturbation do_pt2 True",
            f"qp set perturbation pt2_relative_error {relative_error}",
        ],
    )
    energy = read_ezfio_scalar_array(ezfio / "fci" / "energy")
    energy_pt2 = read_ezfio_scalar_array(ezfio / "fci" / "energy_pt2")
    result: dict[str, object] = {
        "stdout": out,
        "fci_energy": energy[0] if energy else None,
        "fci_energy_pt2": energy_pt2[0] if energy_pt2 else None,
    }
    if result["fci_energy"] is not None and result["fci_energy_pt2"] is not None:
        result["pt2_correction"] = float(result["fci_energy_pt2"]) - float(result["fci_energy"])
    else:
        result["pt2_correction"] = None
    return result


def run_orb_opt(qp: QPPaths, ezfio: Path, *, optimization_method: str = "diag") -> str:
    return run_qp_binary(
        qp,
        ezfio,
        qp.orb_opt_binary,
        extra_setup=[
            f"qp set mo_optimization_utils optimization_method {optimization_method}",
        ],
    )


def derived_thick_nkeep(params: CCIKParams) -> int | None:
    if params.ccik_total_det_budget <= 0:
        return None
    if params.ccik_restart_m_cycle <= 1 or params.ccik_max_cycle <= 1:
        return None
    denom = 2 * params.ccik_restart_m_cycle + max(1, params.ccik_restart_nroot)
    return max(
        1,
        (params.ccik_total_det_budget - params.ccik_restart_m_cycle * params.ccik_external_keep) // denom,
    )


def effective_nkeep(params: CCIKParams) -> int:
    derived = derived_thick_nkeep(params)
    return params.ccik_nkeep if derived is None else int(derived)


def estimate_thick_storage_slots(params: CCIKParams) -> int | None:
    if params.ccik_restart_m_cycle <= 1 or params.ccik_max_cycle <= 1:
        return None
    nkeep_eff = effective_nkeep(params)
    return (
        params.ccik_restart_m_cycle * (2 * nkeep_eff + params.ccik_external_keep)
        + max(1, params.ccik_restart_nroot) * nkeep_eff
    )
