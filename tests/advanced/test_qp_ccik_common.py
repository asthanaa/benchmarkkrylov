from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = ROOT / "scripts" / "_shared" / "qp_ccik_common.py"
SPEC = importlib.util.spec_from_file_location('qp_ccik_common', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CCIKParams = MODULE.CCIKParams
collect_qp_macho_validation = MODULE.collect_qp_macho_validation
detect_physical_cpu_count = MODULE.detect_physical_cpu_count
derived_thick_nkeep = MODULE.derived_thick_nkeep
ensure_ezfio_defaults = MODULE.ensure_ezfio_defaults
estimate_thick_storage_slots = MODULE.estimate_thick_storage_slots
ensure_ccik_plugin_installed = MODULE.ensure_ccik_plugin_installed
infer_qp_root_from_rebuild_manifest = MODULE.infer_qp_root_from_rebuild_manifest
qp_parallel_env = MODULE.qp_parallel_env
repair_qp_runtime_install_names = MODULE.repair_qp_runtime_install_names
resolve_omp_threads = MODULE.resolve_omp_threads
resolve_rebuildable_qp_root = MODULE.resolve_rebuildable_qp_root
resolve_installed_ccik_plugin_dir = MODULE.resolve_installed_ccik_plugin_dir
resolve_plugin_build_ninja = MODULE.resolve_plugin_build_ninja
resolve_qp_build_ninja = MODULE.resolve_qp_build_ninja
resolve_workspace_ccik_plugin_dir = MODULE.resolve_workspace_ccik_plugin_dir
rebuild_ccik_plugin = MODULE.rebuild_ccik_plugin
sync_workspace_ccik_plugin = MODULE.sync_workspace_ccik_plugin
validate_fresh_qp_root = MODULE.validate_fresh_qp_root
validate_rebuildable_qp_root = MODULE.validate_rebuildable_qp_root
write_ezfio_value = MODULE.write_ezfio_value


def test_ccik_driver_refreshes_wavefunction_providers_after_load() -> None:
    driver = (ROOT / "ccik_qp" / "stable" / "ccik" / "ccik_driver.irp.f").read_text(encoding="ascii")
    assert "subroutine ccik_refresh_working_wf_providers()" in driver
    assert "PROVIDE psi_det_size psi_det psi_coef N_states N_det" in driver
    assert "PROVIDE psi_det_generators psi_det_sorted_bit n_act_orb list_act list_act_reverse psi_det_propagators_sorted psi_det_working_sorted" in driver
    assert "call ccik_capture_working_vector(current_vec)" in driver


def test_ccik_driver_has_stale_topk_generator_guard() -> None:
    driver = (ROOT / "ccik_qp" / "stable" / "ccik" / "ccik_driver.irp.f").read_text(encoding="ascii")
    assert "stale generator/provider state detected in connected_sd_fullq" in driver
    assert "compression_policy == 'topk'" in driver
    assert "N_det_generators <= 1" in driver


def test_resolve_omp_threads_accepts_auto_and_integer(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, 'detect_physical_cpu_count', lambda: 24)
    assert resolve_omp_threads('auto') == 24
    assert resolve_omp_threads(None) == 24
    assert resolve_omp_threads('8') == 8
    assert resolve_omp_threads(6) == 6


def test_qp_parallel_env_caps_blas_threads(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, 'detect_physical_cpu_count', lambda: 24)
    env = qp_parallel_env('auto')
    assert env['OMP_NUM_THREADS'] == '24'
    assert env['OMP_PROC_BIND'] == 'spread'
    assert env['OMP_PLACES'] == 'cores'
    assert env['OPENBLAS_NUM_THREADS'] == '1'
    assert env['MKL_NUM_THREADS'] == '1'
    assert env['VECLIB_MAXIMUM_THREADS'] == '1'


def test_budget_derivation_and_storage_estimate() -> None:
    params = CCIKParams(
        ccik_basis_size=16,
        ccik_generator_keep=512,
        ccik_generator_fraction=0.0,
        ccik_selector_keep=32768,
        ccik_external_keep=32768,
        ccik_pt2_rank_weight=1.0,
        ccik_nkeep=32768,
        ccik_total_det_budget=10_000_000,
        ccik_conv_tol=1e-6,
        ccik_ritz_overlap_tol=1e-8,
        ccik_ritz_max_drop=5.0,
        ccik_save_final_wf=True,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=False,
        ccik_max_cycle=10,
        ccik_restart_m_cycle=10,
        ccik_restart_nroot=2,
        ccik_verbose=False,
    )
    assert derived_thick_nkeep(params) == 439650
    assert estimate_thick_storage_slots(params) == 9_999_980


def test_storage_estimate_uses_manual_nkeep_without_budget() -> None:
    params = CCIKParams(
        ccik_basis_size=16,
        ccik_generator_keep=512,
        ccik_generator_fraction=0.0,
        ccik_selector_keep=32768,
        ccik_external_keep=32768,
        ccik_pt2_rank_weight=1.0,
        ccik_nkeep=32768,
        ccik_total_det_budget=0,
        ccik_conv_tol=1e-6,
        ccik_ritz_overlap_tol=1e-8,
        ccik_ritz_max_drop=5.0,
        ccik_save_final_wf=True,
        ccik_use_existing_wf=False,
        ccik_do_post_pt2=False,
        ccik_max_cycle=10,
        ccik_restart_m_cycle=10,
        ccik_restart_nroot=2,
        ccik_verbose=False,
    )
    assert derived_thick_nkeep(params) is None
    assert estimate_thick_storage_slots(params) == 1_048_576


def test_ensure_ezfio_defaults_writes_threshold_knobs(tmp_path: Path) -> None:
    ezfio = tmp_path / "case.ezfio"
    ensure_ezfio_defaults(ezfio)
    assert (ezfio / "ccik" / "ccik_compression_policy").read_text(encoding="ascii").strip() == "fixed_budget"
    assert (ezfio / "ccik" / "ccik_fixed_store_keep").read_text(encoding="ascii").strip() == "65536"
    assert (ezfio / "ccik" / "ccik_fixed_propagate_keep").read_text(encoding="ascii").strip() == "512"
    assert float((ezfio / "ccik" / "ccik_store_abs_tol").read_text(encoding="ascii").strip()) == 0.0
    assert float((ezfio / "ccik" / "ccik_propagate_abs_tol").read_text(encoding="ascii").strip()) == 0.0


def test_resolve_installed_ccik_plugin_dir(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    assert resolve_installed_ccik_plugin_dir(qp_root) == plugin_dir.resolve()


def test_resolve_qp_build_ninja(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    build_ninja = qp_root / "config" / "build.ninja"
    build_ninja.parent.mkdir(parents=True)
    build_ninja.write_text("build all: phony\n", encoding="ascii")
    assert resolve_qp_build_ninja(qp_root) == build_ninja.resolve()


def test_resolve_plugin_build_ninja(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_build_ninja = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik" / "IRPF90_temp" / "build.ninja"
    plugin_build_ninja.parent.mkdir(parents=True)
    plugin_build_ninja.write_text("build all: phony\n", encoding="ascii")
    assert resolve_plugin_build_ninja(qp_root) == plugin_build_ninja.resolve()


def test_resolve_plugin_build_ninja_prefers_irpf90_over_wrapper(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    irpf90_build_ninja = plugin_dir / "IRPF90_temp" / "build.ninja"
    wrapper_build_ninja = plugin_dir / "build.ninja"
    irpf90_build_ninja.parent.mkdir(parents=True)
    wrapper_build_ninja.parent.mkdir(parents=True, exist_ok=True)
    irpf90_build_ninja.write_text("build ccik: phony\n", encoding="ascii")
    wrapper_build_ninja.write_text("build local: phony\n", encoding="ascii")
    assert resolve_plugin_build_ninja(qp_root) == irpf90_build_ninja.resolve()


def test_sync_workspace_ccik_plugin_copies_only_changed_files(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    installed = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    installed.mkdir(parents=True)
    workspace = tmp_path / "workspace" / "ccik"
    workspace.mkdir(parents=True)

    (workspace / "ccik_driver.irp.f").write_text("new driver\n", encoding="ascii")
    (workspace / "README.rst").write_text("new readme\n", encoding="ascii")
    (installed / "ccik_driver.irp.f").write_text("old driver\n", encoding="ascii")
    (installed / "README.rst").write_text("new readme\n", encoding="ascii")

    summary = sync_workspace_ccik_plugin(qp_root, workspace_plugin_dir=workspace)

    assert summary["copied_count"] == 1
    assert summary["copied_files"] == ["ccik_driver.irp.f"]
    assert (installed / "ccik_driver.irp.f").read_text(encoding="ascii") == "new driver\n"


def test_sync_workspace_ccik_plugin_creates_installed_dir_when_missing(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    workspace = tmp_path / "workspace" / "ccik"
    workspace.mkdir(parents=True)
    (workspace / "ccik_driver.irp.f").write_text("new driver\n", encoding="ascii")

    summary = sync_workspace_ccik_plugin(qp_root, workspace_plugin_dir=workspace)

    installed = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    assert installed.is_dir()
    assert summary["installed_plugin_dir"] == str(installed.resolve())
    assert (installed / "ccik_driver.irp.f").read_text(encoding="ascii") == "new driver\n"


def test_ensure_ccik_plugin_installed_runs_qp_plugins_install(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    quantum_package_rc = qp_root / "quantum_package.rc"
    quantum_package_rc.parent.mkdir(parents=True, exist_ok=True)
    quantum_package_rc.write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")

    commands: list[list[str]] = []

    def fake_run(cmd, check=False, text=True, capture_output=True, cwd=None, env=None):
        commands.append([str(part) for part in cmd])
        (qp_root / "src").mkdir(parents=True, exist_ok=True)
        src_link = qp_root / "src" / "ccik"
        if not src_link.exists():
            src_link.symlink_to(plugin_dir, target_is_directory=True)

        class Result:
            returncode = 0
            stdout = "installed"
            stderr = ""

        return Result()

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    summary = ensure_ccik_plugin_installed(qp_root)

    assert summary["already_installed"] is False
    assert any("qp_plugins install ccik" in " ".join(cmd) for cmd in commands)
    assert (qp_root / "src" / "ccik").is_symlink()


def test_repair_qp_runtime_install_names_rewrites_old_temp_paths(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    lib_dir = qp_root / "lib"
    lib_dir.mkdir(parents=True)
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    tools_dir = qp_root / "src" / "tools"
    tools_dir.mkdir(parents=True)

    libsimple = lib_dir / "libsimple_gpu-cpu.1.dylib"
    libtrexio = lib_dir / "libtrexio.0.dylib"
    libf77zmq = lib_dir / "libf77zmq.0.dylib"
    for path in (libsimple, libtrexio, libf77zmq):
        path.write_text("", encoding="ascii")

    ccik_binary = plugin_dir / "ccik"
    fcidump_binary = tools_dir / "fcidump"
    pt2_binary = qp_root / "src" / "fci" / "pt2"
    orb_opt_binary = qp_root / "src" / "mo_optimization" / "orb_opt"
    for path in (ccik_binary, fcidump_binary, pt2_binary, orb_opt_binary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="ascii")

    qp = MODULE.QPPaths(
        qp_root=qp_root,
        quantum_package_rc=qp_root / "quantum_package.rc",
        qp_set_mo_class=qp_root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio="qp_create_ezfio",
        scf_binary=qp_root / "src" / "hartree_fock" / "scf",
        ccik_binary=ccik_binary,
        pt2_binary=pt2_binary,
        orb_opt_binary=orb_opt_binary,
    )

    calls: list[list[str]] = []
    old_prefix = "/private/tmp/qp2_fresh_20260320/lib"

    def fake_run(cmd, check=False, text=True, capture_output=True, cwd=None, env=None):
        calls.append([str(part) for part in cmd])
        target = Path(cmd[-1])
        if cmd[:2] == ["otool", "-D"]:
            class Result:
                returncode = 0
                stdout = f"{target}:\n{old_prefix}/{target.name}\n"
                stderr = ""
            return Result()
        if cmd[:2] == ["otool", "-L"]:
            if target.name in {"ccik", "fcidump", "pt2", "orb_opt"}:
                deps = [
                    f"\t{old_prefix}/libsimple_gpu-cpu.1.dylib (compatibility version 1.0.0, current version 1.0.0)",
                    f"\t{old_prefix}/libtrexio.0.dylib (compatibility version 1.0.0, current version 1.0.0)",
                    f"\t{old_prefix}/libf77zmq.0.dylib (compatibility version 1.0.0, current version 1.0.0)",
                ]
            else:
                deps = []
            class Result:
                returncode = 0
                stdout = f"{target}:\n" + "\n".join(deps) + ("\n" if deps else "")
                stderr = ""
            return Result()
        if cmd[0] == "install_name_tool":
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""
            return Result()
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    summary = repair_qp_runtime_install_names(qp)

    assert len(summary["id_repairs"]) == 3
    repaired_targets = {Path(item["target"]).name for item in summary["dependency_repairs"]}
    assert {"ccik", "fcidump", "pt2", "orb_opt"}.issubset(repaired_targets)
    assert any(call[0] == "install_name_tool" and "-id" in call for call in calls)
    assert any(call[0] == "install_name_tool" and "-change" in call for call in calls)


def test_resolve_workspace_ccik_plugin_dir_accepts_explicit_override(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace" / "ccik"
    workspace.mkdir(parents=True)
    assert resolve_workspace_ccik_plugin_dir(workspace) == workspace.resolve()


def test_rebuild_ccik_plugin_uses_top_level_ninja(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    (qp_root / "src").mkdir(parents=True)
    (qp_root / "src" / "ccik").symlink_to(plugin_dir, target_is_directory=True)
    lib_dir = qp_root / "lib"
    lib_dir.mkdir(parents=True)
    for name in ("libsimple_gpu-cpu.1.dylib", "libtrexio.0.dylib", "libf77zmq.0.dylib"):
        (lib_dir / name).write_text("", encoding="ascii")
    build_ninja = qp_root / "config" / "build.ninja"
    build_ninja.parent.mkdir(parents=True)
    build_ninja.write_text("build all: phony\n", encoding="ascii")
    quantum_package_rc = qp_root / "quantum_package.rc"
    quantum_package_rc.write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")
    ccik_binary = plugin_dir / "ccik"
    ccik_binary.write_text("", encoding="ascii")

    qp = MODULE.QPPaths(
        qp_root=qp_root,
        quantum_package_rc=quantum_package_rc,
        qp_set_mo_class=qp_root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio="qp_create_ezfio",
        scf_binary=qp_root / "src" / "hartree_fock" / "scf",
        ccik_binary=ccik_binary,
        pt2_binary=qp_root / "src" / "fci" / "pt2",
        orb_opt_binary=qp_root / "src" / "mo_optimization" / "orb_opt",
    )

    captured: dict[str, object] = {}

    def fake_run(cmd, check=False, text=True, capture_output=True, cwd=None, env=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return Result()

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    result = rebuild_ccik_plugin(qp, sync_workspace=False, clean=True)

    assert result["binary_exists"] is True
    assert "ninja -f" in result["command"]
    assert str(build_ninja) in result["command"]
    assert str(ccik_binary) in result["command"]


def test_rebuild_ccik_plugin_prefers_plugin_local_ninja(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    (qp_root / "src").mkdir(parents=True)
    (qp_root / "src" / "ccik").symlink_to(plugin_dir, target_is_directory=True)
    lib_dir = qp_root / "lib"
    lib_dir.mkdir(parents=True)
    for name in ("libsimple_gpu-cpu.1.dylib", "libtrexio.0.dylib", "libf77zmq.0.dylib"):
        (lib_dir / name).write_text("", encoding="ascii")
    (plugin_dir / "IRPF90_temp").mkdir()
    plugin_build_ninja = plugin_dir / "IRPF90_temp" / "build.ninja"
    plugin_build_ninja.write_text("build all: phony\n", encoding="ascii")
    build_ninja = qp_root / "config" / "build.ninja"
    build_ninja.parent.mkdir(parents=True)
    build_ninja.write_text("build all: phony\n", encoding="ascii")
    quantum_package_rc = qp_root / "quantum_package.rc"
    quantum_package_rc.write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")
    ccik_binary = plugin_dir / "ccik"
    ccik_binary.write_text("", encoding="ascii")

    qp = MODULE.QPPaths(
        qp_root=qp_root,
        quantum_package_rc=quantum_package_rc,
        qp_set_mo_class=qp_root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio="qp_create_ezfio",
        scf_binary=qp_root / "src" / "hartree_fock" / "scf",
        ccik_binary=ccik_binary,
        pt2_binary=qp_root / "src" / "fci" / "pt2",
        orb_opt_binary=qp_root / "src" / "mo_optimization" / "orb_opt",
    )

    def fake_run(cmd, check=False, text=True, capture_output=True, cwd=None, env=None):
        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return Result()

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    result = rebuild_ccik_plugin(qp, sync_workspace=False, clean=True)

    assert result["build_ninja_kind"] == "plugin_local"
    assert str(plugin_build_ninja) in result["command"]
    assert "ninja -f" in result["command"]
    assert "-t clean" in result["command"]
    assert str(ccik_binary) in result["command"]
    assert result["runtime_patch_summary"] is None


def test_rebuild_ccik_plugin_accepts_plugin_root_ninja(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    (qp_root / "src").mkdir(parents=True)
    (qp_root / "src" / "ccik").symlink_to(plugin_dir, target_is_directory=True)
    lib_dir = qp_root / "lib"
    lib_dir.mkdir(parents=True)
    for name in ("libsimple_gpu-cpu.1.dylib", "libtrexio.0.dylib", "libf77zmq.0.dylib"):
        (lib_dir / name).write_text("", encoding="ascii")
    (plugin_dir / "build.ninja").write_text("build all: phony\n", encoding="ascii")
    build_ninja = qp_root / "config" / "build.ninja"
    build_ninja.parent.mkdir(parents=True)
    build_ninja.write_text("build all: phony\n", encoding="ascii")
    quantum_package_rc = qp_root / "quantum_package.rc"
    quantum_package_rc.write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")
    ccik_binary = plugin_dir / "ccik"
    ccik_binary.write_text("", encoding="ascii")

    qp = MODULE.QPPaths(
        qp_root=qp_root,
        quantum_package_rc=quantum_package_rc,
        qp_set_mo_class=qp_root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio="qp_create_ezfio",
        scf_binary=qp_root / "src" / "hartree_fock" / "scf",
        ccik_binary=ccik_binary,
        pt2_binary=qp_root / "src" / "fci" / "pt2",
        orb_opt_binary=qp_root / "src" / "mo_optimization" / "orb_opt",
    )

    def fake_run(cmd, check=False, text=True, capture_output=True, cwd=None, env=None):
        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    result = rebuild_ccik_plugin(qp, sync_workspace=False, clean=True)

    assert result["build_ninja_kind"] == "plugin_local"
    assert str(plugin_dir / "build.ninja") in result["command"]
    assert " clean" in result["command"]
    assert " local" in result["command"]


def test_set_ccik_params_writes_ezfio_directly(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    (qp_root / "quantum_package.rc").write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")
    build_ninja = qp_root / "config" / "build.ninja"
    build_ninja.parent.mkdir(parents=True)
    build_ninja.write_text("build all: phony\n", encoding="ascii")
    qp = MODULE.QPPaths(
        qp_root=qp_root,
        quantum_package_rc=qp_root / "quantum_package.rc",
        qp_set_mo_class=qp_root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio="qp_create_ezfio",
        scf_binary=qp_root / "src" / "hartree_fock" / "scf",
        ccik_binary=plugin_dir / "ccik",
        pt2_binary=qp_root / "src" / "fci" / "pt2",
        orb_opt_binary=qp_root / "src" / "mo_optimization" / "orb_opt",
    )
    ezfio = tmp_path / "case.ezfio"
    params = CCIKParams(
        ccik_basis_size=24,
        ccik_selector_mode="abs_residual",
        ccik_matvec_backend="connected_sd_fullq",
        ccik_compression_policy="threshold",
        ccik_store_abs_tol=1e-5,
        ccik_propagate_abs_tol=1e-6,
    )
    MODULE.set_ccik_params(qp, ezfio, params)
    assert int((ezfio / "ccik" / "ccik_basis_size").read_text(encoding="ascii").strip()) == 24
    assert (ezfio / "ccik" / "ccik_compression_policy").read_text(encoding="ascii").strip() == "threshold"
    assert float((ezfio / "ccik" / "ccik_store_abs_tol").read_text(encoding="ascii").strip()) == 1e-5
    assert float((ezfio / "ccik" / "ccik_propagate_abs_tol").read_text(encoding="ascii").strip()) == 1e-6


def test_validate_rebuildable_qp_root(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    (qp_root / "quantum_package.rc").write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")
    build_ninja = qp_root / "config" / "build.ninja"
    build_ninja.parent.mkdir(parents=True)
    build_ninja.write_text("build all: phony\n", encoding="ascii")

    payload = validate_rebuildable_qp_root(qp_root)
    assert payload["is_rebuildable"] is True


def test_collect_qp_macho_validation_flags_forbidden_temp_dependency(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    lib_dir = qp_root / "lib"
    lib_dir.mkdir(parents=True)
    ccik_binary = plugin_dir / "ccik"
    ccik_binary.write_text("", encoding="ascii")
    fcidump_binary = qp_root / "src" / "tools" / "fcidump"
    fcidump_binary.parent.mkdir(parents=True)
    fcidump_binary.write_text("", encoding="ascii")
    dylib = lib_dir / "libsimple_gpu-cpu.1.dylib"
    dylib.write_text("", encoding="ascii")

    qp = MODULE.QPPaths(
        qp_root=qp_root,
        quantum_package_rc=qp_root / "quantum_package.rc",
        qp_set_mo_class=qp_root / "ocaml" / "qp_set_mo_class",
        qp_create_ezfio="qp_create_ezfio",
        scf_binary=qp_root / "src" / "hartree_fock" / "scf",
        ccik_binary=ccik_binary,
        pt2_binary=qp_root / "src" / "fci" / "pt2",
        orb_opt_binary=qp_root / "src" / "mo_optimization" / "orb_opt",
    )

    def fake_run(cmd, check=False, text=True, capture_output=True, cwd=None, env=None):
        target = Path(cmd[-1])
        if cmd[:2] == ["otool", "-D"]:
            class Result:
                returncode = 0
                stdout = f"{target}:\n{target}\n"
                stderr = ""
            return Result()
        if cmd[:2] == ["otool", "-L"]:
            deps = [
                "/private/tmp/qp2_fresh_20260320/lib/libsimple_gpu-cpu.1.dylib (compatibility version 1.0.0, current version 1.0.0)"
            ]
            class Result:
                returncode = 0
                stdout = f"{target}:\n" + "\n".join(f"\t{dep}" for dep in deps) + "\n"
                stderr = ""
            return Result()
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    payload = collect_qp_macho_validation(qp)
    assert payload["passed"] is False
    assert any("forbidden_dependency" in issue for issue in payload["issues"])


def test_validate_fresh_qp_root_combines_structural_macho_and_runtime(monkeypatch, tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    (qp_root / "quantum_package.rc").write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")
    build_ninja = qp_root / "config" / "build.ninja"
    build_ninja.parent.mkdir(parents=True)
    build_ninja.write_text("build all: phony\n", encoding="ascii")
    ccik_binary = plugin_dir / "ccik"
    ccik_binary.write_text("", encoding="ascii")
    fcidump_binary = qp_root / "src" / "tools" / "fcidump"
    fcidump_binary.parent.mkdir(parents=True)
    fcidump_binary.write_text("", encoding="ascii")

    monkeypatch.setattr(MODULE, "collect_qp_macho_validation", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(
        MODULE,
        "_run_qp_runtime_smoke",
        lambda *args, **kwargs: {"passed": True, "new_diagnostic_reports": []},
    )

    payload = validate_fresh_qp_root(qp_root, outdir=tmp_path / "validation")
    assert payload["passed"] is True
    assert payload["structural"]["is_rebuildable"] is True


def test_infer_qp_root_from_rebuild_manifest(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    plugin_dir = qp_root / "plugins" / "ccik_qp" / "stable" / "ccik"
    plugin_dir.mkdir(parents=True)
    manifest = tmp_path / "qp_plugin_rebuild.json"
    manifest.write_text(
        "{\n"
        f'  "plugin_dir": "{plugin_dir}"\n'
        "}\n",
        encoding="utf-8",
    )

    assert infer_qp_root_from_rebuild_manifest(manifest) == qp_root.resolve()


def test_resolve_rebuildable_qp_root_prefers_explicit_then_env_then_manifest(monkeypatch, tmp_path: Path) -> None:
    explicit_root = tmp_path / "explicit"
    env_root = tmp_path / "env"
    manifest_root = tmp_path / "manifest"
    for root in (explicit_root, env_root, manifest_root):
        plugin_dir = root / "plugins" / "ccik_qp" / "stable" / "ccik"
        plugin_dir.mkdir(parents=True)
        (root / "quantum_package.rc").write_text("export QP_ROOT=/tmp/qp2\n", encoding="ascii")
        build_ninja = root / "config" / "build.ninja"
        build_ninja.parent.mkdir(parents=True)
        build_ninja.write_text("build all: phony\n", encoding="ascii")

    manifest = tmp_path / "qp_plugin_rebuild.json"
    manifest.write_text(
        "{\n"
        f'  "binary_path": "{manifest_root / "plugins" / "ccik_qp" / "stable" / "ccik" / "ccik"}"\n'
        "}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("QP_ROOT", str(env_root))
    assert resolve_rebuildable_qp_root(explicit_root, rebuild_manifest=manifest) == explicit_root.resolve()
    assert resolve_rebuildable_qp_root(None, rebuild_manifest=manifest) == env_root.resolve()
    monkeypatch.delenv("QP_ROOT")
    assert resolve_rebuildable_qp_root(None, rebuild_manifest=manifest) == manifest_root.resolve()
