from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = ROOT / "scripts" / "advanced" / "qp_ccik_bridge.py"
SPEC = importlib.util.spec_from_file_location("qp_ccik_bridge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

QPToolchain = MODULE.QPToolchain
_params_from_args = MODULE._params_from_args
build_install_commands = MODULE.build_install_commands
build_run_commands = MODULE.build_run_commands
build_set_param_commands = MODULE.build_set_param_commands
make_parser = MODULE.make_parser
plugin_repo_uri = MODULE.plugin_repo_uri
resolve_ccik_binary = MODULE.resolve_ccik_binary


def test_plugin_repo_uri_is_file_uri(tmp_path: Path) -> None:
    repo = tmp_path / "ccik_qp"
    repo.mkdir()
    uri = plugin_repo_uri(repo)
    assert uri.startswith("file://")
    assert "ccik_qp" in uri


def test_build_install_commands() -> None:
    toolchain = QPToolchain(qp="/usr/bin/qp", qp_plugins="/usr/bin/qp_plugins", qp_root="/tmp/qp2")
    repo = Path("/tmp/ccik_qp")
    cmds = build_install_commands(toolchain, repo)
    assert cmds == [
        ["/usr/bin/qp_plugins", "download", repo.resolve().as_uri()],
        ["/usr/bin/qp_plugins", "install", "ccik"],
    ]


def test_build_set_param_commands() -> None:
    toolchain = QPToolchain(qp="/usr/bin/qp", qp_plugins="/usr/bin/qp_plugins", qp_root="/tmp/qp2")
    cmds = build_set_param_commands(
        toolchain,
        "mol.ezfio",
        {"ccik_basis_size": "32", "ccik_external_keep": "20000"},
    )
    assert cmds == [
        ["/usr/bin/qp", "set_file", "mol.ezfio"],
        ["/usr/bin/qp", "set", "ccik", "ccik_basis_size", "32"],
        ["/usr/bin/qp", "set", "ccik", "ccik_external_keep", "20000"],
    ]


def test_resolve_ccik_binary() -> None:
    toolchain = QPToolchain(qp="/usr/bin/qp", qp_plugins="/usr/bin/qp_plugins", qp_root="/tmp/qp2")
    assert resolve_ccik_binary(toolchain) == str((Path("/tmp/qp2") / "src" / "ccik" / "ccik").resolve())


def test_resolve_ccik_binary_falls_back_to_local_layout(tmp_path: Path) -> None:
    qp_root = tmp_path / "qp2"
    binary = qp_root / "plugins" / "local" / "ccik" / "ccik"
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="ascii")
    toolchain = QPToolchain(qp="/usr/bin/qp", qp_plugins="/usr/bin/qp_plugins", qp_root=str(qp_root))
    assert resolve_ccik_binary(toolchain) == str(binary.resolve())


def test_build_run_commands_direct() -> None:
    toolchain = QPToolchain(qp="/usr/bin/qp", qp_plugins="/usr/bin/qp_plugins", qp_root="/tmp/qp2")
    cmds = build_run_commands(toolchain, "mol.ezfio", mode="direct")
    assert cmds == [
        ["/usr/bin/qp", "set_file", "mol.ezfio"],
        [str((Path("/tmp/qp2") / "src" / "ccik" / "ccik").resolve()), "mol.ezfio"],
    ]


def test_build_run_commands_qp_run() -> None:
    toolchain = QPToolchain(qp="/usr/bin/qp", qp_plugins="/usr/bin/qp_plugins", qp_root="/tmp/qp2")
    cmds = build_run_commands(toolchain, "mol.ezfio", mode="qp-run")
    assert cmds == [
        ["/usr/bin/qp", "set_file", "mol.ezfio"],
        ["/usr/bin/qp", "run", "ccik", "mol.ezfio"],
    ]


def test_params_from_args_accepts_threshold_and_matvec_knobs() -> None:
    parser = make_parser()
    args = parser.parse_args(
        [
            "set-params",
            "mol.ezfio",
            "--ccik-selector-mode",
            "abs_residual",
            "--ccik-matvec-backend",
            "connected_sd_fullq",
            "--ccik-compression-policy",
            "threshold",
            "--ccik-fixed-store-keep",
            "10000",
            "--ccik-fixed-propagate-keep",
            "500",
            "--ccik-secondortho",
            "true",
            "--ccik-store-abs-tol",
            "1e-5",
            "--ccik-propagate-abs-tol",
            "1e-6",
        ]
    )
    params = _params_from_args(args)
    assert params["ccik_selector_mode"] == "abs_residual"
    assert params["ccik_matvec_backend"] == "connected_sd_fullq"
    assert params["ccik_compression_policy"] == "threshold"
    assert params["ccik_fixed_store_keep"] == "10000"
    assert params["ccik_fixed_propagate_keep"] == "500"
    assert params["ccik_secondortho"] == "true"
    assert params["ccik_store_abs_tol"] == "1e-05"
    assert params["ccik_propagate_abs_tol"] == "1e-06"


def test_rebuild_plugin_subcommand_is_available() -> None:
    parser = make_parser()
    args = parser.parse_args(["rebuild-plugin", "--repo", "/tmp/plugin"])
    assert args.command == "rebuild-plugin"
    assert args.repo == "/tmp/plugin"
