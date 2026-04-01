"""Bridge script for running the Quantum Package ccik plugin from CompressK.

This script assumes Quantum Package is installed separately and that the
external plugin repository is available at `ccik_qp/` by default.

It intentionally does not depend on PySCF. Its job is to:

- discover the local QP toolchain
- install/register the plugin repository
- set EZFIO parameters
- run the installed `ccik` plugin
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
DEFAULT_PLUGIN_REPO = ROOT / "ccik_qp"

from qp_ccik_common import rebuild_ccik_plugin, resolve_installed_ccik_plugin_dir, sync_workspace_ccik_plugin


@dataclass(frozen=True)
class QPToolchain:
    qp: str
    qp_plugins: str
    qp_root: str | None


def plugin_repo_uri(path: Path) -> str:
    return path.resolve().as_uri()


def discover_qp_toolchain() -> QPToolchain:
    qp = shutil.which("qp")
    qp_plugins = shutil.which("qp_plugins")
    if not qp or not qp_plugins:
        missing = []
        if not qp:
            missing.append("qp")
        if not qp_plugins:
            missing.append("qp_plugins")
        raise RuntimeError(
            "Quantum Package toolchain not found in PATH. Missing: "
            + ", ".join(missing)
            + ". Install/source QP first."
        )
    qp_root = os.environ.get("QP_ROOT")
    return QPToolchain(qp=qp, qp_plugins=qp_plugins, qp_root=qp_root)


def build_install_commands(toolchain: QPToolchain, plugin_repo: Path) -> list[list[str]]:
    uri = plugin_repo_uri(plugin_repo)
    return [
        [toolchain.qp_plugins, "download", uri],
        [toolchain.qp_plugins, "install", "ccik"],
    ]


def build_set_param_commands(toolchain: QPToolchain, ezfio: str, params: dict[str, str]) -> list[list[str]]:
    cmds: list[list[str]] = [[toolchain.qp, "set_file", ezfio]]
    for key, value in params.items():
        cmds.append([toolchain.qp, "set", "ccik", key, str(value)])
    return cmds


def resolve_ccik_binary(toolchain: QPToolchain) -> str:
    if not toolchain.qp_root:
        raise RuntimeError("QP_ROOT is not set. Source quantum_package.rc first.")
    binary = Path(toolchain.qp_root) / "plugins" / "ccik_qp" / "stable" / "ccik" / "ccik"
    return str(binary)


def build_run_commands(toolchain: QPToolchain, ezfio: str, *, mode: str = "direct") -> list[list[str]]:
    cmds: list[list[str]] = [[toolchain.qp, "set_file", ezfio]]
    if mode == "direct":
        cmds.append([resolve_ccik_binary(toolchain), ezfio])
        return cmds
    if mode == "qp-run":
        cmds.append([toolchain.qp, "run", "ccik", ezfio])
        return cmds
    raise ValueError(f"Unsupported run mode: {mode}")


def run_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    for cmd in commands:
        print("$ " + " ".join(shlex.quote(part) for part in cmd))
        if dry_run:
            continue
        subprocess.run(cmd, check=True)


def _params_from_args(args: argparse.Namespace) -> dict[str, str]:
    params: dict[str, str] = {}
    for name in (
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
    ):
        val = getattr(args, name, None)
        if val is not None:
            params[name] = str(val)
    return params


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge for the Quantum Package ccik plugin")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Check whether qp and qp_plugins are available")
    p_status.add_argument("--dry-run", action="store_true", help="Ignored for status")

    p_install = sub.add_parser("install-plugin", help="Register and install the ccik plugin")
    p_install.add_argument("--repo", default=str(DEFAULT_PLUGIN_REPO), help="Path to the ccik_qp plugin repository")
    p_install.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    p_sync = sub.add_parser("sync-plugin", help="Sync workspace plugin sources into the installed QP plugin tree")
    p_sync.add_argument("--repo", default=str(DEFAULT_PLUGIN_REPO / "stable" / "ccik"), help="Workspace plugin source directory")
    p_sync.add_argument("--dry-run", action="store_true", help="Print what would be synced without executing")

    p_rebuild = sub.add_parser("rebuild-plugin", help="Sync and rebuild the installed QP plugin with /bin/bash")
    p_rebuild.add_argument("--repo", default=str(DEFAULT_PLUGIN_REPO / "stable" / "ccik"), help="Workspace plugin source directory")
    p_rebuild.add_argument("--skip-sync", action="store_true", help="Rebuild the installed plugin tree without syncing workspace sources first")
    p_rebuild.add_argument("--dry-run", action="store_true", help="Print the resolved installed plugin directory and rebuild action without executing")

    p_set = sub.add_parser("set-params", help="Set EZFIO parameters for the ccik plugin")
    p_set.add_argument("ezfio", help="Target EZFIO directory")
    p_set.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p_set.add_argument("--ccik-basis-size", dest="ccik_basis_size", type=int, default=None)
    p_set.add_argument("--ccik-generator-keep", dest="ccik_generator_keep", type=int, default=None)
    p_set.add_argument("--ccik-generator-fraction", dest="ccik_generator_fraction", type=float, default=None)
    p_set.add_argument("--ccik-generator-abs-coeff-cutoff", dest="ccik_generator_abs_coeff_cutoff", type=float, default=None)
    p_set.add_argument("--ccik-max-parents-for-expansion", dest="ccik_max_parents_for_expansion", type=int, default=None)
    p_set.add_argument("--ccik-include-current-support", dest="ccik_include_current_support", choices=("true", "false"), default=None)
    p_set.add_argument("--ccik-selector-keep", dest="ccik_selector_keep", type=int, default=None)
    p_set.add_argument("--ccik-external-keep", dest="ccik_external_keep", type=int, default=None)
    p_set.add_argument("--ccik-external-candidate-factor", dest="ccik_external_candidate_factor", type=int, default=None)
    p_set.add_argument("--ccik-pt2-rank-weight", dest="ccik_pt2_rank_weight", type=float, default=None)
    p_set.add_argument("--ccik-selector-mode", dest="ccik_selector_mode", default=None)
    p_set.add_argument("--ccik-matvec-backend", dest="ccik_matvec_backend", default=None)
    p_set.add_argument("--ccik-compression-policy", dest="ccik_compression_policy", choices=("topk", "threshold", "fixed_budget"), default=None)
    p_set.add_argument("--ccik-store-abs-tol", dest="ccik_store_abs_tol", type=float, default=None)
    p_set.add_argument("--ccik-propagate-abs-tol", dest="ccik_propagate_abs_tol", type=float, default=None)
    p_set.add_argument("--ccik-nkeep", dest="ccik_nkeep", type=int, default=None)
    p_set.add_argument("--ccik-tmp-keep-factor", dest="ccik_tmp_keep_factor", type=int, default=None)
    p_set.add_argument("--ccik-preserve-prev-support", dest="ccik_preserve_prev_support", choices=("true", "false"), default=None)
    p_set.add_argument("--ccik-preserve-prev-support-cap", dest="ccik_preserve_prev_support_cap", type=int, default=None)
    p_set.add_argument("--ccik-total-det-budget", dest="ccik_total_det_budget", type=int, default=None)
    p_set.add_argument("--ccik-conv-tol", dest="ccik_conv_tol", type=float, default=None)
    p_set.add_argument("--ccik-ritz-overlap-tol", dest="ccik_ritz_overlap_tol", type=float, default=None)
    p_set.add_argument("--ccik-ritz-max-drop", dest="ccik_ritz_max_drop", type=float, default=None)
    p_set.add_argument("--ccik-save-final-wf", dest="ccik_save_final_wf", choices=("true", "false"), default=None)
    p_set.add_argument("--ccik-use-existing-wf", dest="ccik_use_existing_wf", choices=("true", "false"), default=None)
    p_set.add_argument("--ccik-do-post-pt2", dest="ccik_do_post_pt2", choices=("true", "false"), default=None)
    p_set.add_argument("--ccik-max-cycle", dest="ccik_max_cycle", type=int, default=None)
    p_set.add_argument("--ccik-restart-m-cycle", dest="ccik_restart_m_cycle", type=int, default=None)
    p_set.add_argument("--ccik-restart-nroot", dest="ccik_restart_nroot", type=int, default=None)
    p_set.add_argument("--ccik-verbose", dest="ccik_verbose", choices=("true", "false"), default=None)

    p_run = sub.add_parser("run", help="Run the ccik plugin on an EZFIO directory")
    p_run.add_argument("ezfio", help="Target EZFIO directory")
    p_run.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p_run.add_argument(
        "--mode",
        choices=("direct", "qp-run"),
        default="direct",
        help="Execution mode. `direct` uses the compiled plugin binary and is the default.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    try:
        toolchain = discover_qp_toolchain()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "status":
        print(f"qp={toolchain.qp}")
        print(f"qp_plugins={toolchain.qp_plugins}")
        print(f"qp_root={toolchain.qp_root}")
        return 0

    if args.command == "install-plugin":
        repo = Path(args.repo)
        if not repo.exists():
            print(f"Plugin repository not found: {repo}", file=sys.stderr)
            return 2
        run_commands(build_install_commands(toolchain, repo), dry_run=bool(args.dry_run))
        return 0

    if args.command == "sync-plugin":
        if not toolchain.qp_root:
            print("QP_ROOT is not set. Source quantum_package.rc first.", file=sys.stderr)
            return 2
        repo = Path(args.repo)
        if bool(args.dry_run):
            print(f"workspace_plugin_dir={repo.resolve()}")
            print(f"installed_plugin_dir={resolve_installed_ccik_plugin_dir(Path(toolchain.qp_root))}")
            return 0
        summary = sync_workspace_ccik_plugin(Path(toolchain.qp_root), workspace_plugin_dir=repo)
        print(summary)
        return 0

    if args.command == "rebuild-plugin":
        if not toolchain.qp_root:
            print("QP_ROOT is not set. Source quantum_package.rc first.", file=sys.stderr)
            return 2
        if bool(args.dry_run):
            print(f"workspace_plugin_dir={Path(args.repo).resolve()}")
            print(f"installed_plugin_dir={resolve_installed_ccik_plugin_dir(Path(toolchain.qp_root))}")
            print("command=/bin/bash -lc 'source quantum_package.rc && make -C <installed-plugin-dir> veryclean ccik'")
            return 0
        from qp_ccik_common import discover_qp_paths as common_discover_qp_paths

        qp = common_discover_qp_paths()
        payload = rebuild_ccik_plugin(
            qp,
            workspace_plugin_dir=Path(args.repo),
            sync_workspace=not bool(args.skip_sync),
            clean=True,
        )
        print(payload["plugin_dir"])
        return 0

    if args.command == "set-params":
        params = _params_from_args(args)
        if not params:
            print("No parameters requested; nothing to do.")
            return 0
        run_commands(build_set_param_commands(toolchain, args.ezfio, params), dry_run=bool(args.dry_run))
        return 0

    if args.command == "run":
        try:
            cmds = build_run_commands(toolchain, args.ezfio, mode=args.mode)
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        run_commands(cmds, dry_run=bool(args.dry_run))
        return 0

    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
