"""Read configuration files and convert them into solver parameter objects.

This module supports the config-driven scan and benchmark scripts. New learners
can safely ignore it at first and use the example scripts directly.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .params import CCIKParams, CCIKStochasticParams, CCIKThickRestartParams
from .pyscf_cas import CASSpec


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file into a plain Python dictionary."""

    import tomllib

    p = Path(path)
    with p.open("rb") as f:
        return tomllib.load(f)


def load_default_toml() -> dict[str, Any]:
    """Load the packaged default configuration."""

    import tomllib
    from importlib import resources

    data = resources.files("ccik.defaults").joinpath("config.toml")
    with data.open("rb") as f:
        return tomllib.load(f)


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load an explicit config file or the packaged default."""

    if path is None:
        return load_default_toml()
    return load_toml(path)


def ccik_params_from_dict(d: dict[str, Any]) -> CCIKParams:
    """Create ``CCIKParams`` from a parsed TOML section."""

    base = asdict(CCIKParams())
    for key in ("m", "nadd", "nkeep", "Kv", "generator_keep", "tmp_keep_factor", "fixed_store_keep", "fixed_propagate_keep"):
        if key in d:
            base[key] = int(d[key])
    if "max_parents_for_expansion" in d:
        base["max_parents_for_expansion"] = None if d["max_parents_for_expansion"] is None else int(d["max_parents_for_expansion"])
    if "preserve_prev_support_cap" in d:
        base["preserve_prev_support_cap"] = None if d["preserve_prev_support_cap"] is None else int(d["preserve_prev_support_cap"])
    if "verbose" in d:
        base["verbose"] = bool(d["verbose"])
    for key in ("include_current_support", "preserve_prev_support", "exact_hq_diagnostics", "secondortho"):
        if key in d:
            base[key] = bool(d[key])
    if "orth_tol" in d:
        base["orth_tol"] = float(d["orth_tol"])
    if "candidate_mode" in d:
        base["candidate_mode"] = str(d["candidate_mode"]).strip()
    if "projected_ritz_mode" in d:
        base["projected_ritz_mode"] = str(d["projected_ritz_mode"]).strip()
    if "compression_policy" in d:
        base["compression_policy"] = str(d["compression_policy"]).strip()
    if "selector_mode" in d:
        base["selector_mode"] = str(d["selector_mode"]).strip()
    if "selector_depth" in d:
        base["selector_depth"] = int(d["selector_depth"])
    for key in ("selector_ci_coeff_cutoff", "selector_cutoff", "generator_abs_coeff_cutoff", "store_abs_tol", "propagate_abs_tol"):
        if key in d:
            base[key] = float(d[key])
    return CCIKParams(**base)


def ccik_thick_restart_params_from_dict(d: dict[str, Any]) -> CCIKThickRestartParams:
    """Create ``CCIKThickRestartParams`` from a parsed TOML section."""

    base = asdict(CCIKThickRestartParams())
    for key in (
        "m_cycle",
        "ncycles",
        "nroot",
        "nadd",
        "nkeep",
        "Kv",
        "generator_keep",
        "tmp_keep_factor",
        "fixed_store_keep",
        "fixed_propagate_keep",
    ):
        if key in d:
            base[key] = int(d[key])
    if "exactized_checkpoint_every" in d:
        base["exactized_checkpoint_every"] = (
            None if d["exactized_checkpoint_every"] is None else int(d["exactized_checkpoint_every"])
        )
    if "tol" in d:
        base["tol"] = float(d["tol"])
    if "max_parents_for_expansion" in d:
        base["max_parents_for_expansion"] = None if d["max_parents_for_expansion"] is None else int(d["max_parents_for_expansion"])
    if "preserve_prev_support_cap" in d:
        base["preserve_prev_support_cap"] = None if d["preserve_prev_support_cap"] is None else int(d["preserve_prev_support_cap"])
    if "verbose" in d:
        base["verbose"] = bool(d["verbose"])
    for key in ("include_current_support", "preserve_prev_support", "exact_hq_diagnostics", "secondortho"):
        if key in d:
            base[key] = bool(d[key])
    if "orth_tol" in d:
        base["orth_tol"] = float(d["orth_tol"])
    if "candidate_mode" in d:
        base["candidate_mode"] = str(d["candidate_mode"]).strip()
    if "projected_ritz_mode" in d:
        base["projected_ritz_mode"] = str(d["projected_ritz_mode"]).strip()
    if "compression_policy" in d:
        base["compression_policy"] = str(d["compression_policy"]).strip()
    if "selector_mode" in d:
        base["selector_mode"] = str(d["selector_mode"]).strip()
    if "selector_depth" in d:
        base["selector_depth"] = int(d["selector_depth"])
    for key in ("selector_ci_coeff_cutoff", "selector_cutoff", "generator_abs_coeff_cutoff", "store_abs_tol", "propagate_abs_tol"):
        if key in d:
            base[key] = float(d[key])
    return CCIKThickRestartParams(**base)


def ccik_stochastic_params_from_dict(d: dict[str, Any]) -> CCIKStochasticParams:
    """Create ``CCIKStochasticParams`` from a parsed TOML section."""

    base = asdict(CCIKStochasticParams())
    for key in ("m", "nadd", "nkeep", "Kv", "n_walkers", "generator_keep", "tmp_keep_factor", "fixed_store_keep", "fixed_propagate_keep"):
        if key in d:
            base[key] = int(d[key])
    if "seed" in d:
        base["seed"] = None if d["seed"] is None else int(d["seed"])
    if "max_parents_for_expansion" in d:
        base["max_parents_for_expansion"] = None if d["max_parents_for_expansion"] is None else int(d["max_parents_for_expansion"])
    if "preserve_prev_support_cap" in d:
        base["preserve_prev_support_cap"] = None if d["preserve_prev_support_cap"] is None else int(d["preserve_prev_support_cap"])
    for key in ("parent_power", "p_double", "mixed_double_weight", "eps_denom", "generator_abs_coeff_cutoff"):
        if key in d:
            base[key] = float(d[key])
    if "verbose" in d:
        base["verbose"] = bool(d["verbose"])
    for key in ("include_current_support", "preserve_prev_support", "exact_hq_diagnostics", "secondortho"):
        if key in d:
            base[key] = bool(d[key])
    if "orth_tol" in d:
        base["orth_tol"] = float(d["orth_tol"])
    if "candidate_mode" in d:
        base["candidate_mode"] = str(d["candidate_mode"]).strip()
    if "compression_policy" in d:
        base["compression_policy"] = str(d["compression_policy"]).strip()
    if "selector_mode" in d:
        base["selector_mode"] = str(d["selector_mode"]).strip()
    if "selector_depth" in d:
        base["selector_depth"] = int(d["selector_depth"])
    for key in ("selector_ci_coeff_cutoff", "selector_cutoff", "store_abs_tol", "propagate_abs_tol"):
        if key in d:
            base[key] = float(d[key])
    return CCIKStochasticParams(**base)


def run_method_from_dict(d: dict[str, Any]) -> str:
    """Return the requested method name, defaulting to ``ccik_thick``."""

    return str(d.get("method", "ccik_thick")).strip()


def run_methods_from_dict(d: dict[str, Any]) -> list[str]:
    """Return the requested method list.

    Supported shapes:
    - ``method = "ccik_thick"``
    - ``methods = ["ccik", "ccik_thick"]``
    """

    if "methods" in d and d["methods"] is not None:
        raw = d["methods"]
        if not isinstance(raw, (list, tuple)):
            raise TypeError("run.methods must be a list of strings")
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else ["ccik_thick"]

    if "method" in d and isinstance(d["method"], (list, tuple)):
        raw = d["method"]
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else ["ccik_thick"]

    return [run_method_from_dict(d)]


def cas_spec_from_dict(d: dict[str, Any]) -> CASSpec:
    """Create ``CASSpec`` from a parsed TOML section."""

    return CASSpec(
        ncas=int(_require(d, "ncas")),
        nelecas=int(_require(d, "nelecas")),
        ncore=int(d.get("ncore", 0)),
    )


def as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion for dataclasses to plain dictionaries."""

    try:
        return asdict(obj)
    except Exception:
        return dict(obj)
