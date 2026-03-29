"""Shared helpers for N2 driver scripts.

These scripts are meant to be runnable directly (without installing the package),
so we keep small shared helpers in the scripts/ folder rather than in src/ccik/.
"""

from __future__ import annotations

import numpy as np


def make_n2_atom(R_ang: float) -> str:
    """Return a PySCF-style atom string for N2 at bond length R_ang (Å)."""

    z = float(R_ang) / 2.0
    return f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """


def safe_log_err(delta: np.ndarray, floor: float = 1e-16) -> np.ndarray:
    """Return max(|delta|, floor) for log plotting."""

    err = np.abs(np.asarray(delta, dtype=float))
    return np.maximum(err, float(floor))
