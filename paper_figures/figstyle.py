"""House style shared by all manuscript figures (final print size, REVTeX aip reprint).

Figures are drawn at the size they are printed (\\textwidth = 510 pt = 7.06 in,
\\columnwidth = 246 pt = 3.40 in) so that every label is 8-9 pt on the page.
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D

TEXTWIDTH = 510 / 72.27   # inches
COLUMNWIDTH = 246 / 72.27

# One colour/marker per method everywhere (Okabe-Ito palette).
COLORS = {
    "exact": "#222222",
    "qrte_fixed": "#0072B2",   # QRTE-T1
    "qrte_mf": "#009E73",      # QRTE-T2
    "qkud": "#D55E00",         # QKUD-T1
    "adapt": "#8C8C8C",
    "star": "#F0E442",
    "guide": "#777777",
}
MARKERS = {"exact": "o", "qrte_fixed": "s", "qrte_mf": "D", "qkud": "^", "adapt": "o"}
LABELS = {
    "exact": "Exact exponential QRTE",
    "qrte_fixed": r"QRTE-T1, $\delta t = 0.05$ a.u.",
    "qrte_mf": r"QRTE-T2, $\delta t = \pi/W_{\mathrm{MF}}$",
    "qkud": r"QKUD-T1, $\epsilon = 0.05$ a.u.",
    "adapt": "ADAPT-VQE",
}
SHORT_LABELS = {"exact": "Exact exponential QRTE", "qrte_fixed": "QRTE-T1", "qrte_mf": "QRTE-T2", "qkud": "QKUD-T1"}
YLABEL = r"$N_{\mathrm{iter}}$ to $E_{\mathrm{chem}}$ (1.6 mHa)"
MS = 5.0          # marker size for line plots (points)
SCATTER_S = 30    # scatter marker area (points^2), ~ms 5.5
N_STEPS = 100

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handletextpad": 0.6,
    "legend.labelspacing": 0.45,
    "legend.borderaxespad": 0.4,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": MS,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
}


def apply():
    mpl.rcParams.update(RC)


def grid(ax):
    ax.grid(which="major", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", axis="y", color="#eeeeee", linewidth=0.45, alpha=0.6)
    ax.set_axisbelow(True)


def panel_label(ax, letter, title=None, x=0.0, y=1.02):
    text = f"({letter})" + (f"  {title}" if title else "")
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold" if title is None else "normal")


def no_hit_arrow(ax, x, limit, color, yscale="log"):
    """Mark a run that stopped at `limit` iterations without reaching chemical accuracy."""
    top = limit * 1.9 if yscale == "log" else limit + 0.1 * limit
    ax.annotate("", xy=(x, top), xytext=(x, limit),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9, "mutation_scale": 8,
                            "shrinkA": 3.0, "shrinkB": 0}, zorder=7)


NOT_CONVERGED_LABEL = f"Could not find solution in {N_STEPS} iterations"


def no_hit_handle(label=NOT_CONVERGED_LABEL):
    return Line2D([], [], color=COLORS["guide"], marker=r"$\uparrow$", ls="none", ms=7, label=label)


def series_handle(key, label=None, open_marker=False, ls="none", lw=1.2):
    color = COLORS[key]
    return Line2D([], [], color=color, marker=MARKERS[key], ls=ls, lw=lw, ms=MS,
                  mfc="white" if open_marker else color, mec=color, mew=0.9 if open_marker else 0.5,
                  label=label if label is not None else LABELS[key])


def fit_semilog(x, y):
    """Least-squares fit of log(y) vs x; returns x_fit, y_fit, R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    log_y = np.log(y)
    slope, intercept = np.polyfit(x, log_y, 1)
    x_fit = np.linspace(float(x.min()), float(x.max()), 200)
    predicted = slope * x + intercept
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return x_fit, np.exp(slope * x_fit + intercept), float(r2)


def save(fig, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    return prefix
