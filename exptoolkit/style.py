"""Shared visual style so every chart in the toolkit looks like one family.

Dark surface, one colour per experiment arm that never changes, an acid lime
highlight reserved for the thing the reader should look at, and titles that
state the finding instead of naming the chart.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#111116"
INK = "#F4F1EA"
INK_2 = "#B9B5C2"
MUTED = "#85818F"
GRID = "#24232B"
BASELINE = "#3A3944"
HIGHLIGHT = "#D4FF3F"

# Fixed colour per arm. Control is a quiet grey on purpose: it is the baseline.
ARM_COLORS = {
    "control": "#55535F",
    "mens_email": "#7B68FF",
    "womens_email": "#F0563F",
}
ARM_LABELS = {
    "control": "No email",
    "mens_email": "Men's email",
    "womens_email": "Women's email",
}
ARM_ORDER = ["control", "womens_email", "mens_email"]

# Notebooks 10 and 11: the treated state, its comparisons, and placebo states.
STATE_COLORS = {
    "treated": HIGHLIGHT,
    "average": "#B9B5C2",
    "synthetic": "#3B8FE0",
    "placebo": "#3A3944",
}

# Targeting approaches in notebooks 08 and 09.
MODEL_COLORS = {
    "T-learner": "#3B8FE0",
    "X-learner": "#C4831C",
    "Past spend rule": "#F4F1EA",
    "Random": "#55535F",
}

# Ways of running a test (stopping rules) in notebook 05.
STRATEGY_COLORS = {
    "naive": "#C4831C",
    "sequential": "#3B8FE0",
    "fixed": "#55535F",
}

# Methods get their own colours so they never look like an experiment arm.
METHOD_COLORS = {
    "Welch t-test": "#3B8FE0",
    "Bootstrap": "#C4831C",
}

FONT_STACK = ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"]
FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"


def use_style() -> None:
    """Apply the toolkit style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "ytick.major.size": 0,
        "xtick.major.size": 0,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.solid_capstyle": "round",
        "text.parse_math": False,  # so dollar signs in labels print as dollars
    })


def headline(fig, title: str, subtitle: str = "") -> None:
    """Big finding-first title with a quieter subtitle, left aligned."""
    from matplotlib.transforms import offset_copy

    fig.text(0.02, 0.98, title, ha="left", va="top", fontsize=17,
             fontweight="bold", color=INK)
    if subtitle:
        below = offset_copy(fig.transFigure, fig=fig, y=-30, units="points")
        fig.text(0.02, 0.98, subtitle, ha="left", va="top", fontsize=11,
                 color=MUTED, transform=below)


def save(fig, name: str) -> None:
    """Save a figure to docs/figures for use in the README."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.3)
