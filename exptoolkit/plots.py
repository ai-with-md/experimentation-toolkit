"""Charts used across the notebooks, all built on exptoolkit.style."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style as s


def money(v: float, decimals: int = 2) -> str:
    """Format dollars with the sign in front: -$0.05, $1,200."""
    sign = "\u2212" if v < 0 else ""
    return f"{sign}${abs(v):,.{decimals}f}"


def _dot(ax, x, y, color, size=110):
    # Surface-coloured ring keeps dots readable where they overlap lines.
    ax.scatter(x, y, s=size, color=color, edgecolor=s.SURFACE, linewidth=2, zorder=4)


def balance_plot(smd: pd.DataFrame, threshold: float = 0.1, labels: dict = None):
    """Love plot: standardized mean difference per covariate, each email vs control.

    `labels` maps covariate names to readable labels; its order sets the row order.
    """
    if labels:
        smd = smd.loc[[c for c in labels if c in smd.index]]
    covs = list(smd.index)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(covs) + 2.4))
    fig.subplots_adjust(top=0.72, left=0.3, right=0.97, bottom=0.14)

    ax.axvspan(-threshold, threshold, color=s.HIGHLIGHT, alpha=0.07, lw=0)
    for edge in (-threshold, threshold):
        ax.axvline(edge, color=s.HIGHLIGHT, lw=1, alpha=0.5)
    ax.axvline(0, color=s.BASELINE, lw=1)

    y = np.arange(len(covs))
    offsets = {"mens_email": -0.12, "womens_email": 0.12}
    for arm, off in offsets.items():
        if arm in smd.columns:
            _dot(ax, smd[arm].values, y + off, s.ARM_COLORS[arm])
            ax.scatter([], [], s=110, color=s.ARM_COLORS[arm], label=s.ARM_LABELS[arm])

    names = [labels.get(c, c) if labels else c.replace("_", " ") for c in covs]
    ax.set_yticks(y, names)
    lim = max(threshold * 1.6, float(np.abs(smd.values).max()) * 1.3)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Standardized mean difference vs no email")
    ax.text(0, len(covs) - 0.35, f"balanced zone (±{threshold})", color=s.HIGHLIGHT,
            fontsize=10, ha="center", va="center", alpha=0.85)
    ax.set_ylim(len(covs) - 0.1, -0.6)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, handletextpad=0.3,
              columnspacing=1.2)
    return fig, ax


def _mean_ci(x: pd.Series, z: float = 1.96):
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return m, m - z * se, m + z * se


def group_means_plot(df: pd.DataFrame, outcomes: dict, arm_col: str = "arm"):
    """One panel per outcome: each arm's mean with a 95% interval.

    `outcomes` maps column name to (panel title, value formatter).
    """
    arms = [a for a in s.ARM_ORDER if a in df[arm_col].unique()]
    fig, axes = plt.subplots(1, len(outcomes), figsize=(4.1 * len(outcomes), 3.2),
                             sharey=True)
    fig.subplots_adjust(top=0.62, left=0.12, right=0.98, bottom=0.12, wspace=0.12)
    axes = np.atleast_1d(axes)
    y = np.arange(len(arms))

    for ax, (col, (title, fmt)) in zip(axes, outcomes.items()):
        stats = [_mean_ci(df.loc[df[arm_col] == a, col]) for a in arms]
        hi_max = max(st[2] for st in stats)
        for yi, arm, (m, lo, hi) in zip(y, arms, stats):
            c = s.ARM_COLORS[arm]
            ax.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle="round", zorder=3)
            _dot(ax, m, yi, c, size=130)
            ax.text(hi + hi_max * 0.04, yi, fmt(m), va="center", fontsize=11,
                    color=s.INK, fontweight="bold")
        ax.set_xlim(0, hi_max * 1.32)
        ax.set_title(title, loc="left", fontsize=12, color=s.INK_2, pad=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, f=fmt: f(v)))
        ax.locator_params(axis="x", nbins=4)

    axes[0].set_yticks(y, [s.ARM_LABELS[a] for a in arms])
    axes[0].invert_yaxis()
    return fig, axes


def decision_zones_plot(econ, x_max: float = 0.30):
    """Profit per 1,000 emails as a function of revenue lift, with decision zones.

    Drawn from the plan alone, before any result is placed on it.
    """
    be, ww = econ.break_even_lift, econ.worthwhile_lift
    x_min = -0.05
    x = np.linspace(x_min, x_max, 200)
    y = [econ.profit_per_1000(v) for v in x]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.74, left=0.1, right=0.97, bottom=0.14)
    ax.grid(axis="y", color=s.GRID)

    ax.axvspan(x_min, be, color=s.ARM_COLORS["control"], alpha=0.18, lw=0)
    ax.axvspan(be, ww, color=s.ARM_COLORS["control"], alpha=0.08, lw=0)
    ax.axvspan(ww, x_max, color=s.HIGHLIGHT, alpha=0.08, lw=0)
    ax.axhline(0, color=s.BASELINE, lw=1)
    ax.plot(x, y, color=s.INK, lw=2.5, zorder=3)

    for xv, color in [(be, s.INK_2), (ww, s.HIGHLIGHT)]:
        ax.axvline(xv, color=color, lw=1, alpha=0.8)
        _dot(ax, xv, econ.profit_per_1000(xv), color, size=90)

    top = max(y)
    label_y = top * 0.92
    ax.text((x_min + be) / 2, label_y, "Loses\nmoney", ha="center", va="top",
            color=s.INK_2, fontsize=11, fontweight="bold")
    ax.text((be + ww) / 2, label_y, "Pays for itself,\ntoo small\nto act on", ha="center",
            va="top", color=s.INK_2, fontsize=10.5, fontweight="bold")
    ax.text((ww + x_max) / 2, label_y, "Worth sending", ha="center", va="top",
            color=s.HIGHLIGHT, fontsize=13, fontweight="bold")

    ax.annotate(f"break-even  ${be:.3f}", (be, 0), xytext=(6, -22),
                textcoords="offset points", color=s.INK_2, fontsize=10)
    ax.annotate(f"worth it  ${ww:.3f}", (ww, econ.profit_per_1000(ww)), xytext=(8, -18),
                textcoords="offset points", color=s.INK, fontsize=10)

    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: money(v, 2)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: money(v, 0)))
    ax.tick_params(axis="y", colors=s.MUTED)
    ax.set_xlabel("Revenue lift per customer from the email")
    ax.set_ylabel("Profit per 1,000 emails")
    return fig, ax
