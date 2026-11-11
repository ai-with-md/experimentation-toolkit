"""Charts used across the notebooks, all built on exptoolkit.style."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style as s


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
