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


def calibration_plot(table: pd.DataFrame, metric_titles: dict):
    """False positive rate in A/A tests by sample size, one panel per metric.

    The shaded band is the range a perfect method would land in, given the
    number of simulations.
    """
    metrics = list(metric_titles)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.4), sharey=True)
    fig.subplots_adjust(top=0.70, left=0.08, right=0.9, bottom=0.15, wspace=0.28)
    axes = np.atleast_1d(axes)
    y_max = max(0.1, float(table["False positive rate"].max()) * 1.15)

    for ax, metric in zip(axes, metrics):
        sub = table[table["Metric"] == metric]
        band = sub["Expected range"].iloc[0]
        lo, hi = [float(v.strip().rstrip("%")) / 100 for v in band.split("to")]
        ax.axhspan(lo, hi, color=s.HIGHLIGHT, alpha=0.10, lw=0)
        ax.axhline(0.05, color=s.HIGHLIGHT, lw=1, alpha=0.6)
        for method, grp in sub.groupby("Method"):
            grp = grp.sort_values("Customers per group")
            c = s.METHOD_COLORS.get(method, s.INK)
            ax.plot(grp["Customers per group"], grp["False positive rate"], color=c, lw=2)
            _dot(ax, grp["Customers per group"], grp["False positive rate"], c, size=70)
            last = grp.iloc[-1]
            ax.annotate(method, (last["Customers per group"], last["False positive rate"]),
                        xytext=(10, 0), textcoords="offset points", va="center",
                        color=s.INK_2, fontsize=10)
            ax.scatter([], [], color=c, s=70, label=method)
        ax.set_xscale("log")
        ax.set_xticks(sorted(sub["Customers per group"].unique()))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.tick_params(axis="y", colors=s.MUTED)
        ax.grid(axis="y", color=s.GRID)
        ax.set_title(metric_titles[metric], loc="left", fontsize=12, color=s.INK_2, pad=10)
        ax.set_xlabel("Customers per group")
    axes[0].set_ylabel("False positive rate")
    axes[0].text(axes[0].get_xlim()[0] * 1.05, 0.05, " target 5%", color=s.HIGHLIGHT,
                 fontsize=9.5, va="bottom")
    axes[-1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=2,
                    handletextpad=0.3, columnspacing=1.2)
    return fig, axes


def power_curves_plot(panels: dict, actual_n: int, target: float = 0.8):
    """Power by group size: simulation on real data (dots) vs textbook formula (line).

    `panels` maps a panel title to a dict with keys `sizes`, `simulated` (power
    at each size), `formula` (callable n -> power) and `n_needed` (simulated
    size that reaches the target).
    """
    sim_color = s.METHOD_COLORS["Welch t-test"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), sharey=True)
    fig.subplots_adjust(top=0.70, left=0.08, right=0.97, bottom=0.15, wspace=0.12)
    axes = np.atleast_1d(axes)

    for ax, (title, p) in zip(axes, panels.items()):
        sizes = np.asarray(p["sizes"], float)
        grid = np.geomspace(sizes.min(), sizes.max(), 200)
        ax.axhline(target, color=s.HIGHLIGHT, lw=1, alpha=0.7)
        ax.axvline(actual_n, color=s.ARM_COLORS["control"], lw=1.5)
        ax.plot(grid, p["formula"](grid), color=s.INK_2, lw=1.5, alpha=0.8)
        ax.plot(sizes, p["simulated"], color=sim_color, lw=2)
        _dot(ax, sizes, p["simulated"], sim_color, size=60)

        need = p["n_needed"]
        ax.scatter([need], [target], s=150, color=s.HIGHLIGHT, edgecolor=s.SURFACE,
                   linewidth=2, zorder=5)
        ax.annotate(f"needs {need / 1000:,.0f}K\nper group", (need, target), xytext=(-12, 14),
                    textcoords="offset points", ha="right", color=s.INK, fontsize=10.5,
                    fontweight="bold")
        ax.annotate("this test\n21K per group", (actual_n, 0.02), xytext=(6, 0),
                    textcoords="offset points", color=s.INK_2, fontsize=9.5, va="bottom")

        ax.set_xscale("log")
        ax.set_xticks([10_000, 20_000, 50_000, 100_000, 200_000])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}K"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.tick_params(axis="y", colors=s.MUTED)
        ax.grid(axis="y", color=s.GRID)
        ax.set_title(title, loc="left", fontsize=12, color=s.INK_2, pad=10)
        ax.set_xlabel("Customers per group")

    axes[0].set_ylabel("Chance of detecting the lift")
    axes[-1].plot([], [], color=sim_color, lw=2, marker="o", label="Simulated on real data")
    axes[-1].plot([], [], color=s.INK_2, lw=1.5, label="Textbook formula")
    axes[-1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=2,
                    handletextpad=0.4, columnspacing=1.4)
    return fig, axes


def peeking_plot(naive_bar, sequential_bar, naive_fpr, sequential_fpr, fixed_fpr,
                 alpha: float = 0.05, day_label: str = "Day"):
    """Left: the bar a result must clear on each day. Right: false alarms by day."""
    days = np.arange(1, len(naive_bar) + 1)
    sc = s.STRATEGY_COLORS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(top=0.70, left=0.07, right=0.88, bottom=0.14, wspace=0.42)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=s.GRID)
        ax.set_xticks([1, 4, 7, 10, 14])
        ax.set_xlim(0.5, len(days) + 0.5)
        ax.set_xlabel(day_label)
        ax.tick_params(axis="y", colors=s.MUTED)

    ax1.plot(days, naive_bar, color=sc["naive"], lw=2)
    ax1.plot(days, sequential_bar, color=sc["sequential"], lw=2)
    _dot(ax1, days, sequential_bar, sc["sequential"], size=45)
    ax1.annotate("Peek daily", (days[-1], naive_bar[-1]), xytext=(8, -12),
                 textcoords="offset points", color=s.INK_2, fontsize=10)
    ax1.annotate("Sequential", (days[-1], sequential_bar[-1]), xytext=(8, 6),
                 textcoords="offset points", color=s.INK_2, fontsize=10)
    ax1.set_ylim(0, max(sequential_bar) * 1.1)
    ax1.set_ylabel("Evidence needed (z score)")
    ax1.set_title("The bar to call a winner", loc="left", fontsize=12, color=s.INK_2, pad=10)

    ax2.axhline(alpha, color=s.HIGHLIGHT, lw=1, alpha=0.7)
    ax2.text(0.7, alpha, f" promised {alpha:.0%}", color=s.HIGHLIGHT, fontsize=9.5, va="bottom")
    ax2.plot(days, naive_fpr, color=sc["naive"], lw=2)
    _dot(ax2, days, naive_fpr, sc["naive"], size=45)
    ax2.plot(days, sequential_fpr, color=sc["sequential"], lw=2)
    _dot(ax2, days, sequential_fpr, sc["sequential"], size=45)
    _dot(ax2, days[-1], fixed_fpr, sc["fixed"], size=110)
    ax2.annotate(f"Peek daily  {naive_fpr[-1]:.0%}", (days[-1], naive_fpr[-1]), xytext=(10, 0),
                 textcoords="offset points", va="center", color=s.INK, fontsize=10.5,
                 fontweight="bold")
    ax2.annotate(f"Sequential  {sequential_fpr[-1]:.1%}", (days[-1], sequential_fpr[-1]),
                 xytext=(10, 8), textcoords="offset points", va="center", color=s.INK_2,
                 fontsize=10)
    ax2.annotate(f"Look once  {fixed_fpr:.1%}", (days[-1], fixed_fpr), xytext=(10, -9),
                 textcoords="offset points", va="center", color=s.INK_2, fontsize=10)
    ax2.set_ylim(0, max(naive_fpr) * 1.15)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax2.set_ylabel("Tests with a false winner so far")
    ax2.set_title("False alarms when nothing changed", loc="left", fontsize=12,
                  color=s.INK_2, pad=10)

    for key, label in [("naive", "Peek daily, stop at first win"),
                       ("sequential", "Sequential (O'Brien-Fleming)"),
                       ("fixed", "Look once at the end")]:
        ax2.scatter([], [], color=sc[key], s=60, label=label)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=3, handletextpad=0.3,
               columnspacing=1.2)
    return fig, (ax1, ax2)
