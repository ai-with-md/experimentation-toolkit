"""Causal effects without randomization: difference-in-differences and synthetic controls.

Both methods build a stand-in for what would have happened to the treated unit
without the policy. Difference-in-differences uses the average of the other
units and assumes the gap would have stayed constant. Synthetic control builds
a weighted blend of other units that tracks the treated unit before the policy.

With one treated unit, the usual standard errors don't apply. Instead every
method is re-run with each untreated unit pretending to be treated (placebos),
and the real result is judged against that spread.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

DEFAULT_SMOKING = Path(__file__).resolve().parent.parent / "data" / "smoking.csv"


def load_smoking(path=DEFAULT_SMOKING) -> pd.DataFrame:
    """Cigarette packs sold per person, by state and year, as a state-by-year table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df.pivot(index="year", columns="state", values="cigsale")


def did(panel: pd.DataFrame, treated: str, start: int) -> float:
    """Classic difference-in-differences: change in the treated unit minus change in the rest."""
    others = panel.drop(columns=treated).mean(axis=1)
    pre, post = panel.index < start, panel.index >= start
    return float((panel.loc[post, treated].mean() - panel.loc[pre, treated].mean())
                 - (others[post].mean() - others[pre].mean()))


def event_study(panel: pd.DataFrame, treated: str, reference: int) -> pd.Series:
    """Gap between the treated unit and the average of the rest, each year, relative to `reference`.

    Before the policy these should hover around zero if the two would have moved
    in parallel. A steady drift before the policy means they wouldn't have.
    """
    gap = panel[treated] - panel.drop(columns=treated).mean(axis=1)
    return gap - gap.loc[reference]


def placebo_band(panel: pd.DataFrame, treated: str, statistic, q=(0.05, 0.95)) -> pd.DataFrame:
    """Run `statistic(panel_without_treated, unit)` with every untreated unit as a fake treated unit.

    Returns the lower and upper quantiles across placebos, per index value.
    """
    donors = panel.drop(columns=treated)
    runs = pd.concat({unit: statistic(donors, unit) for unit in donors.columns}, axis=1)
    return pd.DataFrame({"low": runs.quantile(q[0], axis=1), "high": runs.quantile(q[1], axis=1)})


def synth_weights(treated_pre: np.ndarray, donors_pre: np.ndarray, penalty: float = 1e4) -> np.ndarray:
    """Non-negative donor weights that sum to one and best match the treated unit before the policy.

    Solved as non-negative least squares, with the sum-to-one condition added as
    one heavily weighted extra row.
    """
    scale = np.abs(treated_pre).mean() or 1.0
    A = np.vstack([donors_pre / scale, penalty * np.ones(donors_pre.shape[1])])
    b = np.concatenate([treated_pre / scale, [penalty]])
    # Some numpy builds on macOS raise spurious matmul warnings inside nnls; the result is checked below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w, _ = nnls(A, b, maxiter=10_000)
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise RuntimeError("Synthetic control weights could not be computed.")
    return w / w.sum()


@dataclass
class SynthResult:
    unit: str
    weights: pd.Series
    actual: pd.Series
    synthetic: pd.Series
    start: int

    @property
    def gap(self) -> pd.Series:
        return self.actual - self.synthetic

    @property
    def pre_rmspe(self) -> float:
        g = self.gap[self.gap.index < self.start]
        return float(np.sqrt((g ** 2).mean()))

    @property
    def post_rmspe(self) -> float:
        g = self.gap[self.gap.index >= self.start]
        return float(np.sqrt((g ** 2).mean()))

    @property
    def ratio(self) -> float:
        """Post-policy misfit relative to pre-policy misfit. Large means a real break."""
        return self.post_rmspe / self.pre_rmspe

    @property
    def average_effect(self) -> float:
        return float(self.gap[self.gap.index >= self.start].mean())


def synthetic_control(panel: pd.DataFrame, treated: str, start: int) -> SynthResult:
    """Fit a synthetic version of `treated` from the other columns, using years before `start`."""
    donors = panel.drop(columns=treated)
    pre = panel.index < start
    w = synth_weights(panel.loc[pre, treated].to_numpy(), donors.loc[pre].to_numpy())
    weights = pd.Series(w, index=donors.columns)
    synthetic = pd.Series(np.einsum("ij,j->i", donors.to_numpy(), w), index=panel.index)
    return SynthResult(treated, weights, panel[treated], synthetic, start)


def synth_placebos(panel: pd.DataFrame, treated: str, start: int) -> dict:
    """Synthetic control for every untreated unit, using the other untreated units as donors."""
    donors = panel.drop(columns=treated)
    return {unit: synthetic_control(donors, unit, start) for unit in donors.columns}
