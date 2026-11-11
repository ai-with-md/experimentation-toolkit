"""Treatment effect estimators that share one interface.

Each estimator takes treatment and control outcomes and returns an Estimate:
the difference in means (treatment minus control), a confidence interval and,
where it makes sense, a p-value.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    effect: float
    ci_low: float
    ci_high: float
    p_value: Optional[float] = None

    def covers(self, value: float = 0.0) -> bool:
        return self.ci_low <= value <= self.ci_high


def welch_t(y_t, y_c, alpha: float = 0.05) -> Estimate:
    """Difference in means with a Welch t interval (unequal variances)."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    n_t, n_c = len(y_t), len(y_c)
    v_t, v_c = y_t.var(ddof=1) / n_t, y_c.var(ddof=1) / n_c
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(v_t + v_c)
    if se == 0:
        return Estimate(diff, diff, diff, 1.0)
    dof = (v_t + v_c) ** 2 / (v_t ** 2 / (n_t - 1) + v_c ** 2 / (n_c - 1))
    q = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(diff) / se, dof)
    return Estimate(diff, diff - q * se, diff + q * se, float(p))


def bootstrap_percentile(y_t, y_c, alpha: float = 0.05, n_boot: int = 1000,
                         rng: Optional[np.random.Generator] = None) -> Estimate:
    """Difference in means with a percentile bootstrap interval.

    The p-value is the two-sided share of bootstrap draws on the other side of zero.
    """
    rng = rng or np.random.default_rng()
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    bt = y_t[rng.integers(0, len(y_t), (n_boot, len(y_t)))].mean(axis=1)
    bc = y_c[rng.integers(0, len(y_c), (n_boot, len(y_c)))].mean(axis=1)
    draws = bt - bc
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return Estimate(float(y_t.mean() - y_c.mean()), float(lo), float(hi), float(p))


ESTIMATORS = {
    "Welch t-test": welch_t,
    "Bootstrap": bootstrap_percentile,
}
