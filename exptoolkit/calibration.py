"""A/A calibration: does a method behave on the data we actually have?

Split real control-group customers into two fake groups many times. Nobody got
a treatment, so the true effect is exactly zero. A well calibrated method should
then flag a "significant" difference about alpha of the time, and its intervals
should contain zero about 1 - alpha of the time.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Calibration:
    method: str
    metric: str
    n_per_arm: int
    n_sims: int
    false_positive_rate: float
    coverage: float
    alpha: float

    @property
    def fpr_band(self):
        """Range of false positive rates consistent with a perfect method, given n_sims."""
        lo, hi = stats.binom.ppf([0.025, 0.975], self.n_sims, self.alpha) / self.n_sims
        return float(lo), float(hi)

    @property
    def well_calibrated(self) -> bool:
        lo, hi = self.fpr_band
        return lo <= self.false_positive_rate <= hi


def aa_test(values, estimator: Callable, n_per_arm: int, n_sims: int = 1000,
            alpha: float = 0.05, seed: int = 0, method: str = "", metric: str = "",
            **kwargs) -> Calibration:
    """Run n_sims A/A splits of `values` and score the estimator."""
    values = np.asarray(values, float)
    if 2 * n_per_arm > len(values):
        raise ValueError("n_per_arm is too large for the number of control customers")
    rng = np.random.default_rng(seed)
    false_pos = covered = 0
    for _ in range(n_sims):
        idx = rng.choice(len(values), 2 * n_per_arm, replace=False)
        a, b = values[idx[:n_per_arm]], values[idx[n_per_arm:]]
        if estimator.__name__ == "bootstrap_percentile":
            est = estimator(a, b, alpha=alpha, rng=rng, **kwargs)
        else:
            est = estimator(a, b, alpha=alpha, **kwargs)
        false_pos += est.p_value is not None and est.p_value < alpha
        covered += est.covers(0.0)
    return Calibration(method, metric, n_per_arm, n_sims,
                       false_pos / n_sims, covered / n_sims, alpha)


def calibration_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        lo, hi = r.fpr_band
        rows.append({
            "Method": r.method, "Metric": r.metric, "Customers per group": r.n_per_arm,
            "False positive rate": r.false_positive_rate, "Interval coverage": r.coverage,
            "Expected range": f"{lo:.1%} to {hi:.1%}", "Calibrated": r.well_calibrated,
        })
    return pd.DataFrame(rows)
