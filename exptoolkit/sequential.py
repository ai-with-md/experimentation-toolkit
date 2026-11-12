"""Peeking and sequential designs.

A test that is checked every day and stopped at the first "significant" result
raises far more false alarms than its p-value promises. A group sequential
design fixes this by setting a stricter bar for early looks and relaxing it as
data accumulates (O'Brien-Fleming shape). Here the bar is found by simulation,
then checked on real data with A/A tests.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


def obf_boundaries(n_looks: int, alpha: float = 0.05, n_paths: int = 200_000,
                   seed: int = 0) -> np.ndarray:
    """Two-sided O'Brien-Fleming style z thresholds for equally spaced looks.

    The threshold at look k is c / sqrt(k / n_looks). The constant c is found by
    simulating the test statistic under no effect, so that the chance of
    crossing at any look is alpha.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_looks + 1) / n_looks
    steps = rng.standard_normal((n_paths, n_looks)) * np.sqrt(1 / n_looks)
    z = np.abs(np.cumsum(steps, axis=1) / np.sqrt(t))
    # Path crosses c / sqrt(t) somewhere iff c < max_k z_k * sqrt(t_k).
    c = np.quantile((z * np.sqrt(t)).max(axis=1), 1 - alpha)
    return c / np.sqrt(t)


def naive_boundaries(n_looks: int, alpha: float = 0.05) -> np.ndarray:
    """The same fixed threshold at every look (what daily peeking does)."""
    return np.full(n_looks, stats.norm.ppf(1 - alpha / 2))


def _cumulative_z(a: np.ndarray, b: np.ndarray, n_looks: int) -> np.ndarray:
    """Welch z statistic (b minus a) using the first k/n_looks of each group, per look."""
    n = len(a)
    ends = (np.arange(1, n_looks + 1) * n) // n_looks
    out = np.empty(n_looks)
    sa, sb = np.cumsum(a), np.cumsum(b)
    qa, qb = np.cumsum(a * a), np.cumsum(b * b)
    for k, m in enumerate(ends):
        ma, mb = sa[m - 1] / m, sb[m - 1] / m
        va = (qa[m - 1] - m * ma * ma) / (m - 1)
        vb = (qb[m - 1] - m * mb * mb) / (m - 1)
        se = np.sqrt(va / m + vb / m)
        out[k] = 0.0 if se == 0 else (mb - ma) / se
    return out


@dataclass
class SequentialResult:
    crossed_by_look: np.ndarray   # share of tests that have stopped by each look
    stop_look: np.ndarray         # look at which each test stopped (0 = never)

    @property
    def rejection_rate(self) -> float:
        return float(self.crossed_by_look[-1])

    @property
    def average_stop(self) -> float:
        """Average look at which the test ended, counting non-stops as the last look."""
        n_looks = len(self.crossed_by_look)
        return float(np.where(self.stop_look > 0, self.stop_look, n_looks).mean())


def run_sequential(spend, boundaries: np.ndarray, n_per_arm: int, n_sims: int = 2000,
                   revenue_lift: float = 0.0, seed: int = 0,
                   rng: Optional[np.random.Generator] = None) -> SequentialResult:
    """Simulate tests that look at the data len(boundaries) times and stop at the first crossing.

    Both groups are resampled from real control spend. With `revenue_lift` > 0,
    extra non-buyers in group B become buyers, as in exptoolkit.power.
    """
    rng = rng or np.random.default_rng(seed)
    spend = np.asarray(spend, float)
    buyer_spend = spend[spend > 0]
    flip = revenue_lift / buyer_spend.mean() / (1 - (spend > 0).mean())
    n_looks = len(boundaries)
    stop = np.zeros(n_sims, int)
    for i in range(n_sims):
        a = spend[rng.integers(0, len(spend), n_per_arm)]
        b = spend[rng.integers(0, len(spend), n_per_arm)]
        if revenue_lift > 0:
            new = (b == 0) & (rng.random(n_per_arm) < flip)
            b[new] = buyer_spend[rng.integers(0, len(buyer_spend), new.sum())]
        z = np.abs(_cumulative_z(a, b, n_looks))
        crossed = np.nonzero(z > boundaries)[0]
        if len(crossed):
            stop[i] = crossed[0] + 1
    by_look = np.array([(stop > 0) & (stop <= k) for k in range(1, n_looks + 1)]).mean(axis=1)
    return SequentialResult(by_look, stop)
