"""How big does a test need to be? Textbook formulas and simulation on real data.

The simulation injects an effect the way the email actually worked in this
test: it turns some non-buyers into buyers (spending like real buyers), rather
than nudging everyone's spend up a little.
"""

from typing import Optional

import numpy as np
from scipy import stats

from .estimators import welch_t


def _z(alpha: float, power: float) -> float:
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def required_n_means(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers per group to detect a difference in means of `delta` (equal groups)."""
    return int(np.ceil(2 * (_z(alpha, power) * sd / delta) ** 2))


def required_n_proportions(p: float, delta: float, alpha: float = 0.05,
                           power: float = 0.8) -> int:
    """Customers per group to detect a rise from rate p to p + delta."""
    p2 = p + delta
    var = p * (1 - p) + p2 * (1 - p2)
    return int(np.ceil(_z(alpha, power) ** 2 * var / delta ** 2))


def mde_means(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference in means detectable with n customers per group."""
    return float(_z(alpha, power) * sd * np.sqrt(2 / n_per_arm))


def power_means(sd: float, delta: float, n_per_arm, alpha: float = 0.05):
    """Textbook power to detect a difference in means of `delta`."""
    se = sd * np.sqrt(2 / np.asarray(n_per_arm, float))
    return stats.norm.cdf(delta / se - stats.norm.ppf(1 - alpha / 2))


def power_proportions(p: float, delta: float, n_per_arm, alpha: float = 0.05):
    """Textbook power to detect a rise from rate p to p + delta."""
    p2 = p + delta
    se = np.sqrt((p * (1 - p) + p2 * (1 - p2)) / np.asarray(n_per_arm, float))
    return stats.norm.cdf(delta / se - stats.norm.ppf(1 - alpha / 2))


def n_for_power(sizes, powers, target: float = 0.8) -> float:
    """Interpolate the group size where simulated power first reaches `target`."""
    sizes, powers = np.asarray(sizes, float), np.asarray(powers, float)
    above = np.nonzero(powers >= target)[0]
    if len(above) == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(sizes[0])
    x0, x1, y0, y1 = sizes[i - 1], sizes[i], powers[i - 1], powers[i]
    return float(np.exp(np.log(x0) + (target - y0) / (y1 - y0) * (np.log(x1) - np.log(x0))))


def simulate_power(spend, revenue_lift: float, n_per_arm: int, metric: str = "spend",
                   n_sims: int = 500, alpha: float = 0.05, seed: int = 0,
                   rng: Optional[np.random.Generator] = None) -> float:
    """Share of simulated tests that detect a true revenue lift of `revenue_lift` per customer.

    `spend` holds real control-group spend (zero for non-buyers). Both groups are
    resampled from it. In the treated group, extra non-buyers become buyers with
    spend drawn from real buyers, enough to raise revenue per customer by
    `revenue_lift` on average. `metric` is "spend" or "conversion".
    """
    rng = rng or np.random.default_rng(seed)
    spend = np.asarray(spend, float)
    buyer_spend = spend[spend > 0]
    p0 = (spend > 0).mean()
    extra_buyers = revenue_lift / buyer_spend.mean()
    flip = extra_buyers / (1 - p0)
    if not 0 <= flip <= 1:
        raise ValueError("revenue_lift is too large for this data")

    hits = 0
    for _ in range(n_sims):
        a = spend[rng.integers(0, len(spend), n_per_arm)]
        b = spend[rng.integers(0, len(spend), n_per_arm)]
        new = (b == 0) & (rng.random(n_per_arm) < flip)
        b[new] = buyer_spend[rng.integers(0, len(buyer_spend), new.sum())]
        if metric == "conversion":
            a, b = (a > 0).astype(float), (b > 0).astype(float)
        est = welch_t(b, a, alpha=alpha)
        hits += est.p_value < alpha and est.effect > 0
    return hits / n_sims
