"""Who does the treatment change? Uplift models, Qini curves and targeting policies.

Because treatment was randomized with a known probability, every customer's
expected uplift can be estimated by comparing similar treated and untreated
customers. Models are trained on one half of the data and judged only on the
other half.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def default_model():
    """A deliberately cautious booster: shallow trees, large leaves, few rounds."""
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                         min_samples_leaf=300, l2_regularization=1.0,
                                         random_state=0)


class TLearner:
    """Two models, one per group. Uplift is the difference of their predictions."""

    def __init__(self, make_model: Callable = default_model):
        self.make_model = make_model

    def fit(self, X, y, t):
        X, y, t = np.asarray(X), np.asarray(y, float), np.asarray(t)
        self.m1 = self.make_model().fit(X[t == 1], y[t == 1])
        self.m0 = self.make_model().fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.m1.predict(X) - self.m0.predict(X)


class XLearner:
    """Kunzel et al. (2019). Imputes each customer's effect, then models it directly.

    With a randomized 50/50 split the two stage-two models are averaged equally.
    """

    def __init__(self, make_model: Callable = default_model, propensity: float = 0.5):
        self.make_model = make_model
        self.propensity = propensity

    def fit(self, X, y, t):
        X, y, t = np.asarray(X), np.asarray(y, float), np.asarray(t)
        m1 = self.make_model().fit(X[t == 1], y[t == 1])
        m0 = self.make_model().fit(X[t == 0], y[t == 0])
        d1 = y[t == 1] - m0.predict(X[t == 1])      # treated: actual minus what control predicts
        d0 = m1.predict(X[t == 0]) - y[t == 0]      # control: what treated predicts minus actual
        self.tau1 = self.make_model().fit(X[t == 1], d1)
        self.tau0 = self.make_model().fit(X[t == 0], d0)
        return self

    def predict(self, X):
        g = self.propensity
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


def qini_curve(score, y, t, n_points: int = 51):
    """Incremental outcome per customer when targeting the top share ranked by `score`.

    At each share, the treated customers' total outcome is compared with the
    control customers' total, scaled to the same group size, then divided by the
    number of customers in the whole evaluation set.
    """
    score, y, t = np.asarray(score, float), np.asarray(y, float), np.asarray(t)
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    yt, yc = np.cumsum(y * (t == 1)), np.cumsum(y * (t == 0))
    nt, nc = np.cumsum(t == 1), np.cumsum(t == 0)
    n = len(y)
    shares = np.linspace(0, 1, n_points)
    out = np.zeros(n_points)
    for i, s in enumerate(shares[1:], start=1):
        k = max(1, int(round(s * n))) - 1
        if nt[k] and nc[k]:
            out[i] = (yt[k] - yc[k] * nt[k] / nc[k]) / n
    return shares, out


def _trapezoid(y, x):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)


def cross_fit_scores(make_learner: Callable, X, y, t, n_folds: int = 2, seed: int = 0):
    """Out-of-fold uplift scores: every customer is scored by a model that never saw them."""
    X, y, t = np.asarray(X), np.asarray(y, float), np.asarray(t)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), int)
    for group in (0, 1):                       # keep the treated share equal in every fold
        idx = np.nonzero(t == group)[0]
        folds[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
    scores = np.empty(len(y))
    for k in range(n_folds):
        train, test = folds != k, folds == k
        scores[test] = make_learner().fit(X[train], y[train], t[train]).predict(X[test])
    return scores


def qini_coefficient(score, y, t) -> float:
    """Area between the Qini curve and random targeting (positive = better than random)."""
    shares, curve = qini_curve(score, y, t)
    random_line = shares * curve[-1]
    return float(_trapezoid(curve - random_line, shares))


def bootstrap_qini(score, y, t, n_boot: int = 300, seed: int = 0):
    """Bootstrap draws of the Qini coefficient on the evaluation set."""
    rng = np.random.default_rng(seed)
    score, y, t = np.asarray(score, float), np.asarray(y, float), np.asarray(t)
    n = len(y)
    return np.array([qini_coefficient(score[idx], y[idx], t[idx])
                     for idx in (rng.integers(0, n, n) for _ in range(n_boot))])


def segment_effects(df: pd.DataFrame, segments: dict, metric: str, arm: str,
                    control: str = "control", arm_col: str = "arm") -> pd.DataFrame:
    """Lift of `arm` over `control` inside each segment.

    `segments` maps a group name to {label: boolean mask}. Rows keep that order.
    """
    from .estimators import welch_t

    rows = []
    for group, masks in segments.items():
        for label, mask in masks.items():
            sub = df[np.asarray(mask)]
            treated = sub.loc[sub[arm_col] == arm, metric]
            ctrl = sub.loc[sub[arm_col] == control, metric]
            est = welch_t(treated, ctrl)
            rows.append({"group": group, "label": label, "customers": len(treated) + len(ctrl),
                         "buyers": int((sub.loc[sub[arm_col].isin([arm, control]), metric] > 0).sum()),
                         "effect": est.effect, "ci_low": est.ci_low, "ci_high": est.ci_high})
    return pd.DataFrame(rows)


def split_qini(make_scores: Callable, X, y, t, n_splits: int = 20, seed: int = 0):
    """Qini on a held-out half, for many random halvings.

    `make_scores(X_train, y_train, t_train, X_test, test_index)` returns scores
    for the test half.
    """
    X, y, t = np.asarray(X), np.asarray(y, float), np.asarray(t)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        test = np.zeros(len(y), bool)
        for group in (0, 1):
            idx = np.nonzero(t == group)[0]
            test[rng.choice(idx, len(idx) // 2, replace=False)] = True
        train = ~test
        scores = make_scores(X[train], y[train], t[train], X[test], np.nonzero(test)[0])
        out.append(qini_coefficient(scores, y[test], t[test]))
    return np.array(out)
