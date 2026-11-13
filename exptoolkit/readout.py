"""The formal readout: effects with intervals, variance reduction and multiple comparisons."""

from typing import Sequence

import numpy as np
import pandas as pd

from .estimators import Estimate, bootstrap_percentile, welch_t


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # einsum rather than @: some numpy builds on macOS raise spurious matmul warnings.
    return np.einsum("ij,j->i", x, theta)


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def cuped(y_t, y_c, x_t, x_c, alpha: float = 0.05) -> Estimate:
    """Difference in means after adjusting for pre-treatment covariates (CUPED).

    The adjustment coefficients come from the pooled data. With one covariate
    this is classic CUPED; with several it is regression adjustment. Because
    the covariates were measured before treatment, the estimate stays unbiased
    and only its noise shrinks.
    """
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    x_t, x_c = _as_2d(x_t), _as_2d(x_c)
    x = np.vstack([x_t, x_c])
    y = np.concatenate([y_t, y_c])
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    theta, *_ = np.linalg.lstsq(xc, y - y.mean(), rcond=None)
    adj_t = y_t - _predict(x_t - x_mean, theta)
    adj_c = y_c - _predict(x_c - x_mean, theta)
    return welch_t(adj_t, adj_c, alpha=alpha)


def variance_reduction(y, x) -> float:
    """Share of the metric's variance that the covariates explain (what CUPED removes)."""
    y = np.asarray(y, float)
    x = _as_2d(x)
    xc = x - x.mean(axis=0)
    theta, *_ = np.linalg.lstsq(xc, y - y.mean(), rcond=None)
    resid = y - y.mean() - _predict(xc, theta)
    return float(1 - resid.var() / y.var())


def covariate_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Numeric covariates, with text columns turned into 0/1 indicators."""
    return pd.get_dummies(df[list(columns)], drop_first=True, dtype=float).to_numpy()


def effects_table(df: pd.DataFrame, spec, covariates: Sequence[str] = (), arm_col: str = "arm",
                  n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Every treatment arm against control, for the primary and secondary metrics.

    The primary metric is estimated three ways (Welch, bootstrap, CUPED) to show
    they agree. Secondary metrics get Holm-adjusted p-values across all
    arm-by-metric comparisons, as the plan specifies.
    """
    rng = np.random.default_rng(seed)
    alpha = spec.statistics["alpha"]
    control = df[df[arm_col] == spec.control_arm]
    x_all = covariate_matrix(df, covariates) if covariates else None
    rows = []
    for arm in spec.treatment_arms:
        treated = df[df[arm_col] == arm]
        primary = spec.primary_metric["column"]
        methods = {
            "Welch t-test": welch_t(treated[primary], control[primary], alpha),
            "Bootstrap": bootstrap_percentile(treated[primary], control[primary], alpha,
                                              n_boot=n_boot, rng=rng),
        }
        if x_all is not None:
            x_t = x_all[(df[arm_col] == arm).to_numpy()]
            x_c = x_all[(df[arm_col] == spec.control_arm).to_numpy()]
            methods["CUPED"] = cuped(treated[primary], control[primary], x_t, x_c, alpha)
        for method, est in methods.items():
            rows.append({"arm": arm, "metric": primary, "role": "primary", "method": method,
                         "effect": est.effect, "ci_low": est.ci_low, "ci_high": est.ci_high,
                         "p_value": est.p_value})
        for m in spec.secondary_metrics:
            est = welch_t(treated[m["column"]], control[m["column"]], alpha)
            rows.append({"arm": arm, "metric": m["column"], "role": "secondary",
                         "method": "Welch t-test", "effect": est.effect, "ci_low": est.ci_low,
                         "ci_high": est.ci_high, "p_value": est.p_value})
    table = pd.DataFrame(rows)
    table["p_adjusted"] = table["p_value"]
    sec = table["role"] == "secondary"
    if spec.statistics.get("multiple_comparisons") == "holm" and sec.any():
        table.loc[sec, "p_adjusted"] = holm(table.loc[sec, "p_value"])
    table["significant"] = table["p_adjusted"] < alpha
    return table
