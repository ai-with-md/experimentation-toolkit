"""Sanity checks to run before reading any experiment results."""

import numpy as np
import pandas as pd
from scipy import stats


def sample_ratio_check(df: pd.DataFrame, arm_col: str = "arm", expected=None) -> dict:
    """Chi-square test that arm sizes match the planned split.

    `expected` maps arm name to planned share. Defaults to an equal split.
    A very small p-value (for example below 0.001) points to a problem with
    assignment or logging, and the results should not be trusted until it is
    explained.
    """
    counts = df[arm_col].value_counts().sort_index()
    if expected is None:
        shares = np.full(len(counts), 1 / len(counts))
    else:
        shares = np.array([expected[a] for a in counts.index])
    expected_counts = shares * counts.sum()
    stat, p = stats.chisquare(counts.values, expected_counts)
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "expected": {k: round(float(v), 1) for k, v in zip(counts.index, expected_counts)},
        "chi2": float(stat),
        "p_value": float(p),
    }


def standardized_mean_diff(df: pd.DataFrame, covariates, arm_col: str = "arm",
                           control: str = "control") -> pd.DataFrame:
    """Standardized mean difference of each covariate, each arm vs control.

    Values below 0.1 in absolute terms are usually read as well balanced.
    """
    ctrl = df[df[arm_col] == control]
    rows = []
    for arm in sorted(a for a in df[arm_col].unique() if a != control):
        trt = df[df[arm_col] == arm]
        for col in covariates:
            pooled_sd = np.sqrt((ctrl[col].var() + trt[col].var()) / 2)
            smd = (trt[col].mean() - ctrl[col].mean()) / pooled_sd if pooled_sd > 0 else 0.0
            rows.append({"arm": arm, "covariate": col, "smd": smd})
    return (
        pd.DataFrame(rows)
        .pivot(index="covariate", columns="arm", values="smd")
        .round(4)
    )
