"""From effects to a decision: expected profit, chance each option is best, expected loss.

Uncertainty about each group's average comes from the Bayesian bootstrap: every
draw reweights the real customers at random, which gives a distribution for the
average without assuming a bell curve. Profit follows from the plan's economics.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def bayesian_bootstrap_means(values, n_draws: int = 4000, chunk: int = 250,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draws of the population mean of `values` (Rubin's Bayesian bootstrap)."""
    rng = rng or np.random.default_rng()
    values = np.asarray(values, float)
    out = np.empty(n_draws)
    for start in range(0, n_draws, chunk):
        k = min(chunk, n_draws - start)
        w = rng.exponential(size=(k, len(values)))
        # einsum rather than @: some numpy builds on macOS raise spurious matmul warnings.
        out[start:start + k] = np.einsum("ij,j->i", w, values) / w.sum(axis=1)
    return out


def profit_draws(df: pd.DataFrame, spec, n_draws: int = 4000, seed: int = 0,
                 arm_col: str = "arm") -> pd.DataFrame:
    """Draws of profit per 1,000 emails for every option, relative to sending nothing.

    The control option ("no email") is zero by definition. Each email's profit is
    margin times its revenue lift, minus the cost of sending.
    """
    rng = np.random.default_rng(seed)
    metric = spec.primary_metric["column"]
    means = {arm: bayesian_bootstrap_means(df.loc[df[arm_col] == arm, metric], n_draws, rng=rng)
             for arm in spec.arms}
    econ = spec.economics
    draws = {spec.control_arm: np.zeros(n_draws)}
    for arm in spec.treatment_arms:
        lift = means[arm] - means[spec.control_arm]
        draws[arm] = 1000 * (econ.gross_margin * lift - econ.cost_per_email)
    return pd.DataFrame(draws)[list(spec.arms)]


def lift_draws(df: pd.DataFrame, spec, n_draws: int = 4000, seed: int = 0,
               arm_col: str = "arm") -> pd.DataFrame:
    """Draws of revenue lift per customer for each treatment arm vs control."""
    rng = np.random.default_rng(seed)
    metric = spec.primary_metric["column"]
    means = {arm: bayesian_bootstrap_means(df.loc[df[arm_col] == arm, metric], n_draws, rng=rng)
             for arm in spec.arms}
    return pd.DataFrame({arm: means[arm] - means[spec.control_arm] for arm in spec.treatment_arms})


def decision_table(draws: pd.DataFrame) -> pd.DataFrame:
    """Expected profit, chance of being best and expected loss for each option.

    Expected loss of an option is the average profit given up by choosing it,
    compared with whichever option turns out best in each draw.
    """
    best = draws.max(axis=1)
    winner = draws.idxmax(axis=1)
    return pd.DataFrame({
        "expected_profit": draws.mean(),
        "profit_low": draws.quantile(0.025),
        "profit_high": draws.quantile(0.975),
        "prob_best": [(winner == c).mean() for c in draws.columns],
        "expected_loss": [(best - draws[c]).mean() for c in draws.columns],
    }, index=draws.columns)


def apply_rule(table: pd.DataFrame, spec) -> Dict[str, object]:
    """Apply the plan's decision rule to a decision table."""
    rule = spec.decision_rule["ship_if"]
    choice = table["expected_profit"].idxmax()
    row = table.loc[choice]
    passes_prob = row["prob_best"] >= rule["probability_best_at_least"]
    passes_loss = row["expected_loss"] <= rule["expected_loss_per_1000_at_most"]
    return {
        "choice": choice,
        "ship": bool(passes_prob and passes_loss),
        "prob_best": float(row["prob_best"]),
        "expected_loss": float(row["expected_loss"]),
        "passes_prob": bool(passes_prob),
        "passes_loss": bool(passes_loss),
    }


def sensitivity(lifts: pd.DataFrame, control_arm: str, margins: Sequence[float],
                costs: Sequence[float], prob_threshold: float, loss_threshold: float) -> pd.DataFrame:
    """Re-run the decision across a grid of margin and email-cost assumptions.

    `lifts` holds revenue-lift draws per treatment arm. Returns one row per
    (margin, cost) with the option chosen and whether the rule says ship.
    """
    rows = []
    for m in margins:
        for c in costs:
            draws = 1000 * (m * lifts - c)
            draws.insert(0, control_arm, 0.0)
            table = decision_table(draws)
            choice = table["expected_profit"].idxmax()
            r = table.loc[choice]
            ship = r["prob_best"] >= prob_threshold and r["expected_loss"] <= loss_threshold
            rows.append({"margin": m, "cost": c, "choice": choice, "ship": bool(ship),
                         "prob_best": r["prob_best"], "expected_loss": r["expected_loss"]})
    return pd.DataFrame(rows)
