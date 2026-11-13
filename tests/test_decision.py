import numpy as np
import pandas as pd
import pytest

from exptoolkit.decision import (apply_rule, bayesian_bootstrap_means, decision_table,
                                 profit_draws, sensitivity)
from exptoolkit.spec import load_spec


def test_bayesian_bootstrap_centres_on_the_mean_with_the_right_spread():
    rng = np.random.default_rng(0)
    values = rng.exponential(2.0, 5_000)
    draws = bayesian_bootstrap_means(values, n_draws=2_000, rng=rng)
    assert draws.mean() == pytest.approx(values.mean(), rel=0.01)
    assert draws.std() == pytest.approx(values.std() / np.sqrt(len(values)), rel=0.1)


def test_decision_table_basics():
    draws = pd.DataFrame({"a": np.zeros(4), "b": [1.0, 2.0, -1.0, 3.0], "c": [0.5, 0.5, 0.5, 0.5]})
    table = decision_table(draws)
    assert table["prob_best"].sum() == pytest.approx(1.0)
    assert table.loc["b", "prob_best"] == 0.75
    # b loses 1.5 in the draw where c wins (0.5 - (-1.0)); 1.5 / 4 draws
    assert table.loc["b", "expected_loss"] == pytest.approx(0.375)


def _fake_test(lift=1.0, n=4_000, seed=0):
    rng = np.random.default_rng(seed)
    arms = np.repeat(["control", "mens_email", "womens_email"], n)
    base = rng.exponential(5.0, len(arms))
    base[arms == "mens_email"] += lift
    return pd.DataFrame({"arm": arms, "spend": base})


def test_clear_winner_ships():
    spec = load_spec("hillstrom_email.yaml")
    table = decision_table(profit_draws(_fake_test(lift=2.0), spec, n_draws=500, seed=1))
    result = apply_rule(table, spec)
    assert result["choice"] == "mens_email"
    assert result["ship"]


def test_no_difference_does_not_ship_an_email():
    spec = load_spec("hillstrom_email.yaml")
    table = decision_table(profit_draws(_fake_test(lift=0.0), spec, n_draws=500, seed=2))
    result = apply_rule(table, spec)
    assert not (result["ship"] and result["choice"] != "control")


def test_sensitivity_switches_to_sending_nothing_when_email_is_expensive():
    lifts = pd.DataFrame({"mens_email": np.random.default_rng(3).normal(0.5, 0.01, 1_000)})
    grid = sensitivity(lifts, "control", margins=[0.4], costs=[0.0, 1.0],
                       prob_threshold=0.8, loss_threshold=10)
    assert list(grid["choice"]) == ["mens_email", "control"]
    assert grid["ship"].all()
