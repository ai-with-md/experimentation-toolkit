import numpy as np
import pandas as pd
import pytest

from exptoolkit.uplift import (TLearner, XLearner, cross_fit_scores, evaluate_policies,
                               policy_profit, qini_coefficient, qini_curve, segment_effects,
                               split_qini)


def _heterogeneous(n=8_000, seed=0):
    """Treatment helps only customers with x > 0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = rng.integers(0, 2, n)
    y = x[:, 1] + 2.0 * t * (x[:, 0] > 0) + rng.normal(scale=0.5, size=n)
    return x, y, t


@pytest.mark.parametrize("Learner", [TLearner, XLearner])
def test_learners_find_who_responds(Learner):
    x, y, t = _heterogeneous()
    tau = Learner().fit(x, y, t).predict(x)
    assert tau[x[:, 0] > 0.5].mean() > 1.5
    assert abs(tau[x[:, 0] < -0.5].mean()) < 0.5


def test_qini_rewards_a_good_ranking_and_not_a_random_one():
    x, y, t = _heterogeneous(seed=1)
    good = qini_coefficient(x[:, 0], y, t)
    random = qini_coefficient(np.random.default_rng(2).normal(size=len(y)), y, t)
    assert good > 0.1
    assert abs(random) < good / 3


def test_qini_curve_ends_at_the_overall_effect():
    x, y, t = _heterogeneous(seed=3)
    shares, curve = qini_curve(x[:, 0], y, t)
    overall = y[t == 1].mean() - y[t == 0].mean()
    assert shares[-1] == 1.0
    assert curve[-1] == pytest.approx(overall * (t == 1).mean(), rel=0.05)


def test_cross_fit_scores_every_customer_out_of_fold():
    x, y, t = _heterogeneous(n=2_000, seed=4)
    scores = cross_fit_scores(TLearner, x, y, t, n_folds=2, seed=0)
    assert scores.shape == y.shape and np.isfinite(scores).all()


def test_policy_profit_matches_simple_difference_for_email_everyone():
    rng = np.random.default_rng(5)
    t = rng.integers(0, 2, 20_000)
    y = rng.exponential(1.0, 20_000) + 0.5 * t
    profit = policy_profit(np.ones(len(y), bool), y, t, margin=0.4, cost=0.1)
    expected = 1000 * (0.4 * (y[t == 1].mean() - y[t == 0].mean()) - 0.1)
    assert profit == pytest.approx(expected, rel=0.05)
    assert policy_profit(np.zeros(len(y), bool), y, t, 0.4, 0.1) == pytest.approx(0)


def test_targeting_wins_when_only_some_customers_respond():
    x, y, t = _heterogeneous(seed=6)
    policies = {"everyone": np.ones(len(y), bool), "responders": x[:, 0] > 0}
    table = evaluate_policies(policies, y, t, margin=1.0, cost=0.8, n_boot=200)
    assert table.loc["responders", "profit"] > table.loc["everyone", "profit"]


def test_segment_effects_and_split_qini_shapes():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"arm": rng.choice(["control", "mens_email"], 4_000),
                       "spend": rng.exponential(1.0, 4_000), "g": rng.integers(0, 2, 4_000)})
    seg = segment_effects(df, {"g": {"g=0": df["g"] == 0, "g=1": df["g"] == 1}}, "spend", "mens_email")
    assert list(seg["label"]) == ["g=0", "g=1"]
    x, y, t = _heterogeneous(n=2_000, seed=8)
    q = split_qini(lambda Xtr, ytr, ttr, Xte, idx: Xte[:, 0], x, y, t, n_splits=3)
    assert len(q) == 3 and (q > 0).all()
