import numpy as np
import pandas as pd
import pytest

from exptoolkit.estimators import welch_t
from exptoolkit.readout import covariate_matrix, cuped, effects_table, holm, variance_reduction
from exptoolkit.spec import load_spec


def test_holm_known_values():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_never_lowers_a_p_value_and_caps_at_one():
    p = np.array([0.2, 0.5, 0.9])
    adjusted = holm(p)
    assert np.all(adjusted >= p)
    assert adjusted.max() == 1.0


def _with_covariate(n=4_000, strength=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=2 * n)
    y = strength * x + rng.normal(size=2 * n)
    y[:n] += 0.1
    return y[:n], y[n:], x[:n], x[n:]


def test_cuped_keeps_the_effect_and_narrows_the_interval():
    y_t, y_c, x_t, x_c = _with_covariate()
    plain, adjusted = welch_t(y_t, y_c), cuped(y_t, y_c, x_t, x_c)
    assert adjusted.effect == pytest.approx(0.1, abs=0.05)
    assert (adjusted.ci_high - adjusted.ci_low) < 0.6 * (plain.ci_high - plain.ci_low)


def test_variance_reduction_matches_r_squared():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50_000)
    y = x + rng.normal(size=50_000)          # x explains half the variance
    assert variance_reduction(y, x) == pytest.approx(0.5, abs=0.02)
    assert variance_reduction(rng.normal(size=50_000), x) < 0.001


def test_covariate_matrix_turns_text_into_indicators():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "channel": ["Web", "Phone", "Web"]})
    assert covariate_matrix(df, ["a", "channel"]).shape == (3, 2)


def test_effects_table_applies_holm_to_secondary_metrics_only():
    rng = np.random.default_rng(2)
    n = 3_000
    df = pd.DataFrame({"arm": np.repeat(["control", "mens_email", "womens_email"], n)})
    df["spend"] = rng.exponential(1.0, len(df))
    df["visit"] = (rng.random(len(df)) < 0.1).astype(float)
    df["conversion"] = (rng.random(len(df)) < 0.01).astype(float)
    df["history"] = rng.normal(size=len(df))
    table = effects_table(df, load_spec("hillstrom_email.yaml"), covariates=["history"], n_boot=200)
    primary, secondary = table[table["role"] == "primary"], table[table["role"] == "secondary"]
    assert set(primary["method"]) == {"Welch t-test", "Bootstrap", "CUPED"}
    assert (primary["p_adjusted"] == primary["p_value"]).all()
    assert (secondary["p_adjusted"] >= secondary["p_value"]).all()
