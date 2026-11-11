import numpy as np
import pytest
from scipy import stats

from exptoolkit.estimators import bootstrap_percentile, welch_t


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 2.0, 300), rng.normal(0.5, 1.0, 200)
    est = welch_t(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert est.effect == pytest.approx(a.mean() - b.mean())
    assert est.p_value == pytest.approx(ref.pvalue, rel=1e-6)
    assert est.ci_low < est.effect < est.ci_high


def test_welch_handles_zero_variance():
    est = welch_t(np.zeros(50), np.zeros(50))
    assert est.effect == 0
    assert est.p_value == 1.0


def test_bootstrap_is_reproducible_with_seed():
    rng_data = np.random.default_rng(2)
    a, b = rng_data.exponential(1.0, 200), rng_data.exponential(1.0, 200)
    e1 = bootstrap_percentile(a, b, n_boot=300, rng=np.random.default_rng(7))
    e2 = bootstrap_percentile(a, b, n_boot=300, rng=np.random.default_rng(7))
    assert (e1.ci_low, e1.ci_high, e1.p_value) == (e2.ci_low, e2.ci_high, e2.p_value)


def test_bootstrap_detects_a_large_effect():
    rng = np.random.default_rng(3)
    a, b = rng.normal(1.0, 1.0, 500), rng.normal(0.0, 1.0, 500)
    est = bootstrap_percentile(a, b, n_boot=500, rng=rng)
    assert est.ci_low > 0
    assert est.p_value < 0.01
