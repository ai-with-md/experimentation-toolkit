import numpy as np
import pytest

from exptoolkit.power import (mde_means, n_for_power, power_means, power_proportions,
                              required_n_means, required_n_proportions, simulate_power)


def test_required_n_means_known_value():
    # 2 * ((1.96 + 0.8416) * 1 / 0.5) ** 2 = 62.8, rounded up
    assert required_n_means(sd=1.0, delta=0.5) == 63


def test_mde_and_required_n_agree():
    n = required_n_means(sd=10.0, delta=0.4)
    assert mde_means(sd=10.0, n_per_arm=n) == pytest.approx(0.4, rel=0.01)


def test_formula_power_hits_target_at_required_n():
    n = required_n_means(sd=5.0, delta=0.2)
    assert power_means(5.0, 0.2, n) == pytest.approx(0.8, abs=0.01)
    m = required_n_proportions(0.05, 0.01)
    assert power_proportions(0.05, 0.01, m) == pytest.approx(0.8, abs=0.01)


def test_n_for_power_interpolates():
    sizes, powers = [100, 200, 400], [0.5, 0.7, 0.9]
    n = n_for_power(sizes, powers, target=0.8)
    assert 200 < n < 400
    assert np.isnan(n_for_power(sizes, [0.1, 0.2, 0.3]))


def _spend(n=20_000, buy_rate=0.01, seed=0):
    rng = np.random.default_rng(seed)
    spend = np.zeros(n)
    buyers = rng.random(n) < buy_rate
    spend[buyers] = rng.lognormal(4, 0.8, buyers.sum())
    return spend


def test_simulated_power_is_low_with_no_lift():
    power = simulate_power(_spend(), revenue_lift=0.0, n_per_arm=2_000, n_sims=300, seed=1)
    assert power < 0.06


def test_simulated_power_grows_with_size():
    spend = _spend()
    small = simulate_power(spend, revenue_lift=0.5, n_per_arm=1_000, n_sims=200, seed=2)
    large = simulate_power(spend, revenue_lift=0.5, n_per_arm=20_000, n_sims=200, seed=3)
    assert large > small
    assert large > 0.8


def test_simulated_power_rejects_impossible_lift():
    with pytest.raises(ValueError):
        simulate_power(_spend(), revenue_lift=1e6, n_per_arm=100, n_sims=5)
