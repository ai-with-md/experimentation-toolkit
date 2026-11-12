import numpy as np
import pytest

from exptoolkit.sequential import (_cumulative_z, naive_boundaries, obf_boundaries,
                                   run_sequential)


def test_obf_boundaries_shape():
    bar = obf_boundaries(5, n_paths=50_000)
    assert np.all(np.diff(bar) < 0)               # strict early, relaxing later
    assert bar[-1] > 1.96                          # last look slightly stricter than usual
    assert bar[-1] == pytest.approx(2.04, abs=0.05)  # known value for 5 looks


def test_obf_bar_scales_with_square_root_of_time():
    bar = obf_boundaries(4, n_paths=20_000)
    assert bar[0] == pytest.approx(bar[-1] * 2, rel=1e-9)


def test_naive_boundaries_are_flat():
    assert np.allclose(naive_boundaries(7), 1.959964, atol=1e-5)


def test_cumulative_z_last_look_matches_full_data():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 1_000), rng.normal(0.2, 1, 1_000)
    z = _cumulative_z(a, b, n_looks=4)
    se = np.sqrt(a.var(ddof=1) / 1_000 + b.var(ddof=1) / 1_000)
    assert z[-1] == pytest.approx((b.mean() - a.mean()) / se)


def _spend(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    buyers = rng.random(n) < 0.05
    out[buyers] = rng.lognormal(3, 0.5, buyers.sum())
    return out


def test_peeking_inflates_false_alarms_and_sequential_does_not():
    spend = _spend()
    peek = run_sequential(spend, naive_boundaries(10), 3_000, n_sims=800, seed=1)
    seq = run_sequential(spend, obf_boundaries(10, n_paths=50_000), 3_000, n_sims=800, seed=1)
    assert peek.rejection_rate > 0.12
    assert seq.rejection_rate < 0.075


def test_sequential_stops_early_on_a_big_effect():
    seq = run_sequential(_spend(), obf_boundaries(10, n_paths=50_000), 3_000, n_sims=200,
                         revenue_lift=2.0, seed=2)
    assert seq.rejection_rate > 0.9
    assert seq.average_stop < 10
