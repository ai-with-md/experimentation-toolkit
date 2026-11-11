import numpy as np
import pytest

from exptoolkit.calibration import aa_test, calibration_table
from exptoolkit.estimators import welch_t


def test_aa_test_is_calibrated_on_well_behaved_data():
    values = np.random.default_rng(0).normal(10, 3, 20_000)
    result = aa_test(values, welch_t, n_per_arm=500, n_sims=600, seed=1)
    assert result.well_calibrated
    assert result.coverage == pytest.approx(1 - result.false_positive_rate)


def test_aa_test_rejects_too_large_groups():
    with pytest.raises(ValueError):
        aa_test(np.arange(100.0), welch_t, n_per_arm=60, n_sims=10)


def test_fpr_band_contains_alpha():
    values = np.random.default_rng(0).normal(size=5_000)
    result = aa_test(values, welch_t, n_per_arm=100, n_sims=200, seed=2)
    lo, hi = result.fpr_band
    assert lo < result.alpha < hi


def test_calibration_table_columns():
    values = np.random.default_rng(0).normal(size=2_000)
    table = calibration_table([aa_test(values, welch_t, 100, n_sims=50, method="w", metric="m")])
    assert {"False positive rate", "Interval coverage", "Calibrated"} <= set(table.columns)
