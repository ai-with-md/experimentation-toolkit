import numpy as np
import pandas as pd
import pytest

from exptoolkit.causal import did, event_study, load_smoking, placebo_band


def _panel(effect=-10.0, trend_gap=0.0, seed=0):
    """Ten units over 20 years. Unit 'T' is a 50/50 blend of A and B, treated from year 10."""
    rng = np.random.default_rng(seed)
    years = np.arange(20)
    data = {f"U{i}": 100 + rng.normal(0, 1, 20).cumsum() + i * 3 for i in range(8)}
    data["A"] = 100 - 1.0 * years + rng.normal(0, 0.3, 20)
    data["B"] = 120 - 2.0 * years + rng.normal(0, 0.3, 20)
    t = 0.5 * data["A"] + 0.5 * data["B"] + trend_gap * years
    t[years >= 10] += effect
    data["T"] = t
    return pd.DataFrame(data, index=years)


def test_did_recovers_effect_when_trends_are_parallel():
    rng = np.random.default_rng(1)
    years = np.arange(20)
    base = 100 - years
    panel = pd.DataFrame({f"U{i}": base + i + rng.normal(0, 0.1, 20) for i in range(5)}, index=years)
    panel["T"] = base + 3 + np.where(years >= 10, -5.0, 0.0)
    assert did(panel, "T", 10) == pytest.approx(-5.0, abs=0.2)


def test_event_study_is_zero_at_reference_and_flags_a_pre_trend():
    panel = _panel(effect=0.0, trend_gap=-1.0)
    es = event_study(panel, "T", reference=9)
    assert es.loc[9] == 0
    assert es.loc[0] > 5          # a steady pre-trend shows up before the reference year


def test_placebo_band_is_ordered():
    panel = _panel()
    band = placebo_band(panel, "T", lambda d, u: event_study(d, u, reference=9))
    assert (band["low"] <= band["high"]).all()


def test_load_smoking_explains_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_smoking(tmp_path / "missing.csv")
