import numpy as np
import pandas as pd

from exptoolkit.checks import sample_ratio_check, standardized_mean_diff


def _arms(counts):
    return pd.DataFrame({"arm": np.repeat(list(counts), list(counts.values()))})


def test_srm_passes_for_an_even_split():
    result = sample_ratio_check(_arms({"control": 1000, "treatment": 1010}))
    assert result["p_value"] > 0.5


def test_srm_flags_an_uneven_split():
    result = sample_ratio_check(_arms({"control": 1000, "treatment": 1300}))
    assert result["p_value"] < 0.001


def test_srm_respects_a_planned_uneven_split():
    df = _arms({"control": 900, "treatment": 2700})
    result = sample_ratio_check(df, expected={"control": 0.25, "treatment": 0.75})
    assert result["p_value"] > 0.5


def test_smd_is_zero_for_identical_groups():
    df = pd.DataFrame({"arm": ["control"] * 4 + ["t"] * 4, "x": [1, 2, 3, 4] * 2})
    smd = standardized_mean_diff(df, ["x"])
    assert smd.loc["x", "t"] == 0
