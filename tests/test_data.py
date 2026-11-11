import pandas as pd
import pytest

from exptoolkit.data import load_hillstrom


def _sample_csv(tmp_path, segments):
    rows = [{
        "recency": 1, "history_segment": "1) $0 - $100", "history": 50.0, "mens": 1,
        "womens": 0, "zip_code": "Urban", "newbie": 0, "channel": "Web",
        "segment": seg, "visit": 0, "conversion": 0, "spend": 0.0,
    } for seg in segments]
    path = tmp_path / "hillstrom.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_loader_adds_arm_and_treated(tmp_path):
    df = load_hillstrom(_sample_csv(tmp_path, ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]))
    assert list(df["arm"]) == ["control", "mens_email", "womens_email"]
    assert list(df["treated"]) == [0, 1, 1]


def test_loader_rejects_unknown_segment(tmp_path):
    with pytest.raises(ValueError):
        load_hillstrom(_sample_csv(tmp_path, ["Kids E-Mail"]))


def test_loader_explains_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_hillstrom(tmp_path / "missing.csv")
