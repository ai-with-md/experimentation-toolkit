from pathlib import Path

import pandas as pd
import pytest
import yaml

from exptoolkit.spec import PLAN_DIR, load_spec


def test_repo_plan_loads():
    spec = load_spec("hillstrom_email.yaml")
    assert spec.control_arm == "control"
    assert set(spec.treatment_arms) == {"mens_email", "womens_email"}


def test_economics_thresholds():
    econ = load_spec("hillstrom_email.yaml").economics
    assert econ.profit_per_1000(econ.break_even_lift) == pytest.approx(0)
    assert econ.profit_per_1000(econ.worthwhile_lift) == pytest.approx(econ.min_profit_per_1000)


def _write_plan(tmp_path, **changes):
    raw = yaml.safe_load((PLAN_DIR / "hillstrom_email.yaml").read_text())
    for key, value in changes.items():
        section, _, field = key.partition("__")
        if field:
            raw[section][field] = value
        else:
            raw[section] = value
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_bad_margin_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="gross_margin"):
        load_spec(_write_plan(tmp_path, economics__gross_margin=1.5))


def test_unknown_control_arm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="control_arm"):
        load_spec(_write_plan(tmp_path, control_arm="holdout"))


def test_validate_against_finds_missing_columns():
    spec = load_spec("hillstrom_email.yaml")
    df = pd.DataFrame({"arm": ["control", "mens_email", "womens_email"], "spend": [0, 1, 2]})
    problems = spec.validate_against(df)
    assert any("visit" in p for p in problems)
