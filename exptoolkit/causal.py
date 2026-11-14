"""Causal effects without randomization: difference-in-differences and synthetic controls.

Both methods build a stand-in for what would have happened to the treated unit
without the policy. Difference-in-differences uses the average of the other
units and assumes the gap would have stayed constant. Synthetic control builds
a weighted blend of other units that tracks the treated unit before the policy.

With one treated unit, the usual standard errors don't apply. Instead every
method is re-run with each untreated unit pretending to be treated (placebos),
and the real result is judged against that spread.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SMOKING = Path(__file__).resolve().parent.parent / "data" / "smoking.csv"


def load_smoking(path=DEFAULT_SMOKING) -> pd.DataFrame:
    """Cigarette packs sold per person, by state and year, as a state-by-year table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df.pivot(index="year", columns="state", values="cigsale")


def did(panel: pd.DataFrame, treated: str, start: int) -> float:
    """Classic difference-in-differences: change in the treated unit minus change in the rest."""
    others = panel.drop(columns=treated).mean(axis=1)
    pre, post = panel.index < start, panel.index >= start
    return float((panel.loc[post, treated].mean() - panel.loc[pre, treated].mean())
                 - (others[post].mean() - others[pre].mean()))


def event_study(panel: pd.DataFrame, treated: str, reference: int) -> pd.Series:
    """Gap between the treated unit and the average of the rest, each year, relative to `reference`.

    Before the policy these should hover around zero if the two would have moved
    in parallel. A steady drift before the policy means they wouldn't have.
    """
    gap = panel[treated] - panel.drop(columns=treated).mean(axis=1)
    return gap - gap.loc[reference]


def placebo_band(panel: pd.DataFrame, treated: str, statistic, q=(0.05, 0.95)) -> pd.DataFrame:
    """Run `statistic(panel_without_treated, unit)` with every untreated unit as a fake treated unit.

    Returns the lower and upper quantiles across placebos, per index value.
    """
    donors = panel.drop(columns=treated)
    runs = pd.concat({unit: statistic(donors, unit) for unit in donors.columns}, axis=1)
    return pd.DataFrame({"low": runs.quantile(q[0], axis=1), "high": runs.quantile(q[1], axis=1)})
