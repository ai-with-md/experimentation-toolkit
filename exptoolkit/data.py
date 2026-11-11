"""Load and tidy the Hillstrom email marketing dataset."""

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "hillstrom.csv"

ARM_NAMES = {
    "No E-Mail": "control",
    "Mens E-Mail": "mens_email",
    "Womens E-Mail": "womens_email",
}

EXPECTED_COLUMNS = {
    "recency", "history_segment", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "visit", "conversion", "spend",
}


def load_hillstrom(path=DEFAULT_PATH) -> pd.DataFrame:
    """Return the dataset with a short `arm` column and a `treated` flag.

    `arm` is one of control, mens_email, womens_email.
    `treated` is 1 for either email, 0 for no email.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected file format, missing columns: {sorted(missing)}")

    df["arm"] = df["segment"].map(ARM_NAMES)
    if df["arm"].isna().any():
        raise ValueError("Found segment values that are not in ARM_NAMES.")
    df["treated"] = (df["arm"] != "control").astype(int)
    return df
