"""Download the public datasets used by the notebooks into data/."""

from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    # Hillstrom email marketing challenge (MineThatData, 2008).
    "hillstrom.csv": (
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
    # Cigarette sales by US state, 1970 to 2000 (Abadie, Diamond and Hainmueller, 2010),
    # as published with the SyntheticControlMethods package.
    "smoking.csv": (
        "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/"
        "master/examples/datasets/smoking_data.csv"
    ),
}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        target = DATA_DIR / name
        if target.exists():
            print(f"Already downloaded: {target.name}")
            continue
        print(f"Downloading {name}")
        urlretrieve(url, target)
        print(f"Saved {target.name} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
