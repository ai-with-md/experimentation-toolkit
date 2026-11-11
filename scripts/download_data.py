"""Download the Hillstrom email marketing dataset into data/."""

from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = DATA_DIR / "hillstrom.csv"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if TARGET.exists():
        print(f"Already downloaded: {TARGET}")
        return
    print(f"Downloading {URL}")
    urlretrieve(URL, TARGET)
    print(f"Saved to {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
