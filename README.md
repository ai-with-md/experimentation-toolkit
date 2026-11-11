# Experimentation Toolkit

Practical experiment design and causal inference, worked end to end on public data.

Each notebook answers a question a product or marketing team would actually ask, and ends with a plain-language conclusion someone could act on.

## Questions this project answers

| Notebook | Question | Status |
|---|---|---|
| 01 Explore | What does the data look like, and was the test randomized properly? | In progress |
| 02 Power analysis | How many customers do we need to detect a lift we care about? | Planned |
| 03 A/B analysis | Did the email work, and by how much? | Planned |
| 04 Uplift modeling | Who does the email actually change? | Planned |
| 05 Difference-in-differences and synthetic controls | What was the effect when we could not randomize? | Planned |

## Data

**Hillstrom email marketing dataset** (MineThatData, 2008). 64,000 customers who purchased in the last year were randomly split into three groups: a men's merchandise email, a women's merchandise email, or no email. Outcomes were tracked for two weeks: site visit, purchase, and spend.

The data is not stored in this repo. `scripts/download_data.py` fetches it from the original source.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks]"
python scripts/download_data.py
jupyter notebook notebooks/
```

## Project layout

```
exptoolkit/     reusable functions (data loading, tests, power, uplift)
notebooks/      one notebook per question, with outputs saved
scripts/        data download
data/           downloaded data (not committed)
```
