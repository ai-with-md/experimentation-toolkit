# Contributing

Thanks for taking a look. Ideas, questions and pull requests are all welcome.

## Good ways to help

- **Run an A/A check on your own data.** If the t-test or the bootstrap misbehaves on a metric you work with, open an issue with the metric's shape (share of zeros, how skewed) and the group size. That is exactly the kind of evidence this project collects.
- **Suggest a module.** Each module answers one question a team actually asks. Open an issue with the question first, then the method.
- **Bring another public dataset.** A second randomized test would let every finding be checked somewhere new.
- **Fix something.** Typos, unclear wording and broken links are all fair game.

## Set up

```bash
git clone https://github.com/ai-with-md/experimentation-toolkit
cd experimentation-toolkit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks,dev]"
python scripts/download_data.py
pytest
```

## How the project is put together

- **Reusable code goes in `exptoolkit/`** and gets unit tests in `tests/`. Tests must not need the downloaded data, so they run on GitHub on every push.
- **Each notebook answers one question** and ends with a plain-language section on what it tells us and what it can't.
- **Numbers in the text must match the outputs.** Run the notebook top to bottom before committing, and save it with outputs.
- **Charts use the shared style** in `exptoolkit/style.py`. Each group or method keeps one colour everywhere, and titles state the finding.
- **The website** is built from the README and notebooks by `scripts/build_site.py`. Run it locally to preview, then open `site/index.html`.

## Pull requests

Keep each pull request to one change, describe what it changes and why, and make sure `pytest` passes. For anything bigger than a fix, please open an issue first so we can agree on the approach.
