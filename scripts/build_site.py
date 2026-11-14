"""Build the project website into site/: a landing page, every notebook, and the decision memo.

The landing page reads its findings and module list from README.md, so the
site and the README never disagree. Run locally with `python scripts/build_site.py`
and open site/index.html. GitHub Actions runs the same script on every push.
"""

import html
import re
import shutil
from pathlib import Path

import mistune
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO_URL = "https://github.com/ai-with-md/experimentation-toolkit"
SITE_URL = "https://ai-with-md.github.io/experimentation-toolkit/"
PROFILE_URL = "https://github.com/ai-with-md"

CSS = """
:root {
  color-scheme: dark;
  --bg: #0b0b0f; --surface: #111116; --line: #24232b; --ink: #f4f1ea; --ink-2: #b9b5c2;
  --muted: #85818f; --lime: #d4ff3f; --violet: #7b68ff; --coral: #f0563f;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); line-height: 1.55;
  font-family: "Avenir Next", Avenir, "Segoe UI", system-ui, -apple-system, sans-serif; }
a { color: inherit; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 0 16px; }
header.top { display: flex; justify-content: space-between; align-items: center; padding: 20px 0;
  color: var(--muted); font-weight: 600; font-size: 15px; }
header.top a { text-decoration: none; }
header.top a:hover { color: var(--ink); }
.hero { padding: 48px 0 32px; }
.eyebrow { color: var(--lime); font-weight: 700; letter-spacing: .02em; font-size: 15px; margin: 0 0 12px; }
h1 { font-size: clamp(36px, 6vw, 64px); line-height: 1.04; margin: 0 0 18px; letter-spacing: -0.02em; }
.lede { font-size: clamp(17px, 2.2vw, 21px); color: var(--ink-2); max-width: 720px; margin: 0 0 28px; }
.buttons { display: flex; flex-wrap: wrap; gap: 12px; }
.btn { display: inline-block; padding: 12px 20px; border-radius: 999px; font-weight: 700; text-decoration: none;
  border: 1px solid var(--line); background: var(--surface); }
.btn.primary { background: var(--lime); color: #111116; border-color: var(--lime); }
.btn:hover { transform: translateY(-1px); }
.hero img { width: 100%; border-radius: 16px; margin-top: 36px; border: 1px solid var(--line); display: block; }
section { padding: 40px 0; }
h2 { font-size: clamp(24px, 3.4vw, 34px); margin: 0 0 8px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin: 0 0 24px; }
.findings { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
.finding { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 20px; }
.finding .myth { color: var(--muted); font-size: 14px; font-weight: 600; margin: 0 0 8px; }
.finding .fact { font-size: 17px; margin: 0 0 12px; }
.finding .fact strong { color: var(--lime); }
.finding .links a { color: var(--ink-2); font-size: 14px; }
.modules { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
.module { display: block; text-decoration: none; background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; padding: 20px; transition: border-color .15s, transform .15s; }
.module:hover { border-color: var(--violet); transform: translateY(-2px); }
.module .num { color: var(--violet); font-weight: 800; font-size: 14px; }
.module .q { font-size: 19px; font-weight: 700; margin: 6px 0 8px; line-height: 1.3; }
.module .m { color: var(--muted); font-size: 14px; margin: 0; }
.principles { counter-reset: p; list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }
.principles li { counter-increment: p; background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; padding: 18px 20px 18px 64px; position: relative; color: var(--ink-2); }
.principles li::before { content: counter(p); position: absolute; left: 20px; top: 16px; color: var(--coral);
  font-weight: 800; font-size: 22px; }
.principles strong { color: var(--ink); }
pre { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px; overflow-x: auto; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 14px; }
footer { border-top: 1px solid var(--line); margin-top: 32px; padding: 28px 0 48px; color: var(--muted); font-size: 14px; }
.memo { max-width: 760px; padding: 32px 0 48px; }
.memo h1 { font-size: clamp(30px, 4.5vw, 44px); }
.memo h2 { font-size: 22px; margin-top: 32px; }
.memo table { border-collapse: collapse; width: 100%; display: block; overflow-x: auto; }
.memo th, .memo td { border: 1px solid var(--line); padding: 8px 12px; text-align: left; }
.memo strong { color: var(--ink); }
.memo li, .memo p { color: var(--ink-2); }
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<link rel="canonical" href="{canonical}">
<style>{css}</style>
</head>
<body>
<div class="wrap">
<header class="top"><a href="{home}">experimentation-toolkit</a><a href="{repo}">GitHub</a></header>
{body}
<footer>Built by <a href="{profile}">Meenal Asesh</a>. Code, data sources and tests on <a href="{repo}">GitHub</a>. MIT license.</footer>
</div>
</body>
</html>
"""

DESCRIPTION = ("Decision-first A/B testing in Python: A/A calibration, power by simulation, sequential "
               "testing, uplift and causal inference, measured on real data.")


def _inline(md: str) -> str:
    """Render one line of README markdown (bold, italics, links) to HTML."""
    out = mistune.html(md).strip()
    return re.sub(r"^<p>(.*)</p>$", r"\1", out, flags=re.S)


def _table(readme: str, heading: str):
    """Rows of the first markdown table under `heading`, as lists of cell strings."""
    section = readme.split(heading, 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("|")]
    cells = [[c.strip() for c in row.strip("|").split("|")] for row in rows]
    return [r for r in cells[2:] if r]          # skip header and divider


def _notebook_href(cell: str) -> str:
    return re.sub(r"\(notebooks/([^)]+)\.ipynb\)", r"(notebooks/\1.html)", cell)


def _module_card(num_cell: str, question: str, methods: str) -> str:
    number = re.search(r"\[(\d+)\]", num_cell).group(1)
    page = re.search(r"\(notebooks/([^)]+)\.ipynb\)", num_cell).group(1)
    methods = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", methods)    # drop links inside the card
    return (f'<a class="module" href="notebooks/{page}.html"><span class="num">{number}</span>'
            f'<p class="q">{html.escape(question)}</p><p class="m">{_inline(methods)}</p></a>')


def _principles(readme: str) -> list:
    section = readme.split("## How I approach experiments", 1)[1].split("\n## ", 1)[0]
    return [line.split(". ", 1)[1] for line in section.splitlines() if re.match(r"^\d+\. ", line)]


NAV_CSS = (
    "position:sticky;top:0;z-index:10;display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;"
    "justify-content:space-between;padding:12px 16px;background:#0b0b0f;border-bottom:1px solid #24232b;"
    "font:600 14px 'Avenir Next',Avenir,'Segoe UI',system-ui,sans-serif"
)
LINK_CSS = "color:#b9b5c2;text-decoration:none"


def _notebook_title(nb) -> str:
    """The notebook's first heading, without its markdown and spacing marks."""
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            first = cell.source.strip().splitlines()[0]
            return re.sub(r"\s+", " ", first.lstrip("# ").replace("&nbsp;", " ")).strip()
    return "Notebook"


def _nav(prev_page, next_page) -> str:
    def link(target, label):
        if target is None:
            return ""
        return f'<a href="{target[0]}.html" style="{LINK_CSS}">{label} {html.escape(target[1])}</a>'

    return (f'<nav style="{NAV_CSS}">'
            f'<a href="../index.html" style="color:#f4f1ea;text-decoration:none">experimentation-toolkit</a>'
            f'<span style="display:flex;flex-wrap:wrap;gap:8px 20px">'
            f'{link(prev_page, "Previous:")}{link(next_page, "Next:")}'
            f'<a href="{REPO_URL}" style="{LINK_CSS}">GitHub</a></span></nav>')


def build_notebooks() -> None:
    exporter = HTMLExporter(template_name="lab", theme="dark")
    out = SITE / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    notebooks = [nbformat.read(p, as_version=4) for p in paths]
    pages = [(p.stem, _notebook_title(nb)) for p, nb in zip(paths, notebooks)]
    for i, (nb, (stem, title)) in enumerate(zip(notebooks, pages)):
        resources = {"metadata": {"name": f"{title} | Experimentation Toolkit"}}
        body, _ = exporter.from_notebook_node(nb, resources=resources)
        nav = _nav(pages[i - 1] if i > 0 else None, pages[i + 1] if i + 1 < len(pages) else None)
        body = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + nav, body, count=1)
        (out / f"{stem}.html").write_text(body)


def build_memo() -> None:
    memo = (ROOT / "reports" / "decision_memo.md").read_text()
    body = f'<article class="memo">{mistune.html(memo)}</article>'
    (SITE / "memo.html").write_text(PAGE.format(
        title="Decision memo: Hillstrom email test", description=DESCRIPTION, css=CSS,
        canonical=SITE_URL + "memo.html", home="index.html", repo=REPO_URL, profile=PROFILE_URL, body=body))


def build_index() -> None:
    readme = (ROOT / "README.md").read_text()
    findings = "\n".join(
        f'<div class="finding"><p class="myth">{_inline(myth)}</p>'
        f'<p class="fact">{_inline(fact)}</p>'
        f'<p class="links">See notebook {_inline(_notebook_href(links))}</p></div>'
        for myth, fact, links in _table(readme, "## What most A/B tests get wrong"))
    modules = "\n".join(_module_card(num, q, m) for num, q, m in _table(readme, "## Modules"))
    principles = "\n".join(f"<li>{_inline(p)}</li>" for p in _principles(readme))
    quickstart = readme.split("## Use it on your own tests", 1)[1].split("\n## ", 1)[0]

    body = f"""
<section class="hero">
  <p class="eyebrow">Decision-first experimentation</p>
  <h1>Is your A/B test telling you the truth?</h1>
  <p class="lede">Eleven worked modules on real data: a 64,000-customer email test and California's 1988 tobacco tax.
  Every method is checked on the data before it is trusted, and every result ends in a decision.</p>
  <div class="buttons">
    <a class="btn primary" href="memo.html">Read the decision memo</a>
    <a class="btn" href="notebooks/05_peeking.html">The most surprising result</a>
    <a class="btn" href="{REPO_URL}">View on GitHub</a>
  </div>
  <img src="figures/peeking.png" alt="Checking every day turns a 5% false alarm rate into 21%">
</section>
<section>
  <h2>What most A/B tests get wrong</h2>
  <p class="sub">Measured on real customers, not assumed from a formula.</p>
  <div class="findings">{findings}</div>
</section>
<section>
  <h2>The modules</h2>
  <p class="sub">Each one is a question a team actually asks.</p>
  <div class="modules">{modules}</div>
</section>
<section>
  <h2>How I approach experiments</h2>
  <ol class="principles">{principles}</ol>
</section>
<section>
  <h2>Use it on your own tests</h2>
  {mistune.html(quickstart)}
</section>
"""
    (SITE / "index.html").write_text(PAGE.format(
        title="Experimentation Toolkit", description=DESCRIPTION, css=CSS, canonical=SITE_URL,
        home="index.html",
        repo=REPO_URL, profile=PROFILE_URL, body=body))


def build_sitemap() -> None:
    """sitemap.xml listing every page, to submit in Google Search Console.

    No robots.txt: crawlers only read one at the domain root, not inside a project site.
    """
    pages = ["", "memo.html"] + [f"notebooks/{p.name}" for p in sorted((SITE / "notebooks").glob("*.html"))]
    urls = "\n".join(f"  <url><loc>{SITE_URL}{page}</loc></url>" for page in pages)
    (SITE / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n</urlset>\n")


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    shutil.copytree(ROOT / "docs" / "figures", SITE / "figures")
    build_notebooks()
    build_memo()
    build_index()
    build_sitemap()
    print(f"Built {len(list((SITE / 'notebooks').glob('*.html')))} notebooks, memo and index in {SITE.name}/")


if __name__ == "__main__":
    main()
