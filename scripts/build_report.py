"""Build the short course report (3-5 A4 pages) from Markdown to HTML and PDF.

    python scripts/build_report.py                  # docs/report-short.md -> .html -> .pdf
    python scripts/build_report.py --html-only

Needs the `markdown` package (`pip install markdown`).

Image paths in the Markdown are relative to `docs/`, so the HTML is written next
to it and the PDF is printed by `scripts/export_pdf.py` (headless Chrome/Edge).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_pdf import export  # noqa: E402

CSS = """
@page { size: A4; margin: 16mm 17mm 16mm 17mm }
html { font-family: 'Sarabun', 'Noto Sans Thai', Tahoma, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a }
body { margin: 0 }
h1 { font-size: 16pt; line-height: 1.3; margin: 0 0 2pt; text-align: center }
h1 + p { text-align: center; margin-top: 0 }
.meta { text-align: center; margin: 4pt 0 10pt; font-size: 10pt }
h2 { font-size: 12.5pt; margin: 10pt 0 3pt; border-bottom: 1px solid #bbb; padding-bottom: 1pt }
h3 { font-size: 11pt; margin: 7pt 0 2pt }
p { margin: 3pt 0; text-align: justify }
ul, ol { margin: 2pt 0; padding-left: 18pt }
li { margin: 1pt 0 }
table { border-collapse: collapse; margin: 4pt auto; font-size: 9pt; break-inside: avoid }
th, td { border: 1px solid #999; padding: 1.5pt 5pt }
th { background: #eee }
td { text-align: left }  /* numeric columns get `---:` in the Markdown */
figure { margin: 5pt 0; text-align: center; break-inside: avoid }
figure p { text-align: center }  /* markdown wraps the <img> in a justified <p> */
figure img { display: block; margin: 0 auto; max-width: 100% }
figcaption, .caption { font-size: 9pt; color: #333; text-align: center }
code { font-family: Consolas, monospace; font-size: 9pt }
a { color: #1a4f9c; text-decoration: none }
"""

TEMPLATE = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sarabun:ital,wght@0,400;0,600;0,700;1,400&display=swap">
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_html(md_path: Path, html_path: Path) -> None:
    import markdown

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "md_in_html"])
    title = text.splitlines()[0].lstrip("# ").strip()
    html_path.write_text(TEMPLATE.format(title=title, css=CSS, body=body), encoding="utf-8")
    print(f"{md_path} -> {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md", type=Path, default=ROOT / "docs" / "report-short.md")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    html_path = args.md.with_suffix(".html")
    build_html(args.md, html_path)
    if not args.html_only:
        export(html_path, args.md.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
