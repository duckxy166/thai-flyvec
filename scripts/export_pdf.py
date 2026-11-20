"""Print the slides and the short report to PDF with headless Chrome or Edge.

    python scripts/export_pdf.py docs/slides/index.html docs/slides/slides.pdf
    python scripts/export_pdf.py docs/report-short.html docs/report-short.pdf

The pages load Thai fonts from Google Fonts, so the machine needs internet.
Set CHROME to the browser executable if it is not in a usual place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def find_browser() -> str:
    for candidate in (os.environ.get("CHROME"), *CANDIDATES):
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    raise SystemExit("Chrome/Edge not found; set CHROME to its executable")


def export(html: Path, pdf: Path, wait_ms: int = 15000) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:  # a fresh profile, so a running browser is not reused
        subprocess.run(
            [
                find_browser(),
                "--headless=new",
                "--disable-gpu",
                f"--user-data-dir={profile}",
                "--no-pdf-header-footer",
                f"--virtual-time-budget={wait_ms}",
                f"--print-to-pdf={pdf.resolve()}",
                html.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    print(f"{html} -> {pdf} ({pdf.stat().st_size / 2**20:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--wait-ms", type=int, default=15000, help="time for fonts and images to load")
    args = parser.parse_args()
    export(args.html, args.pdf, args.wait_ms)


if __name__ == "__main__":
    main()
