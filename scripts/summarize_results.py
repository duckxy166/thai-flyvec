"""Turn the raw results into the tables and figures for the report.

    python scripts/summarize_results.py --w 11 --k 32

Reads `results/raw_results.csv` and `results/resources.csv` from
`run_experiments.py` and writes to `results/`:

- `summary.md`, `summary.csv`: mean ± SD over seeds of every score (RQ1, RQ2)
- `best_k.csv`: the best hash length per binary model and w (RQ3)
- `oov.csv`: OOV pair ratio with and without fastText subwords
- `resources_summary.csv`: training time and size per model
- `figures/vs_k_*.png`: score against k (RQ1, RQ3)
- `figures/vs_resources_*.png`: score against training time and size (RQ2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from thai_flyvec import report, use_utf8_output  # noqa: E402


def main() -> None:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--w", type=int, default=11, help="window length for the main table and figures")
    parser.add_argument("--k", type=int, default=32, help="hash length for the resource figure")
    args = parser.parse_args()

    raw, resources = report.load(args.results_dir)
    out, figures = args.results_dir, args.results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    table = report.summary_table(raw, w=args.w)
    table.to_csv(out / "summary.csv", index=False)
    all_w = report.summary_table(raw)
    (out / "summary.md").write_text(
        f"# Results (w = {args.w}, mean ± SD over seeds)\n\n"
        "ρ is on the pairs every model knows (D7), except the fastText + subword OOV row, "
        "which is on all pairs.\n\n"
        + report.to_markdown(table)
        + "\n\n# All window sizes\n\n"
        + report.to_markdown(all_w)
        + "\n",
        encoding="utf-8",
    )
    present = [score for score in report.SCORES.values() if score in report.scores(raw).columns]
    best = [report.best_k(raw, score) for score in present]
    pd.concat(best).to_csv(out / "best_k.csv", index=False)
    report.oov_table(raw).to_csv(out / "oov.csv")
    report.resource_table(resources).to_csv(out / "resources_summary.csv")

    for score in present:
        slug = score.split()[0].lower() + ("_f1" if "F1" in score else "_acc" if "accuracy" in score else "")
        ax = report.plot_vs_k(raw, score, w=args.w)
        ax.figure.tight_layout()
        ax.figure.savefig(figures / f"vs_k_{slug}.png", dpi=150)
        plt.close(ax.figure)
        axes = report.plot_vs_resources(raw, resources, score, k=args.k, w=args.w)
        axes[0].figure.tight_layout()
        axes[0].figure.savefig(figures / f"vs_resources_{slug}.png", dpi=150)
        plt.close(axes[0].figure)
    print(table.to_string(index=False))
    print(f"\nwrote tables to {out} and figures to {figures}")


if __name__ == "__main__":
    main()
