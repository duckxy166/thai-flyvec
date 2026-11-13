"""Compare FlyVec learning rates (issue #33, docs/decisions.md D10).

    python scripts/compare_lr.py

Reads one run per (lr, seed) from `results/runs/lr<LR>_s<SEED>/raw_results.csv`
(made by `run_experiments.py --lr LR --seeds SEED --results-dir ...`) and the
FlyVec metadata `models/lr<LR>/flyvec_w<W>_s<SEED>.json`, then prints

- mean ± SD over seeds of every score, per model / k / lr, next to the
  published lr = 1.0 rows in `results/raw_results.csv`
- the FlyVec mean over k for each lr
- train time and energy per epoch of each FlyVec run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from thai_flyvec import report, use_utf8_output  # noqa: E402

MODELS = ("flyvec", "word2vec")
RUN_DIR = re.compile(r"lr(?P<lr>[0-9.]+)_s(?P<seed>\d+)$")


def load_runs(runs_dir: Path, w: int) -> pd.DataFrame:
    frames = []
    for path in sorted(runs_dir.iterdir()):
        match = RUN_DIR.match(path.name)
        if not (path.is_dir() and match and (path / "raw_results.csv").exists()):
            continue
        raw, _ = report.load(path)
        frames.append(report.scores(raw).assign(lr=match["lr"]))
    if not frames:
        raise SystemExit(f"no lr<LR>_s<SEED>/raw_results.csv under {runs_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df[df.model.isin(MODELS) & (df.w == w)]


def mean_sd(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = [c for c in report.SCORES.values() if c in df.columns]
    grouped = df.groupby(by)
    out = grouped[cols].agg(lambda s: f"{s.mean():.3f} ± {s.std():.3f}" if len(s) > 1 else f"{s.mean():.3f}")
    return out.assign(seeds=grouped.size()).reset_index()


def flyvec_meta(models_root: Path, w: int) -> pd.DataFrame:
    rows = []
    for path in sorted(models_root.glob(f"lr*/flyvec_w{w}_s*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "lr": path.parent.name.removeprefix("lr"),
                "seed": meta["seed"],
                "lr_in_meta": meta["lr"],
                "train_minutes": round(meta["train_minutes"], 1),
                "energy": [round(e) for e in meta["energy"]],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--published", type=Path, default=ROOT / "results")
    parser.add_argument("--w", type=int, default=11)
    args = parser.parse_args()

    runs = load_runs(args.runs_dir, args.w)
    published, _ = report.load(args.published)
    published = report.scores(published)
    published = published[published.model.isin(MODELS) & (published.w == args.w)].assign(lr="1.0 (published)")

    print(f"## mean ± SD over seeds, w = {args.w}\n")
    table = mean_sd(pd.concat([runs, published]), ["model", "k", "lr"])
    print(report.to_markdown(table.sort_values(["model", "k", "lr"])))

    print("\n## FlyVec mean over k (per seed, then mean ± SD over seeds)\n")
    fly = pd.concat([runs, published])
    fly = fly[fly.model == "flyvec"]
    cols = [c for c in report.SCORES.values() if c in fly.columns]
    per_seed = fly.groupby(["lr", "seed"])[cols].mean().reset_index()
    print(report.to_markdown(mean_sd(per_seed, ["lr"])))

    print("\n## FlyVec training\n")
    meta = flyvec_meta(args.models_dir, args.w)
    if len(meta):
        print(report.to_markdown(meta.sort_values(["lr", "seed"])))
        print()
        print(report.to_markdown(meta.groupby("lr").train_minutes.agg(["mean", "std"]).round(1).reset_index()))


if __name__ == "__main__":
    main()
