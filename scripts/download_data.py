"""Download every dataset the project uses into data/.

    python scripts/download_data.py                 # everything
    python scripts/download_data.py --only eval     # one part: wiki, eval, sentiment

- Thai Wikipedia: `wikimedia/wikipedia` (config `20231101.th`) through `datasets`.
  Articles are shuffled with a fixed seed and kept until `--wiki-max-chars`
  characters, enough for the ~10M-token subset of docs/decisions.md D1 (the
  exact token budget is applied by `preprocess_corpus.py`). Writes
  `data/raw/thwiki_subset.jsonl` and the list of article titles used,
  `data/raw/thwiki_articles.txt`.
- TH-WordSim-353, TH-SimLex-999 -> `data/eval/`.
- wisesight_sentiment -> `data/raw/wisesight/{train,validation,test}.jsonl`.

A part whose files already exist is skipped; pass `--force` to fetch it again.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.th"
SIMILARITY_URL = "https://raw.githubusercontent.com/gwohlgen/thai_word_similarity/master/{}"
SIMILARITY_FILES = ("th-wordsim-353.csv", "th-simlex-999.csv")
SENTIMENT_DATASET = "pythainlp/wisesight_sentiment"
SENTIMENT_PARQUET = (
    "hf://datasets/pythainlp/wisesight_sentiment@refs/convert/parquet/wisesight_sentiment/{}/*.parquet"
)


@contextlib.contextmanager
def atomic_paths(*paths: Path):
    """Yield temporary `.part` paths for `paths`; move them into place only if the block succeeds.

    The skip check of each download looks for the final files, so a download
    that stops halfway must never leave one behind.
    """
    parts = [p.with_name(p.name + ".part") for p in paths]
    try:
        yield parts
        for part, path in zip(parts, paths):
            os.replace(part, path)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def download_wiki(data_dir: Path, max_chars: int, seed: int, force: bool) -> None:
    out = data_dir / "raw" / "thwiki_subset.jsonl"
    titles = data_dir / "raw" / "thwiki_articles.txt"
    if out.exists() and titles.exists() and not force:
        print(f"skip wiki: {out} exists")
        return
    from datasets import load_dataset

    articles = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train").shuffle(seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_chars = n_articles = 0
    with atomic_paths(out, titles) as (out_part, titles_part):
        with open(out_part, "w", encoding="utf-8") as f, open(titles_part, "w", encoding="utf-8") as t:
            for article in articles:
                row = {"id": article["id"], "title": article["title"], "text": article["text"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                t.write(f"{article['id']}\t{article['title']}\n")
                n_chars += len(article["text"])
                n_articles += 1
                if n_chars >= max_chars:
                    break
    print(f"wiki: {n_articles:,} articles, {n_chars:,} characters -> {out}")


def download_eval(data_dir: Path, force: bool) -> None:
    eval_dir = data_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name in SIMILARITY_FILES:
        path = eval_dir / name
        if path.exists() and not force:
            print(f"skip eval: {path} exists")
            continue
        with atomic_paths(path) as (part,):
            urllib.request.urlretrieve(SIMILARITY_URL.format(name), part)
        print(f"eval: {path}")


def _load_sentiment():
    from datasets import load_dataset

    try:
        return load_dataset(SENTIMENT_DATASET)
    except Exception as error:  # newer `datasets` refuse the repo's loading script
        print(f"load_dataset({SENTIMENT_DATASET!r}) failed ({error}); using the parquet export")
        files = {split: SENTIMENT_PARQUET.format(split) for split in ("train", "validation", "test")}
        return load_dataset("parquet", data_files=files)


def download_sentiment(data_dir: Path, force: bool) -> None:
    out_dir = data_dir / "raw" / "wisesight"
    splits = ("train", "validation", "test")
    if all((out_dir / f"{s}.jsonl").exists() for s in splits) and not force:
        print(f"skip sentiment: {out_dir} exists")
        return
    dataset = _load_sentiment()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{split}.jsonl" for split in splits]
    with atomic_paths(*paths) as parts:  # all three splits or none
        for split, part in zip(splits, parts):
            with open(part, "w", encoding="utf-8") as f:
                for row in dataset[split]:
                    f.write(json.dumps({"texts": row["texts"], "category": row["category"]}, ensure_ascii=False) + "\n")
    for split, path in zip(splits, paths):
        print(f"sentiment: {len(dataset[split]):,} {split} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", choices=("wiki", "eval", "sentiment"), action="append")
    parser.add_argument(
        "--wiki-max-chars",
        type=int,
        default=80_000_000,
        help="characters of Wikipedia text to keep (default 80M, ~10M+ newmm tokens)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the article order")
    parser.add_argument("--force", action="store_true", help="download again even if the files exist")
    args = parser.parse_args()

    parts = args.only or ["wiki", "eval", "sentiment"]
    if "wiki" in parts:
        download_wiki(args.data_dir, args.wiki_max_chars, args.seed, args.force)
    if "eval" in parts:
        download_eval(args.data_dir, args.force)
    if "sentiment" in parts:
        download_sentiment(args.data_dir, args.force)


if __name__ == "__main__":
    main()
