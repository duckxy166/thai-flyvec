"""Clean, tokenize and encode the Wikipedia subset (workflow steps 1-2).

    python scripts/preprocess_corpus.py --max-tokens 10000000

Reads `data/raw/thwiki_subset.jsonl` in order, cleans and tokenizes each
article (newmm) and stops after the article that reaches `--max-tokens`.
Writes `corpus.npz`, `vocab.tsv` (top `--vocab-size` words with counts),
`stats.json` and `articles_used.txt` to `data/processed/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thai_flyvec.preprocess import VOCAB_SIZE, Corpus, process_article  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "thwiki_subset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    args = parser.parse_args()

    start = time.perf_counter()
    sentences, used, n_tokens = [], [], 0
    pool = Pool(args.workers) if args.workers != 1 else None
    with open(args.input, encoding="utf-8") as f:
        articles = pool.imap(process_article, f, chunksize=16) if pool else map(process_article, f)
        for article_id, title, article in articles:
            sentences += article
            used.append(f"{article_id}\t{title}")
            n_tokens += sum(len(s) for s in article)
            if len(used) % 1000 == 0:
                print(f"{len(used):,} articles, {n_tokens:,} tokens", flush=True)
            if n_tokens >= args.max_tokens:
                break
    if pool:
        pool.terminate()
    if n_tokens < args.max_tokens:
        print(f"warning: input ran out at {n_tokens:,} tokens; download more with --wiki-max-chars")

    corpus = Corpus.from_sentences(sentences, args.vocab_size)
    corpus.save(args.out_dir)
    (args.out_dir / "articles_used.txt").write_text("\n".join(used) + "\n", encoding="utf-8")
    stats = json.loads((args.out_dir / "stats.json").read_text(encoding="utf-8"))
    stats |= {"n_articles": len(used), "minutes": (time.perf_counter() - start) / 60}
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
