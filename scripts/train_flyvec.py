"""Train FlyVec on the processed corpus and print nearest neighbours of test words.

    python scripts/train_flyvec.py --w 11 --seed 0 --epochs 3

Saves `models/flyvec_w<w>_s<seed>.npz` and a `.json` with training time, size
and the energy per epoch (it should fall from epoch to epoch). The
neighbours use the k-bit hash codes and cosine similarity (D3, D4).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thai_flyvec import pipeline  # noqa: E402
from thai_flyvec.flyvec import K_CELLS  # noqa: E402
from thai_flyvec.preprocess import Corpus  # noqa: E402


def print_neighbours(embed, words, topn: int = 10) -> None:
    for word in words:
        neighbours = embed.nearest(word, topn)
        if not neighbours:
            print(f"{word}: (OOV)")
        else:
            print(f"{word}: " + ", ".join(n for n, _ in neighbours))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--cells", type=int, default=K_CELLS, help="number of Kenyon cells K")
    parser.add_argument("--w", type=int, default=11, help="window length")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32, help="hash length for the neighbour check")
    parser.add_argument("--words", nargs="*", default=list(pipeline.TEST_WORDS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    corpus = Corpus.load(args.corpus)
    print(f"corpus: {corpus.n_tokens:,} tokens, vocabulary {len(corpus.words):,}")
    meta = pipeline.train_flyvec(
        corpus,
        args.models_dir,
        w=args.w,
        seed=args.seed,
        epochs=args.epochs,
        n_cells=args.cells,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print(f"trained in {meta['train_minutes']:.1f} min, {meta['size_mb']:.1f} MB")

    model = pipeline.load_flyvec(args.models_dir, args.w, args.seed)
    print(f"\nnearest neighbours (k={args.k}):")
    print_neighbours(pipeline.flyvec_embedding(model, corpus.words, args.k), args.words)


if __name__ == "__main__":
    main()
