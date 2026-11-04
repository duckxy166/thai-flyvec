"""Train the word2vec and fastText baselines on the processed corpus.

    python scripts/train_baselines.py --w 11 --seed 0 --epochs 3

Saves `models/{word2vec,fasttext}_w<w>_s<seed>.kv` (gensim KeyedVectors) and a
`.json` with training time and size, then prints neighbours of the test words.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thai_flyvec import pipeline  # noqa: E402
from thai_flyvec.preprocess import Corpus  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_flyvec import print_neighbours  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--model", choices=("word2vec", "fasttext"), action="append")
    parser.add_argument("--w", type=int, default=11, help="window length, same as FlyVec")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--words", nargs="*", default=list(pipeline.TEST_WORDS))
    args = parser.parse_args()

    corpus = Corpus.load(args.corpus)
    for kind in args.model or ["word2vec", "fasttext"]:
        meta = pipeline.train_baseline(
            kind, corpus, args.models_dir, w=args.w, seed=args.seed, epochs=args.epochs, workers=args.workers
        )
        print(f"{kind}: {meta['train_minutes']:.1f} min, {meta['size_mb']:.1f} MB")
        kv = pipeline.load_keyed_vectors(args.models_dir, kind, args.w, args.seed)
        print_neighbours(pipeline.keyed_embedding(kv, corpus.words, kind), args.words)


if __name__ == "__main__":
    main()
