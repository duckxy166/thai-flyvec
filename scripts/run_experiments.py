"""Run the full experiment grid (docs/decisions.md D9) and save the raw results.

    python scripts/run_experiments.py                       # w = 5 11 15, seeds 0 1 2
    python scripts/run_experiments.py --w 11 --seeds 0      # one setting

For every (w, seed) it trains FlyVec, word2vec and fastText (or reuses the
saved model with `--reuse`), then evaluates

- FlyVec and binarized word2vec for every k in 4 8 16 32 64
- word2vec and fastText (dense, 300 dims)
- fastText with subword vectors for OOV words (extra row, D6)

on TH-WordSim-353, TH-SimLex-999 and wisesight_sentiment. Writes, after every
(w, seed) so a long run keeps what it has:

- `results/raw_results.csv`: model, k, w, seed, task, pairs, metric, value
- `results/resources.csv`: model, w, seed, train_minutes, size_mb, codes_mb
- `results/machine.json`: the machine the run used
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from thai_flyvec import pipeline  # noqa: E402
from thai_flyvec.evaluate import (  # noqa: E402
    known_pairs,
    load_similarity_pairs,
    load_wisesight,
    machine_spec,
    sentiment,
    word_similarity,
)
from thai_flyvec.flyvec import HASH_LENGTHS  # noqa: E402
from thai_flyvec.inputs import W_VALUES  # noqa: E402
from thai_flyvec.preprocess import Corpus  # noqa: E402

SIMILARITY_SETS = {"th-wordsim-353": "th-wordsim-353.csv", "th-simlex-999": "th-simlex-999.csv"}


def train_or_load(kind, corpus, models_dir, w, seed, args) -> dict:
    path = pipeline.model_path(models_dir, kind, w, seed)
    if args.reuse and path.exists() and path.with_suffix(".json").exists():
        print(f"  reuse {path.name}")
        return pipeline.load_meta(path)
    print(f"  train {path.name}", flush=True)
    if kind == "flyvec":
        return pipeline.train_flyvec(
            corpus, models_dir, w=w, seed=seed, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
        )
    return pipeline.train_baseline(kind, corpus, models_dir, w=w, seed=seed, epochs=args.epochs, workers=args.workers)


def embeddings(corpus, models_dir, w, seed):
    """(model, k, Embedding) for every row of the results table."""
    words = corpus.words
    flyvec = pipeline.load_flyvec(models_dir, w, seed)
    w2v = pipeline.load_keyed_vectors(models_dir, "word2vec", w, seed)
    ft = pipeline.load_keyed_vectors(models_dir, "fasttext", w, seed)
    for k in HASH_LENGTHS:
        yield "flyvec", k, pipeline.flyvec_embedding(flyvec, words, k)
    for k in HASH_LENGTHS:
        yield "word2vec_binary", k, pipeline.binary_embedding(w2v, words, k)
    yield "word2vec", None, pipeline.keyed_embedding(w2v, words, "word2vec")
    yield "fasttext", None, pipeline.keyed_embedding(ft, words, "fasttext")
    yield "fasttext_subword", None, pipeline.keyed_embedding(ft, words, "fasttext + subword", subword_oov=True)


def evaluate_setting(corpus, models_dir, w, seed, similarity, wisesight) -> list[dict]:
    rows = []
    embeds = list(embeddings(corpus, models_dir, w, seed))
    in_vocab = [e for model, _, e in embeds if model != "fasttext_subword"]
    for model, k, embed in embeds:
        base = {"model": model, "k": k, "w": w, "seed": seed}
        for task, pairs in similarity.items():
            pair_sets = {"common": known_pairs(pairs, *in_vocab)}
            if model == "fasttext_subword":
                pair_sets["all"] = pairs  # how far subwords get on OOV pairs
            for pair_set, chosen in pair_sets.items():
                result = word_similarity(embed, chosen)
                result["oov_ratio"] = 1 - result["n_pairs"] / len(pairs)  # against the full dataset
                rows += [
                    base | {"task": task, "pairs": pair_set, "metric": m, "value": result[m]}
                    for m in ("rho", "n_pairs", "oov_ratio")
                ]
        if wisesight is not None:
            result = sentiment(embed, wisesight, seed=seed)
            rows += [
                base | {"task": "wisesight", "pairs": "", "metric": m, "value": v} for m, v in result.items()
            ]
        print(f"    {embed.name}: done", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--w", type=int, nargs="+", default=list(W_VALUES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=3, help="same for every model (D6)")
    parser.add_argument("--lr", type=float, default=1.0, help="FlyVec learning rate")
    parser.add_argument("--batch-size", type=int, default=1024, help="FlyVec mini-batch")
    parser.add_argument("--workers", type=int, default=4, help="gensim worker threads")
    parser.add_argument("--reuse", action="store_true", help="reuse models already in models/")
    parser.add_argument("--skip-sentiment", action="store_true")
    args = parser.parse_args()

    corpus = Corpus.load(args.data_dir / "processed")
    similarity = {
        task: load_similarity_pairs(args.data_dir / "eval" / name) for task, name in SIMILARITY_SETS.items()
    }
    wisesight = None
    if not args.skip_sentiment:
        wisesight = load_wisesight(
            args.data_dir / "raw" / "wisesight", cache=args.data_dir / "processed" / "wisesight_tokens.json"
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "machine.json").write_text(json.dumps(machine_spec(), indent=2), encoding="utf-8")
    results, resources = [], []
    for w in args.w:
        for seed in args.seeds:
            print(f"w={w} seed={seed}", flush=True)
            for kind in pipeline.KINDS:
                meta = train_or_load(kind, corpus, args.models_dir, w, seed, args)
                resources.append(
                    {
                        "model": kind,
                        "w": w,
                        "seed": seed,
                        "train_minutes": meta["train_minutes"],
                        "size_mb": meta["size_mb"],
                        "codes_mb": meta.get("codes_mb"),
                    }
                )
            results += evaluate_setting(corpus, args.models_dir, w, seed, similarity, wisesight)
            pd.DataFrame(results).to_csv(args.results_dir / "raw_results.csv", index=False)
            pd.DataFrame(resources).to_csv(args.results_dir / "resources.csv", index=False)
    print(f"wrote {args.results_dir / 'raw_results.csv'} and {args.results_dir / 'resources.csv'}")


if __name__ == "__main__":
    main()
