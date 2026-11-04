"""Training, saving and loading every model the same way, for the scripts and the demo.

Model files live in `models/` and are named `<kind>_w<w>_s<seed>`:
`flyvec_w11_s0.npz`, `word2vec_w11_s0.kv`, `fasttext_w11_s0.kv`. Each has a
`.json` sidecar with its training time and size (`evaluate.resources`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import baselines
from .evaluate import Embedding, resources
from .flyvec import K_CELLS, FlyVec
from .inputs import windows_from_corpus

KINDS = ("flyvec", "word2vec", "fasttext")
TEST_WORDS = (
    "แมว",
    "รถยนต์",
    "กรุงเทพมหานคร",
    "ภาษา",
    "ดนตรี",
    "ฟุตบอล",
    "โรงเรียน",
    "แม่น้ำ",
    "พระมหากษัตริย์",
    "คอมพิวเตอร์",
)


def model_path(models_dir: str | Path, kind: str, w: int, seed: int) -> Path:
    suffix = ".npz" if kind == "flyvec" else ".kv"
    return Path(models_dir) / f"{kind}_w{w}_s{seed}{suffix}"


def _write_meta(path: Path, meta: dict) -> dict:
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_meta(path: str | Path) -> dict:
    return json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))


def train_flyvec(
    corpus,
    models_dir,
    w: int = 11,
    seed: int = 0,
    epochs: int = 3,
    n_cells: int = K_CELLS,
    lr: float = 1.0,
    batch_size: int = 1024,
) -> dict:
    """Train FlyVec on the corpus and save it. Building the windows is timed too."""
    path = model_path(models_dir, "flyvec", w, seed)

    def train():
        windows = windows_from_corpus(corpus, w)
        return FlyVec(len(corpus.words), n_cells, seed).fit(
            windows, corpus.counts, epochs=epochs, lr=lr, batch_size=batch_size
        )

    result = resources(train, lambda model: model.save(path))
    model = result.pop("model")
    return _write_meta(
        path,
        result
        | {
            "kind": "flyvec",
            "w": w,
            "seed": seed,
            "epochs": epochs,
            "n_cells": n_cells,
            "lr": lr,
            "batch_size": batch_size,
            # Only the 0/1 target codes are needed at use time: vocab x n_cells bits.
            "codes_mb": len(corpus.words) * n_cells / 8 / 2**20,
            "energy": [h["energy"] for h in model.history],
        },
    )


def train_baseline(
    kind: str, corpus, models_dir, w: int = 11, seed: int = 0, epochs: int = 3, workers: int = 4
) -> dict:
    """Train word2vec or fastText on the corpus and save its KeyedVectors."""
    path = model_path(models_dir, kind, w, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_fn = {"word2vec": baselines.train_word2vec, "fasttext": baselines.train_fasttext}[kind]
    sentences = corpus.word_sentences()
    counts = {word: int(c) for word, c in zip(corpus.words, corpus.counts)}

    def save(model):
        model.wv.save(str(path))
        return [p for p in path.parent.glob(path.name + "*")]

    result = resources(
        lambda: train_fn(sentences, word_counts=counts, window=w, epochs=epochs, seed=seed, workers=workers),
        save,
    )
    result.pop("model")
    meta = result | {"kind": kind, "w": w, "seed": seed, "epochs": epochs}
    if kind == "word2vec":  # size of the binarized codes: one bit per dimension
        meta["codes_mb"] = len(corpus.words) * baselines.DIM / 8 / 2**20
    return _write_meta(path, meta)


def load_flyvec(models_dir, w: int = 11, seed: int = 0) -> FlyVec:
    return FlyVec.load(model_path(models_dir, "flyvec", w, seed))


def load_keyed_vectors(models_dir, kind: str, w: int = 11, seed: int = 0):
    from gensim.models import KeyedVectors

    return KeyedVectors.load(str(model_path(models_dir, kind, w, seed)))


def flyvec_embedding(model: FlyVec, words: list[str], k: int) -> Embedding:
    return Embedding(words, model.hash_all(k), name=f"FlyVec k={k}")


def keyed_embedding(kv, words: list[str], name: str, subword_oov: bool = False) -> Embedding:
    """Embedding over `words`; with `subword_oov`, fastText n-grams give vectors to OOV words."""
    fallback = None
    if subword_oov:

        def fallback(word: str):
            vector = kv[word]
            return vector if np.any(vector) else None

    return Embedding(words, baselines.vocab_matrix(kv, words), name=name, fallback=fallback)


def binary_embedding(kv, words: list[str], k: int, name: str = "word2vec binary") -> Embedding:
    return Embedding(words, baselines.binarize(baselines.vocab_matrix(kv, words), k), name=f"{name} k={k}")
