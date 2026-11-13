"""Step 5: word2vec and fastText baselines (gensim, 300 dims) + binarized word2vec.

Both baselines use the vocabulary of the FlyVec corpus as is
(`build_vocab_from_freq`), so every model knows exactly the same 20,000 words
(docs/decisions.md D6). Settings shared by both are in D10.
"""

from __future__ import annotations

import numpy as np

DIM = 300
FASTTEXT_BUCKETS = 200_000


def _common(dim: int, window: int, epochs: int, seed: int, workers: int) -> dict:
    return {
        "vector_size": dim,
        "window": window // 2,  # gensim counts words on each side, FlyVec the whole window
        "shrink_windows": False,  # fixed window, like FlyVec
        "sg": 1,
        "negative": 5,
        "min_count": 1,
        "epochs": epochs,
        "seed": seed,
        "workers": workers,
    }


def _train(model, sentences, word_counts: dict[str, int]):
    model.build_vocab_from_freq(word_counts, corpus_count=len(sentences))
    model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
    return model


def train_word2vec(
    sentences,
    dim: int = DIM,
    *,
    word_counts: dict[str, int],
    window: int = 11,
    epochs: int = 3,
    seed: int = 0,
    workers: int = 4,
):
    """Skip-gram word2vec over `word_counts`'s vocabulary.

    `sentences` are lists of words (`Corpus.word_sentences()`), `window` is the
    total window length w used for FlyVec.
    """
    from gensim.models import Word2Vec

    params = _common(dim, window, epochs, seed, workers)
    return _train(Word2Vec(**params), sentences, word_counts)


def train_fasttext(
    sentences,
    dim: int = DIM,
    *,
    word_counts: dict[str, int],
    window: int = 11,
    epochs: int = 3,
    seed: int = 0,
    workers: int = 4,
    buckets: int = FASTTEXT_BUCKETS,
):
    """Skip-gram fastText; same settings as `train_word2vec` plus character n-grams."""
    from gensim.models import FastText

    params = _common(dim, window, epochs, seed, workers) | {"bucket": buckets}
    return _train(FastText(**params), sentences, word_counts)


def vocab_matrix(keyed_vectors, words: list[str]) -> np.ndarray:
    """Vectors of `words` in vocabulary order, as float32 `(len(words), dim)`."""
    return np.stack([keyed_vectors[word] for word in words]).astype(np.float32)


def binarize(vectors, k: int):
    """Set the top-k values of each vector to 1, the rest to 0."""
    vectors = np.asarray(vectors)
    single = vectors.ndim == 1
    vectors = np.atleast_2d(vectors)
    if not 1 <= k <= vectors.shape[1]:
        raise ValueError(f"k must be between 1 and {vectors.shape[1]}")
    top = np.argpartition(-vectors, k - 1, axis=1)[:, :k]
    codes = np.zeros(vectors.shape, dtype=np.uint8)
    np.put_along_axis(codes, top, 1, axis=1)
    return codes[0] if single else codes
