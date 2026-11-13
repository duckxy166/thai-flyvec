"""Step 6: word similarity, sentiment classification, resource usage.

Every model is wrapped in an `Embedding`: one vector per vocabulary word (float
vectors for word2vec/fastText, 0/1 codes for FlyVec and binarized word2vec).
All similarities are cosine (docs/decisions.md D4).
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

SENTIMENT_C = (0.01, 0.1, 1.0, 10.0, 100.0)


class Embedding:
    """Vectors for a fixed vocabulary, optionally with a fallback for OOV words.

    `fallback(word)` returns a vector for a word outside the vocabulary, or None.
    It is only used by the extra fastText-with-subwords row (D6).
    """

    def __init__(
        self,
        words: list[str],
        matrix: np.ndarray,
        name: str = "",
        fallback: Callable[[str], np.ndarray | None] | None = None,
    ):
        self.words = list(words)
        self.matrix = np.asarray(matrix)
        if len(self.words) != len(self.matrix):
            raise ValueError("one row per word")
        self.vocab = {word: i for i, word in enumerate(self.words)}
        self.name = name
        self.fallback = fallback

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def __contains__(self, word: str) -> bool:
        return word in self.vocab

    def get(self, word: str) -> np.ndarray | None:
        i = self.vocab.get(word)
        if i is not None:
            return self.matrix[i]
        return self.fallback(word) if self.fallback is not None else None

    def nearest(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        """Most cosine-similar vocabulary words; [] if the word has no vector."""
        v = self.get(word)
        if v is None:
            return []
        sims = _unit(self.matrix.astype(np.float32)) @ _unit(v.astype(np.float32))
        if word in self.vocab:
            sims[self.vocab[word]] = -np.inf
        top = np.argsort(-sims, kind="stable")[:topn]
        return [(self.words[i], float(sims[i])) for i in top]


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; 0 if either vector is all zeros."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


# --- word similarity ----------------------------------------------------------


def normalize_word(word: str) -> str:
    """Same normalization as the corpus (`clean_wiki_text` + lowercase)."""
    from .preprocess import clean_wiki_text

    return clean_wiki_text(word).strip().lower()


def load_similarity_pairs(path: str | Path) -> list[tuple[str, str, float]]:
    """Read a TH-WordSim-353 / TH-SimLex-999 csv: `word1,word2,score`, no header."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            w1, w2, score = line.rstrip("\n").rsplit(",", 2)
            pairs.append((normalize_word(w1), normalize_word(w2), float(score)))
    return pairs


def known_pairs(pairs, *embeds) -> list[tuple[str, str, float]]:
    """Pairs where every embedding has a vector for both words (D7)."""
    return [
        (w1, w2, s)
        for w1, w2, s in pairs
        if all(e.get(w1) is not None and e.get(w2) is not None for e in embeds)
    ]


def word_similarity(embed, pairs):
    """Spearman rho on TH-WordSim-353 / TH-SimLex-999, plus the OOV ratio.

    `embed` is an `Embedding` (or any mapping with `.get(word)`), `pairs` a list
    of `(word1, word2, score)`. A pair counts as OOV and is left out of rho when
    either word has no vector. Returns `rho`, `n_pairs` (pairs used),
    `n_oov` and `oov_ratio`.
    """
    from scipy.stats import spearmanr

    predicted, gold = [], []
    for w1, w2, score in pairs:
        v1, v2 = embed.get(w1), embed.get(w2)
        if v1 is None or v2 is None:
            continue
        predicted.append(cosine(v1, v2))
        gold.append(score)
    n_oov = len(pairs) - len(gold)
    rho = float(spearmanr(predicted, gold).statistic) if len(gold) > 1 else float("nan")
    return {
        "rho": rho,
        "n_pairs": len(gold),
        "n_oov": n_oov,
        "oov_ratio": n_oov / len(pairs) if pairs else float("nan"),
    }


# --- sentiment ------------------------------------------------------------------

WISESIGHT_SPLITS = ("train", "validation", "test")


def load_wisesight(raw_dir: str | Path, cache: str | Path | None = None) -> dict:
    """Tokenized wisesight_sentiment splits: `{split: (token_lists, labels)}`.

    Reads the `{split}.jsonl` files written by `scripts/download_data.py`
    (fields `texts` and `category`), normalizes them like the corpus and
    tokenizes with newmm (D8). The result is cached as JSON in `cache`.
    """
    from .preprocess import clean_wiki_text, tokenize

    if cache is not None and Path(cache).exists():
        data = json.loads(Path(cache).read_text(encoding="utf-8"))
        return {split: (tokens, labels) for split, (tokens, labels) in data.items()}

    splits = {}
    for split in WISESIGHT_SPLITS:
        tokens, labels = [], []
        with open(Path(raw_dir) / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sentences = tokenize(clean_wiki_text(row["texts"]))
                tokens.append([word for sentence in sentences for word in sentence])
                labels.append(int(row["category"]))
        splits[split] = (tokens, labels)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    return splits


def sentence_vectors(embed, token_lists: Iterable[list[str]]) -> tuple[np.ndarray, int, int, int]:
    """Mean word vector per text (D8).

    Returns `(X, n_empty, n_tokens, n_oov_tokens)`; a text with no known word
    gets a zero vector and is counted in `n_empty`.
    """
    rows, n_empty, n_tokens, n_oov = [], 0, 0, 0
    for tokens in token_lists:
        vectors = [v for v in (embed.get(t) for t in tokens) if v is not None]
        n_tokens += len(tokens)
        n_oov += len(tokens) - len(vectors)
        if vectors:
            rows.append(np.mean(np.asarray(vectors, dtype=np.float32), axis=0))
        else:
            rows.append(np.zeros(embed.dim, dtype=np.float32))
            n_empty += 1
    return np.stack(rows), n_empty, n_tokens, n_oov


def sentiment(embed, splits=None, Cs=SENTIMENT_C, seed: int = 0):
    """Accuracy and macro-F1 with logistic regression on wisesight_sentiment sentence vectors.

    `splits` is `{"train" | "validation" | "test": (token_lists, labels)}` from
    `load_wisesight`. C is picked by macro-F1 on validation, then the model
    is scored once on test. Also returns the chosen C, the number of texts with
    no known word (`n_empty`, all splits) and the OOV token ratio on test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if splits is None:
        raise ValueError("pass the tokenized splits from load_wisesight")
    features, n_empty, oov_ratio = {}, 0, float("nan")
    for split in WISESIGHT_SPLITS:
        tokens, labels = splits[split]
        X, empty, n_tokens, n_oov = sentence_vectors(embed, tokens)
        features[split] = (X, np.asarray(labels))
        n_empty += empty
        if split == "test":
            oov_ratio = n_oov / n_tokens if n_tokens else float("nan")

    best = None
    for C in Cs:
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed)
        )
        clf.fit(*features["train"])
        X_val, y_val = features["validation"]
        f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    X_test, y_test = features["test"]
    predicted = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "C": C,
        "n_empty": n_empty,
        "oov_token_ratio": oov_ratio,
    }


# --- resources ------------------------------------------------------------------


def file_size_mb(*paths: str | Path) -> float:
    return sum(os.path.getsize(p) for p in paths) / 2**20


def resources(train_fn, save_fn=None):
    """Training time (minutes) and model size (MB).

    Runs `train_fn()` and times it. If `save_fn(model)` is given it must save the
    model and return the path (or list of paths) written, whose total size is
    the model size. Returns `{"train_minutes", "size_mb", "model"}`.
    """
    start = time.perf_counter()
    model = train_fn()
    minutes = (time.perf_counter() - start) / 60
    size = float("nan")
    if save_fn is not None:
        paths = save_fn(model)
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        size = file_size_mb(*paths)
    return {"train_minutes": minutes, "size_mb": size, "model": model}


def machine_spec() -> dict:
    """The machine the results came from, saved next to them."""
    import numpy

    spec = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
    }
    try:
        import gensim

        spec["gensim"] = gensim.__version__
    except ImportError:
        pass
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        spec["memory_gb"] = round(pages * page_size / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    spec["processor"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return spec
