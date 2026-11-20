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


def sentiment(embed, splits=None, Cs=SENTIMENT_C, seed: int = 0, return_predictions: bool = False):
    """Accuracy and macro-F1 with logistic regression on wisesight_sentiment sentence vectors.

    `splits` is `{"train" | "validation" | "test": (token_lists, labels)}` from
    `load_wisesight`. C is picked by macro-F1 on validation, then the model
    is scored once on test. Also returns the chosen C, the number of texts with
    no known word (`n_empty`, all splits) and the OOV token ratio on test.
    With `return_predictions` the test predictions are added as `test_pred`.
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
    result = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "C": C,
        "n_empty": n_empty,
        "oov_token_ratio": oov_ratio,
    }
    if return_predictions:
        result["test_pred"] = predicted
    return result


# --- what FlyVec is good at (issue #58, D11) -----------------------------------------


def quantile_bins(values, n_bins: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Split values into `n_bins` groups of about equal size (H1).

    Returns `(bin index per value, edges)`; ties at an edge go to the lower bin,
    so groups can be uneven when many values are equal.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    bins = np.searchsorted(edges[1:-1], values, side="left")
    return bins, edges


def pair_frequency(pairs, counts: dict[str, int]) -> np.ndarray:
    """Corpus count of the rarer word in each `(word1, word2, score)` pair (0 if unseen)."""
    return np.array([min(counts.get(w1, 0), counts.get(w2, 0)) for w1, w2, _ in pairs])


def similarity_by_frequency(embed, pairs, bins) -> list[dict]:
    """`word_similarity` separately for every frequency bin of `pairs` (H1)."""
    bins = np.asarray(bins)
    return [
        {"bin": int(b)} | word_similarity(embed, [p for p, pb in zip(pairs, bins) if pb == b])
        for b in sorted(set(bins.tolist()))
    ]


def oov_ratios(token_lists: Iterable[list[str]], vocab) -> np.ndarray:
    """Share of tokens outside `vocab` in each text; NaN for a text with no tokens (H2)."""
    return np.array(
        [sum(t not in vocab for t in tokens) / len(tokens) if tokens else np.nan for tokens in token_lists]
    )


def macro_f1_by_group(y_true, y_pred, groups) -> list[dict]:
    """Macro-F1 and size of every group of texts (H2); labels absent from a group are ignored."""
    from sklearn.metrics import f1_score

    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    rows = []
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        labels = np.unique(y_true[mask])
        rows.append(
            {
                "group": g,
                "n_texts": int(mask.sum()),
                "macro_f1": float(f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro")),
            }
        )
    return rows


PRACHATHAI_TOPICS = (
    "politics", "human_rights", "quality_of_life", "international", "social", "environment",
    "economics", "culture", "labor", "national_security", "ict", "education",
)  # fmt: skip
PRACHATHAI_SPLITS = ("train", "validation", "test")


def _tokenize_news(text: str) -> list[str]:
    from .preprocess import clean_wiki_text, tokenize

    return [word for sentence in tokenize(clean_wiki_text(text)) for word in sentence]


def load_prachathai(raw_dir: str | Path, cache: str | Path | None = None, workers: int = 1) -> dict:
    """Tokenized prachathai-67k splits: `{split: (token_lists, label_rows)}` (H3).

    Reads the `{split}.jsonl` files written by `scripts/download_data.py`
    (fields `title`, `body_text` and one 0/1 field per topic). The title and
    the body are joined, cleaned and tokenized like wisesight (D8, D11).
    `label_rows` holds one 0/1 list per article in `PRACHATHAI_TOPICS` order.
    The result is cached as JSON in `cache`.
    """
    if cache is not None and Path(cache).exists():
        data = json.loads(Path(cache).read_text(encoding="utf-8"))
        return {split: (tokens, labels) for split, (tokens, labels) in data.items()}

    from multiprocessing import Pool

    splits = {}
    pool = Pool(workers) if workers > 1 else None
    try:
        for split in PRACHATHAI_SPLITS:
            texts, labels = [], []
            with open(Path(raw_dir) / f"{split}.jsonl", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    texts.append(f"{row['title']}\n{row['body_text']}")
                    labels.append([int(row[topic]) for topic in PRACHATHAI_TOPICS])
            if pool is None:
                tokens = [_tokenize_news(text) for text in texts]
            else:
                tokens = pool.map(_tokenize_news, texts, chunksize=64)
            splits[split] = (tokens, labels)
            print(f"prachathai {split}: {len(tokens):,} articles tokenized", flush=True)
    finally:
        if pool is not None:
            pool.close()
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    return splits


def topic_classification(embed, splits, Cs=SENTIMENT_C, seed: int = 0, n_jobs: int = 1):
    """Multi-label topic classification on prachathai-67k sentence vectors (H3).

    Same features as `sentiment` (mean word vector, D8) with one logistic
    regression per topic (one-vs-rest). C is picked by macro-F1 on validation,
    then the model is scored once on test. Returns macro-F1, micro-F1, the
    chosen C and the number of articles with no known word (`n_empty`).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, n_empty = {}, 0
    for split in PRACHATHAI_SPLITS:
        tokens, labels = splits[split]
        X, empty, _, _ = sentence_vectors(embed, tokens)
        features[split] = (X, np.asarray(labels))
        n_empty += empty

    best = None
    for C in Cs:
        clf = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000, random_state=seed), n_jobs=n_jobs),
        )
        clf.fit(*features["train"])
        X_val, y_val = features["validation"]
        f1 = f1_score(y_val, clf.predict(X_val), average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    X_test, y_test = features["test"]
    predicted = clf.predict(X_test)
    return {
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_test, predicted, average="micro", zero_division=0)),
        "C": C,
        "n_empty": n_empty,
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
