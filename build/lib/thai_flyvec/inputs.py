"""Step 3: sliding-window inputs (context bag-of-words + one-hot target).

Following docs/decisions.md D2, a window is stored as the word ids it covers
instead of the dense 0/1 vector of length 2 * vocab_size: an int32 row of length
w whose middle entry is the target word. Positions outside the sentence and OOV
words are -1. `dense_input` turns a row back into the 0/1 vector of the
original definition.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .preprocess import OOV, encode

W_VALUES = (5, 11, 15)


def window_vectors(sentences: Iterable, vocab: dict[str, int] | None = None, w: int = 11) -> np.ndarray:
    """Windows of w words around every in-vocabulary target word.

    `sentences` are lists of words (pass `vocab`) or arrays of word ids with
    OOV = -1 (leave `vocab` as None). Returns int32 `(n_windows, w)`; column
    `w // 2` is the target. Windows do not cross sentence boundaries: positions
    past either end of the sentence are -1. Windows whose target is OOV are
    skipped.
    """
    if w < 3 or w % 2 == 0:
        raise ValueError(f"w must be odd and at least 3, got {w}")
    if vocab is not None:
        sentences = encode(sentences, vocab)
    half = w // 2

    # Lay the sentences end to end with `half` padding ids between them, so a
    # window around any real token never reaches into the next sentence.
    pieces, n_real = [np.full(half, OOV, dtype=np.int32)], 0
    for sentence in sentences:
        sentence = np.asarray(sentence, dtype=np.int32)
        pieces += [sentence, np.full(half, OOV, dtype=np.int32)]
        n_real += len(sentence)
    padded = np.concatenate(pieces)
    if n_real == 0:
        return np.zeros((0, w), dtype=np.int32)

    # Row i of `all_windows` is centred on padded[i + half].
    all_windows = np.lib.stride_tricks.sliding_window_view(padded, w)
    return all_windows[all_windows[:, half] >= 0]


def windows_from_corpus(corpus, w: int = 11) -> np.ndarray:
    """`window_vectors` for a `preprocess.Corpus`."""
    return window_vectors(corpus.sentences(), w=w)


def active_columns(windows: np.ndarray, vocab_size: int) -> np.ndarray:
    """Columns of the 2 * vocab_size input that are 1 for each window.

    Context words map to column `id`, the target to `vocab_size + id`. Each row
    keeps a column once even if the word repeats in the window (the input is
    0/1), and inactive slots are -1.
    """
    windows = np.asarray(windows)
    half = windows.shape[1] // 2
    cols = windows.astype(np.int64)
    target = cols[:, half]
    cols[:, half] = np.where(target >= 0, target + vocab_size, OOV)
    cols.sort(axis=1)
    repeated = np.zeros_like(cols, dtype=bool)
    repeated[:, 1:] = cols[:, 1:] == cols[:, :-1]
    cols[repeated] = OOV
    return cols


def dense_input(window: np.ndarray, vocab_size: int) -> np.ndarray:
    """The 0/1 input vector of length 2 * vocab_size for one window."""
    x = np.zeros(2 * vocab_size, dtype=np.uint8)
    cols = active_columns(np.asarray(window)[None, :], vocab_size)[0]
    x[cols[cols >= 0]] = 1
    return x
