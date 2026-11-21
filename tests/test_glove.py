import shutil

import numpy as np
import pytest

from thai_flyvec import pipeline
from thai_flyvec.glove import GloVe, cooccurrence_matrix
from thai_flyvec.preprocess import Corpus


def two_topic_sentences(seed=0, n_sentences=600, length=8):
    """Words 0-9 only appear with each other, words 10-19 only with each other."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 10, size=length) + 10 * (i % 2) for i in range(n_sentences)]


def _cooc_dict(rows, cols, counts):
    return {(int(r), int(c)): float(v) for r, c, v in zip(rows, cols, counts)}


def test_cooccurrence_symmetry_distance_boundary_and_oov():
    # Sentence 0 has an OOV (-1) between word 0 and word 2; dropping OOV puts them at d=1.
    # Sentence 1 starts with word 3; must not co-occur with sentence 0.
    sentences = [np.array([0, -1, 2, 1], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
    rows, cols, counts = cooccurrence_matrix(sentences, window=5, vocab_size=5)  # half = 2
    d = _cooc_dict(rows, cols, counts)

    # Symmetry
    for (i, j), val in d.items():
        assert d[(j, i)] == pytest.approx(val)

    # Distance weighting after dropping -1: sentence 0 becomes [0, 2, 1]
    assert d[(0, 2)] == pytest.approx(1.0)
    assert d[(0, 1)] == pytest.approx(0.5)
    assert d[(2, 1)] == pytest.approx(1.0)
    assert d[(3, 4)] == pytest.approx(1.0)

    # Sentence boundary respected
    for i in (0, 1, 2):
        for j in (3, 4):
            assert (i, j) not in d


def test_cooccurrence_rejects_bad_window():
    with pytest.raises(ValueError):
        cooccurrence_matrix([np.array([0, 1], dtype=np.int32)], window=2, vocab_size=2)
    with pytest.raises(ValueError):
        cooccurrence_matrix([np.array([0, 1], dtype=np.int32)], window=4, vocab_size=2)


def test_glove_fit_shape_loss_and_topics():
    sentences = two_topic_sentences()
    cooc = cooccurrence_matrix(sentences, window=5, vocab_size=20)
    model = GloVe(20, dim=16, seed=0).fit(cooc, epochs=15, lr=0.05, workers=1, use_c=False)

    assert model.vectors.shape == (20, 16)
    assert model.vectors.dtype == np.float32
    assert len(model.history) == 15
    assert model.history[-1]["loss"] < model.history[0]["loss"]

    norms = np.linalg.norm(model.vectors, axis=1, keepdims=True)
    unit = model.vectors / np.where(norms == 0, 1, norms)
    cos = unit @ unit.T
    same = np.mean([cos[i, j] for i in range(20) for j in range(20) if i != j and i // 10 == j // 10])
    other = np.mean([cos[i, j] for i in range(20) for j in range(20) if i // 10 != j // 10])
    assert same > other + 0.3


def test_glove_seed_reproducibility():
    cooc = cooccurrence_matrix(two_topic_sentences(), window=5, vocab_size=20)
    a = GloVe(20, dim=8, seed=1).fit(cooc, epochs=3, workers=1, use_c=False)
    b = GloVe(20, dim=8, seed=1).fit(cooc, epochs=3, workers=1, use_c=False)
    c = GloVe(20, dim=8, seed=2).fit(cooc, epochs=3, workers=1, use_c=False)
    assert np.array_equal(a.vectors, b.vectors)
    assert not np.array_equal(a.vectors, c.vectors)


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="C compiler not available")
def test_glove_c_matches_numpy_single_thread():
    cooc = cooccurrence_matrix(two_topic_sentences(n_sentences=200), window=5, vocab_size=20)
    py_model = GloVe(20, dim=16, seed=3).fit(cooc, epochs=5, lr=0.05, workers=1, use_c=False)
    c_model = GloVe(20, dim=16, seed=3).fit(cooc, epochs=5, lr=0.05, workers=1, use_c=True)
    np.testing.assert_allclose(c_model.vectors, py_model.vectors, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        [h["loss"] for h in c_model.history],
        [h["loss"] for h in py_model.history],
        rtol=1e-4,
        atol=1e-6,
    )


def test_glove_save_load_and_pipeline_roundtrip(tmp_path):
    words = ["แมว", "ปลา", "หมา", "กระดูก"]
    sentences = [["แมว", "กิน", "ปลา"], ["หมา", "กิน", "กระดูก"]] * 15
    corpus = Corpus.from_sentences(sentences, size=10)
    models_dir = tmp_path / "models"

    meta = pipeline.train_glove(corpus, models_dir, w=5, seed=0, epochs=4, dim=8, workers=1)
    assert meta["kind"] == "glove"
    assert meta["dim"] == 8
    assert meta["size_mb"] > 0

    loaded = pipeline.load_glove(models_dir, w=5, seed=0)
    assert loaded.vocab_size == len(corpus.words)
    assert loaded.dim == 8
    assert len(loaded.history) == 4

    embed = pipeline.glove_embedding(loaded, corpus.words)
    assert embed.dim == 8
    assert embed.get(words[0]).shape == (8,)
    assert embed.get("ไม่รู้จัก") is None
