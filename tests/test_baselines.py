import numpy as np
import pytest

from thai_flyvec.baselines import binarize, train_fasttext, train_word2vec, vocab_matrix


@pytest.mark.parametrize("k", [1, 4, 64])
def test_binarize_k_ones_per_row(k):
    vectors = np.random.default_rng(0).standard_normal((50, 300))
    codes = binarize(vectors, k)
    assert codes.shape == (50, 300)
    assert codes.dtype == np.uint8
    assert (codes.sum(axis=1) == k).all()


def test_binarize_keeps_largest_values():
    assert binarize(np.array([0.1, -2.0, 3.0, 0.5]), 2).tolist() == [0, 0, 1, 1]
    assert binarize(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), 1).tolist() == [[0, 0, 1], [1, 0, 0]]


def test_binarize_rejects_bad_k():
    with pytest.raises(ValueError):
        binarize(np.zeros((2, 3)), 4)


SENTENCES = [["แมว", "กิน", "ปลา"], ["หมา", "กิน", "กระดูก"], ["แมว", "นอน"]] * 20
COUNTS = {"แมว": 40, "กิน": 40, "ปลา": 20, "หมา": 20, "กระดูก": 20}  # "นอน" left out


@pytest.mark.parametrize("train", [train_word2vec, train_fasttext])
def test_baselines_use_the_given_vocabulary(train):
    model = train(SENTENCES, dim=16, word_counts=COUNTS, window=5, epochs=2, seed=0, workers=1)
    assert set(model.wv.key_to_index) == set(COUNTS)
    assert model.wv.vector_size == 16
    assert model.window == 2
    matrix = vocab_matrix(model.wv, list(COUNTS))
    assert matrix.shape == (5, 16)
    assert matrix.dtype == np.float32


def test_word2vec_seed_is_reproducible():
    a = train_word2vec(SENTENCES, dim=8, word_counts=COUNTS, seed=1, workers=1)
    b = train_word2vec(SENTENCES, dim=8, word_counts=COUNTS, seed=1, workers=1)
    assert np.array_equal(a.wv["แมว"], b.wv["แมว"])


def test_fasttext_has_subword_vectors_for_oov():
    model = train_fasttext(SENTENCES, dim=8, word_counts=COUNTS, seed=0, workers=1, buckets=1000)
    assert "แมวน้ำ" not in model.wv.key_to_index
    assert np.any(model.wv["แมวน้ำ"])
