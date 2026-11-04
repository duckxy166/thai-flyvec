import numpy as np
import pytest

from thai_flyvec.flyvec import FlyVec
from thai_flyvec.inputs import window_vectors


def two_topic_corpus(seed=0, n_sentences=2000, length=8):
    """Words 0-9 only appear with each other, words 10-19 only with each other."""
    rng = np.random.default_rng(seed)
    sentences = []
    for i in range(n_sentences):
        topic = i % 2
        sentences.append(rng.integers(0, 10, size=length) + 10 * topic)
    counts = np.bincount(np.concatenate(sentences), minlength=20)
    return window_vectors(sentences, w=5), counts


@pytest.fixture(scope="module")
def trained():
    windows, counts = two_topic_corpus()
    return FlyVec(20, n_cells=40, seed=0).fit(windows, counts, epochs=10, batch_size=64)


def test_weights_shape(trained):
    assert trained.weights.shape == (40, 40)


def test_energy_logged_and_decreasing(trained):
    energy = [h["energy"] for h in trained.history]
    assert len(energy) == 10
    assert energy[-1] < energy[0]


def test_winning_rows_converge_to_unit_norm(trained):
    norms = np.linalg.norm(trained.weights, axis=1)
    assert np.median(norms) == pytest.approx(1, abs=0.1)


def test_learns_topics(trained):
    codes = trained.hash_all(4).astype(float)
    overlap = codes @ codes.T
    same = np.mean([overlap[i, j] for i in range(20) for j in range(20) if i != j and i // 10 == j // 10])
    other = np.mean([overlap[i, j] for i in range(20) for j in range(20) if i // 10 != j // 10])
    assert same > 2 * other


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_hash_has_exactly_k_ones(trained, k):
    code = trained.hash(3, k)
    assert code.shape == (40,)
    assert code.dtype == np.uint8
    assert set(np.unique(code)) <= {0, 1}
    assert code.sum() == k
    assert np.array_equal(code, trained.hash_all(k)[3])
    assert (trained.hash_all(k).sum(axis=1) == k).all()


def test_hash_is_top_k_of_target_column(trained):
    column = trained.weights[:, 20 + 5]
    assert set(np.flatnonzero(trained.hash(5, 3))) == set(np.argsort(-column)[:3])


def test_hash_rejects_bad_k(trained):
    with pytest.raises(ValueError):
        trained.hash(0, 0)
    with pytest.raises(ValueError):
        trained.hash(0, 41)


def test_same_seed_same_model():
    windows, counts = two_topic_corpus()
    a = FlyVec(20, n_cells=10, seed=3).fit(windows, counts, epochs=1)
    b = FlyVec(20, n_cells=10, seed=3).fit(windows, counts, epochs=1)
    c = FlyVec(20, n_cells=10, seed=4).fit(windows, counts, epochs=1)
    assert np.array_equal(a.weights, b.weights)
    assert not np.array_equal(a.weights, c.weights)


def test_save_load_roundtrip(trained, tmp_path):
    path = trained.save(tmp_path / "models" / "flyvec")
    assert path.suffix == ".npz"
    loaded = FlyVec.load(path)
    assert (loaded.vocab_size, loaded.n_cells, loaded.seed) == (20, 40, 0)
    assert np.array_equal(loaded.weights, trained.weights)
    assert np.array_equal(loaded.hash_all(8), trained.hash_all(8))
    for got, want in zip(loaded.history, trained.history, strict=True):
        assert got == pytest.approx(want)


def test_word_freq_length_checked():
    windows, counts = two_topic_corpus(n_sentences=4)
    with pytest.raises(ValueError):
        FlyVec(21, n_cells=4).fit(windows, counts)


def test_untrained_hash_raises():
    with pytest.raises(RuntimeError):
        FlyVec(5).hash(0, 4)
