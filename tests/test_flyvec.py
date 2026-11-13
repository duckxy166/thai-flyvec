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


def reference_fit(inputs, word_freq, vocab_size, n_cells, seed, epochs, lr, batch_size):
    """The original dense implementation of FlyVec.fit, kept to check the fast one against."""
    from thai_flyvec.inputs import active_columns

    word_freq = np.asarray(word_freq, dtype=np.float64)
    p = np.maximum(word_freq / word_freq.sum(), np.finfo(np.float64).tiny)
    inv_p = np.tile(1.0 / p, 2).astype(np.float32)
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal((2 * vocab_size, n_cells), dtype=np.float32)
    wt = w0 / np.linalg.norm(w0, axis=0, keepdims=True)
    update = np.empty_like(wt)
    n = len(inputs)
    n_batches = -(-n // batch_size)
    total_steps = max(epochs * n_batches, 1)
    step, energies = 0, []
    for _ in range(epochs):
        energy = 0.0
        order = rng.permutation(n)
        for b in range(n_batches):
            batch = order[b * batch_size : (b + 1) * batch_size]
            cols = active_columns(inputs[batch], vocab_size)
            active = cols >= 0
            cols = np.where(active, cols, 0)
            x = np.where(active, inv_p[cols], 0).astype(np.float32)
            rows = wt[cols]
            winner = np.einsum("bw,bwk->bk", active.astype(np.float32), rows).argmax(axis=1)
            win_act = np.einsum("bw,bw->b", x, rows[np.arange(len(batch)), :, winner])
            norms = np.sqrt(np.einsum("ik,ik->k", wt, wt))
            energy -= float(np.sum(win_act / norms[winner]))
            decay = np.bincount(winner, weights=win_act, minlength=n_cells).astype(np.float32)
            np.multiply(wt, -decay, out=update)
            np.add.at(update, (cols[active], np.repeat(winner, cols.shape[1])[active.ravel()]), x[active])
            scale = max(float(update.max()), -float(update.min()), 1e-30)
            update *= lr * (1 - step / total_steps) / scale
            wt += update
            step += 1
        energies.append(energy / n)
    return wt.T, energies


@pytest.mark.parametrize(
    ("lr", "n_sentences", "epochs"),
    [
        (0.3, 300, 3),
        (1.0, 300, 3),
        # At lr = 3 training is chaotic: changing a single float32 rounding in the
        # dense reference itself makes it drift apart after ~20 steps. Check the
        # first steps only.
        (3.0, 40, 1),
    ],
)
def test_fit_matches_the_dense_reference(lr, n_sentences, epochs):
    """Same update as the original implementation, up to float rounding."""
    windows, counts = two_topic_corpus(n_sentences=n_sentences)
    args = dict(vocab_size=20, n_cells=12, seed=1, epochs=epochs, lr=lr, batch_size=32)
    want, want_energy = reference_fit(windows, counts, **args)
    got = FlyVec(20, n_cells=12, seed=1).fit(windows, counts, epochs=epochs, lr=lr, batch_size=32)
    np.testing.assert_allclose(got.weights, want, rtol=1e-3, atol=1e-4 * np.abs(want).max())
    np.testing.assert_allclose([h["energy"] for h in got.history], want_energy, rtol=1e-4)


def test_fit_matches_the_reference_on_a_zipf_corpus():
    """Many cells, a skewed vocabulary and rows that win often, as in the real corpus."""
    rng = np.random.default_rng(0)
    vocab = 200
    p = 1 / np.arange(1, vocab + 1)
    sentences = [rng.choice(vocab, size=12, p=p / p.sum()) for _ in range(400)]
    windows = window_vectors(sentences, w=5)
    counts = np.bincount(np.concatenate(sentences), minlength=vocab)
    want, want_energy = reference_fit(windows, counts, vocab, n_cells=50, seed=2, epochs=2, lr=1.0, batch_size=128)
    got = FlyVec(vocab, n_cells=50, seed=2).fit(windows, counts, epochs=2, lr=1.0, batch_size=128)
    np.testing.assert_allclose(got.weights, want, rtol=1e-3, atol=1e-4 * np.abs(want).max())
    np.testing.assert_allclose([h["energy"] for h in got.history], want_energy, rtol=1e-4)


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
