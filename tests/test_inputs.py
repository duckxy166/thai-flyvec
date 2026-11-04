import numpy as np
import pytest

from thai_flyvec.inputs import active_columns, dense_input, window_vectors


def test_windows_centre_on_each_known_word():
    windows = window_vectors([[0, 1, 2, 3, 4]], w=3)
    assert windows.dtype == np.int32
    assert windows.tolist() == [[-1, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, -1]]


def test_windows_do_not_cross_sentences():
    windows = window_vectors([[0, 1], [2, 3]], w=5)
    assert windows.tolist() == [
        [-1, -1, 0, 1, -1],
        [-1, 0, 1, -1, -1],
        [-1, -1, 2, 3, -1],
        [-1, 2, 3, -1, -1],
    ]


def test_oov_targets_skipped_but_kept_as_context():
    windows = window_vectors([[0, -1, 1]], w=3)
    assert windows.tolist() == [[-1, 0, -1], [-1, 1, -1]]


def test_windows_from_words_with_vocab():
    vocab = {"แมว": 0, "กิน": 1, "ปลา": 2}
    windows = window_vectors([["แมว", "กิน", "ปลา", "ทู"]], vocab, w=3)
    assert windows.shape == (3, 3)
    assert windows[:, 1].tolist() == [0, 1, 2]
    assert windows[2].tolist() == [1, 2, -1]  # "ทู" is OOV


def test_window_length_must_be_odd():
    with pytest.raises(ValueError):
        window_vectors([[0, 1]], w=4)


def test_empty_input():
    assert window_vectors([], w=11).shape == (0, 11)


def test_dense_input_positions():
    vocab_size = 10
    x = dense_input(np.array([3, 7, 5]), vocab_size)
    assert x.shape == (2 * vocab_size,)
    assert np.flatnonzero(x).tolist() == [3, 5, vocab_size + 7]


def test_dense_input_is_binary_with_repeated_context():
    x = dense_input(np.array([4, 4, 2, -1, 4]), vocab_size=5)
    assert np.flatnonzero(x).tolist() == [4, 5 + 2]
    assert x.max() == 1


def test_active_columns_marks_repeats_inactive():
    cols = active_columns(np.array([[4, 4, 2, -1, 4]]), vocab_size=5)
    assert sorted(cols[0].tolist()) == [-1, -1, -1, 4, 7]


def test_target_same_word_as_context_is_separate_column():
    x = dense_input(np.array([2, 2, 2]), vocab_size=3)
    assert np.flatnonzero(x).tolist() == [2, 3 + 2]
