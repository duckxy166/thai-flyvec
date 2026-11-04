import json

import numpy as np
import pytest

from thai_flyvec.evaluate import (
    Embedding,
    cosine,
    known_pairs,
    load_similarity_pairs,
    load_wisesight,
    machine_spec,
    resources,
    sentence_vectors,
    sentiment,
    word_similarity,
)


@pytest.fixture
def embed():
    words = ["แมว", "เสือ", "รถ", "ถนน"]
    matrix = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.8, 0.6]], dtype=np.float32)
    return Embedding(words, matrix, name="toy")


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine([0, 0], [1, 0]) == 0
    # for 0/1 codes with k ones cosine is overlap / k (D4)
    assert cosine([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_word_similarity_perfect_ranking(embed):
    pairs = [("แมว", "เสือ", 9.0), ("รถ", "ถนน", 7.0), ("แมว", "รถ", 1.0)]
    result = word_similarity(embed, pairs)
    assert result["rho"] == pytest.approx(1)
    assert result["n_pairs"] == 3
    assert result["oov_ratio"] == 0


def test_word_similarity_skips_oov_pairs(embed):
    pairs = [("แมว", "เสือ", 9.0), ("รถ", "ถนน", 1.0), ("แมว", "รถ", 7.0), ("แมว", "ช้าง", 5.0)]
    result = word_similarity(embed, pairs)
    assert result["n_pairs"] == 3
    assert result["n_oov"] == 1
    assert result["oov_ratio"] == pytest.approx(0.25)
    assert result["rho"] == pytest.approx(0.5)


def test_known_pairs_and_fallback(embed):
    with_fallback = Embedding(embed.words, embed.matrix, fallback=lambda w: np.ones(3) if w == "ช้าง" else None)
    pairs = [("แมว", "เสือ", 9.0), ("แมว", "ช้าง", 5.0), ("แมว", "ม้า", 5.0)]
    assert known_pairs(pairs, embed) == pairs[:1]
    assert known_pairs(pairs, with_fallback) == pairs[:2]
    assert known_pairs(pairs, embed, with_fallback) == pairs[:1]


def test_nearest(embed):
    neighbours = embed.nearest("รถ", 3)
    assert [w for w, _ in neighbours] == ["ถนน", "เสือ", "แมว"]
    assert neighbours[0][1] == pytest.approx(0.8)
    assert embed.nearest("ช้าง") == []


def test_load_similarity_pairs(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("เสือ,แมว,7.05\nคอมพิวเตอร์,แป้นพิมพ์,7.4\n\n", encoding="utf-8")
    assert load_similarity_pairs(path) == [("เสือ", "แมว", 7.05), ("คอมพิวเตอร์", "แป้นพิมพ์", 7.4)]


def test_sentence_vectors_mean_and_empty(embed):
    X, n_empty, n_tokens, n_oov = sentence_vectors(embed, [["แมว", "รถ", "ช้าง"], ["ช้าง"]])
    assert X[0] == pytest.approx([0.5, 0.5, 0])
    assert X[1] == pytest.approx([0, 0, 0])
    assert (n_empty, n_tokens, n_oov) == (1, 4, 2)


def toy_sentiment_splits(n=40):
    rng = np.random.default_rng(0)
    words = {0: ["ดี", "อร่อย", "ชอบ"], 1: ["แย่", "เกลียด", "เบื่อ"]}

    def split(size):
        labels = rng.integers(0, 2, size=size).tolist()
        tokens = [list(rng.choice(words[y], size=3)) + ["ร้าน"] for y in labels]
        return tokens, labels

    return {"train": split(n), "validation": split(n // 2), "test": split(n // 2)}


def test_sentiment_separable():
    words = ["ดี", "อร่อย", "ชอบ", "แย่", "เกลียด", "เบื่อ"]
    matrix = np.repeat(np.eye(2), 3, axis=0) + 0.01
    result = sentiment(Embedding(words, matrix), toy_sentiment_splits())
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 1
    assert result["oov_token_ratio"] == pytest.approx(0.25)  # "ร้าน"
    assert result["n_empty"] == 0
    assert result["C"] in (0.01, 0.1, 1.0, 10.0, 100.0)


def test_load_wisesight(tmp_path):
    raw = tmp_path / "wisesight"
    raw.mkdir()
    for split in ("train", "validation", "test"):
        rows = [{"texts": "อาหารอร่อยมากกก", "category": 0}, {"texts": "บริการแย่", "category": 2}]
        (raw / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    cache = tmp_path / "cache.json"
    splits = load_wisesight(raw, cache)
    assert splits["test"] == ([["อาหาร", "อร่อย", "มาก"], ["บริการ", "แย่"]], [0, 2])
    assert cache.exists()
    assert load_wisesight(tmp_path / "missing", cache) == splits


def test_resources(tmp_path):
    def save(model):
        path = tmp_path / "model.bin"
        path.write_bytes(b"0" * 2**20)
        return path

    result = resources(lambda: "model", save)
    assert result["model"] == "model"
    assert result["size_mb"] == pytest.approx(1)
    assert result["train_minutes"] >= 0


def test_machine_spec():
    spec = machine_spec()
    assert spec["cpu_count"] >= 1
    assert "python" in spec
