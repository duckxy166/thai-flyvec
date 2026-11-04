from collections import Counter

import numpy as np

from thai_flyvec.preprocess import OOV, Corpus, build_vocab, clean_wiki_text, encode, tokenize


def test_clean_removes_markup():
    text = (
        "{{กล่องข้อมูล|ชื่อ={{lang|th|ไทย}}}}\n"
        "'''ประเทศไทย''' หรือ [[ราชอาณาจักรไทย|ราชอาณาจักร]]ไทย<ref name=\"a\">อ้างอิง</ref>"
        " อยู่ใน[[เอเชีย]]<ref name=\"b\" />\n"
        "== ประวัติ ==\n"
        "* [https://example.com เว็บไซต์] [[ไฟล์:ธง.png|thumb|ธง [[ไทย]]]]\n"
        "<!-- note -->[[หมวดหมู่:ประเทศ]]\n"
        "{| class=\"wikitable\"\n| เซลล์\n|}\n"
        "ก&nbsp;ข <br/>ค https://example.org"
    )
    assert clean_wiki_text(text).splitlines() == [
        "ประเทศไทย หรือ ราชอาณาจักรไทย อยู่ในเอเชีย",
        "ประวัติ",
        "เว็บไซต์",
        "ก ข ค",
    ]


def test_clean_normalizes_digits_and_repeats():
    assert clean_wiki_text("ปี ๒๕๖๓ อร่อยยยย มากกก") == "ปี 2563 อร่อย มาก"
    # duplicated tone marks / vowels are removed by pythainlp.util.normalize
    assert clean_wiki_text("น้้ำ") == "น้ำ"
    # Arabic digit runs are not shortened
    assert clean_wiki_text("5555") == "5555"


def test_tokenize_splits_sentences_and_words():
    sentences = tokenize("ฉันชอบแมว แมวนอนบนเตียง\nHello World!")
    assert sentences == [["ฉัน", "ชอบ", "แมว"], ["แมว", "นอน", "บน", "เตียง"], ["hello", "world"]]


def test_tokenize_drops_punctuation_and_keeps_numbers():
    assert tokenize("ใน พ.ศ. 2563 (ค.ศ. 2020)") == [["ใน"], ["พ.ศ.", "2563", "ค.ศ.", "2020"]]


def test_build_vocab_most_frequent_first():
    sentences = [["ก", "ข", "ค"], ["ข", "ค"], ["ค", "ง"]]
    assert build_vocab(sentences, 2) == {"ค": 0, "ข": 1}
    assert build_vocab(sentences) == {"ค": 0, "ข": 1, "ก": 2, "ง": 3}


def test_build_vocab_ties_follow_first_occurrence():
    assert list(build_vocab([["ข", "ก"], ["ก", "ข"], ["ค"]], 2)) == ["ข", "ก"]
    assert build_vocab(Counter({"ก": 1, "ข": 5}), 1) == {"ข": 0}


def test_encode_marks_oov():
    ids = encode([["ก", "x", "ข"]], {"ก": 0, "ข": 1})
    assert ids[0].tolist() == [0, OOV, 1]
    assert ids[0].dtype == np.int32


def test_corpus_roundtrip(tmp_path):
    sentences = [["ก", "ข", "ค"], ["ข", "ค"], ["ค", "ง"]]
    corpus = Corpus.from_sentences(sentences, size=2)
    assert corpus.words == ["ค", "ข"]
    assert corpus.counts.tolist() == [3, 2]
    assert [s.tolist() for s in corpus.sentences()] == [[OOV, 1, 0], [1, 0], [0, OOV]]
    assert corpus.word_sentences() == [["ข", "ค"], ["ข", "ค"], ["ค"]]

    corpus.save(tmp_path)
    loaded = Corpus.load(tmp_path)
    assert loaded.words == corpus.words
    assert np.array_equal(loaded.ids, corpus.ids)
    assert np.array_equal(loaded.offsets, corpus.offsets)
    assert np.array_equal(loaded.counts, corpus.counts)
    assert (tmp_path / "stats.json").exists()
