"""Steps 1-2: clean Thai Wikipedia, tokenize, build the vocabulary."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

VOCAB_SIZE = 20_000
OOV = -1

# --- step 1: cleaning --------------------------------------------------------

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")  # innermost template, removed repeatedly
_FILE_LINK = re.compile(
    r"\[\[(?:file|image|ไฟล์|ภาพ|category|หมวดหมู่):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
    re.I,
)
_PIPED_LINK = re.compile(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]")
_LINK = re.compile(r"\[\[([^\[\]]*)\]\]")
_EXT_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]")
_URL = re.compile(r"https?://\S+")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_BOLD_ITALIC = re.compile(r"'{2,}")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_LIST_MARK = re.compile(r"^[*#:;]+\s*", re.M)
_REPEATED_THAI = re.compile(r"([ก-๎])\1{2,}")
_SPACES = re.compile(r"[ \t ​]+")


def clean_wiki_text(text: str) -> str:
    """Strip wiki markup with regex and normalize (repeated chars, Thai/Arabic digits).

    Works on raw wikitext from the dump and on the plain text from
    `wikimedia/wikipedia`, which only has stray markup left. Paragraphs stay on
    separate lines because `tokenize` treats a line break as a sentence break.
    """
    from pythainlp.util import normalize, thai_digit_to_arabic_digit

    text = _COMMENT.sub("", text)
    text = _REF.sub("", text)
    text = _TABLE.sub("", text)
    while True:
        stripped = _TEMPLATE.sub("", text)
        if stripped == text:
            break
        text = stripped
    text = _FILE_LINK.sub("", text)
    text = _PIPED_LINK.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _EXT_LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    text = _BOLD_ITALIC.sub("", text)
    text = _HEADING.sub(r"\1", text)
    text = _LIST_MARK.sub("", text)

    text = thai_digit_to_arabic_digit(text)
    text = _REPEATED_THAI.sub(r"\1", text)
    lines = (normalize(_SPACES.sub(" ", line)).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


# --- step 2: tokenization and vocabulary --------------------------------------

_THAI = "฀-๿"
# Thai marks sentence/clause breaks with a space. Split on a space only when a
# Thai character sits on both sides, so English phrases and "พ.ศ. 2563" stay whole.
_SENTENCE_BREAK = re.compile(rf"(?<=[{_THAI}])\s+(?=[{_THAI}])")
_WORD = re.compile(rf"[{_THAI}A-Za-z0-9]")


def tokenize(text: str) -> list[list[str]]:
    """Split into sentences, then words with PyThaiNLP newmm.

    Sentences are split on line breaks and on spaces between two Thai
    characters (Thai has no sentence-final punctuation). Tokens without a Thai letter, Latin letter or digit
    (punctuation, symbols) are dropped and Latin letters are lowercased.
    """
    from pythainlp.tokenize import word_tokenize

    sentences = []
    for line in text.splitlines():
        for chunk in _SENTENCE_BREAK.split(line):
            words = [
                token.strip().lower()
                for token in word_tokenize(chunk, engine="newmm", keep_whitespace=False)
            ]
            words = [token for token in words if _WORD.search(token)]
            if words:
                sentences.append(words)
    return sentences


def process_article(line: str) -> tuple[str, str, list[list[str]]]:
    """Clean and tokenize one JSON line `{"id", "title", "text"}` of the Wikipedia subset."""
    article = json.loads(line)
    return article["id"], article["title"], tokenize(clean_wiki_text(article["text"]))


def count_words(sentences: Iterable[list[str]]) -> Counter:
    counts: Counter = Counter()
    for sentence in sentences:
        counts.update(sentence)
    return counts


def build_vocab(sentences: list[list[str]], size: int = VOCAB_SIZE) -> dict[str, int]:
    """Map the `size` most frequent words to ids.

    Id 0 is the most frequent word. Ties are broken by first occurrence, so the
    same corpus always gives the same vocabulary. `sentences` may also be a
    `Counter` of word counts.
    """
    counts = sentences if isinstance(sentences, Counter) else count_words(sentences)
    return {word: i for i, (word, _) in enumerate(counts.most_common(size))}


def encode(sentences: Iterable[list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    """Turn word sentences into int32 id arrays, with OOV words as -1."""
    return [np.array([vocab.get(word, OOV) for word in s], dtype=np.int32) for s in sentences]


# --- saving and loading the processed corpus ---------------------------------


class Corpus:
    """Tokenized corpus as word ids, plus the vocabulary and word counts.

    `ids` is every token in order (OOV = -1) and sentence `i` is
    `ids[offsets[i]:offsets[i + 1]]`. `counts[j]` is how often vocabulary word
    `j` occurs in the corpus.
    """

    def __init__(self, ids: np.ndarray, offsets: np.ndarray, words: list[str], counts: np.ndarray):
        self.ids = np.asarray(ids, dtype=np.int32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.words = list(words)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.vocab = {word: i for i, word in enumerate(self.words)}

    @classmethod
    def from_sentences(cls, sentences: list[list[str]], size: int = VOCAB_SIZE) -> Corpus:
        counts = count_words(sentences)
        vocab = build_vocab(counts, size)
        words = list(vocab)
        ids = encode(sentences, vocab)
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(s) for s in ids])
        flat = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int32)
        return cls(flat, offsets, words, np.array([counts[w] for w in words]))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    def sentences(self) -> Iterable[np.ndarray]:
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            yield self.ids[start:end]

    def word_sentences(self) -> list[list[str]]:
        """Sentences as words with OOV tokens dropped, for gensim."""
        return [[self.words[i] for i in s if i >= 0] for s in self.sentences()]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "corpus.npz", ids=self.ids, offsets=self.offsets)
        with open(directory / "vocab.tsv", "w", encoding="utf-8") as f:
            for word, count in zip(self.words, self.counts):
                f.write(f"{word}\t{count}\n")
        stats = {
            "n_sentences": len(self),
            "n_tokens": self.n_tokens,
            "vocab_size": len(self.words),
            "oov_token_ratio": float(np.mean(self.ids < 0)) if self.n_tokens else 0.0,
        }
        (directory / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> Corpus:
        directory = Path(directory)
        data = np.load(directory / "corpus.npz")
        words, counts = load_vocab(directory / "vocab.tsv")
        return cls(data["ids"], data["offsets"], words, counts)


def load_vocab(path: str | Path) -> tuple[list[str], np.ndarray]:
    words, counts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, count = line.rstrip("\n").split("\t")
            words.append(word)
            counts.append(int(count))
    return words, np.array(counts, dtype=np.int64)
