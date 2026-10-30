"""Steps 1-2: clean Thai Wikipedia, tokenize, build the vocabulary."""

VOCAB_SIZE = 20_000


def clean_wiki_text(text: str) -> str:
    """Strip wiki markup with regex and normalize (repeated chars, Thai/Arabic digits)."""
    # TODO
    raise NotImplementedError


def tokenize(text: str) -> list[list[str]]:
    """Split into sentences, then words with PyThaiNLP newmm."""
    # TODO
    raise NotImplementedError


def build_vocab(sentences: list[list[str]], size: int = VOCAB_SIZE) -> dict[str, int]:
    """Map the `size` most frequent words to ids."""
    # TODO
    raise NotImplementedError
