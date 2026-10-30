"""Step 5: word2vec and fastText baselines (gensim, 300 dims) + binarized word2vec."""

DIM = 300


def train_word2vec(sentences, dim: int = DIM):
    # TODO
    raise NotImplementedError


def train_fasttext(sentences, dim: int = DIM):
    # TODO
    raise NotImplementedError


def binarize(vectors, k: int):
    """Set the top-k values of each vector to 1, the rest to 0."""
    # TODO
    raise NotImplementedError
