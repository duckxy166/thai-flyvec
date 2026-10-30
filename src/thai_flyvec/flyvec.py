"""Step 4: FlyVec learning rule in NumPy (Liang et al., ICLR 2021)."""

K_CELLS = 400
HASH_LENGTHS = (4, 8, 16, 32, 64)


class FlyVec:
    def __init__(self, vocab_size: int, n_cells: int = K_CELLS, seed: int = 0):
        self.vocab_size = vocab_size
        self.n_cells = n_cells
        self.seed = seed
        self.weights = None  # PN -> Kenyon cells, shape (n_cells, 2 * vocab_size)

    def fit(self, inputs, word_freq, epochs: int = 3, lr: float = 1e-3):
        """Winner-take-all update toward each input, weighted by inverse word frequency."""
        # TODO
        raise NotImplementedError

    def hash(self, word_id: int, k: int):
        """Binary code of length n_cells with 1 at the top-k activated Kenyon cells."""
        # TODO
        raise NotImplementedError
