"""Step 4: FlyVec learning rule in NumPy (Liang et al., ICLR 2021).

Each of the `n_cells` Kenyon cells has a weight row over the 2 * vocab_size
projection-neuron input (context half, then target half). For an input window
v^A and word probabilities p, training minimises (Eq. 1 of the paper)

    E = - sum_A <W_mu^, v^A / p> / ||W_mu^||,   mu^ = argmax_mu <W_mu, v^A>

The winner mu^ is the cell with the largest plain activation <W_mu, v^A> (the
sum of the columns of W at the active positions); the 1/p weighting enters the
energy and the update (docs/decisions.md D2).

with the winner-take-all rule of the paper,

    dW_mu = g_mu * (v^A / p - <W_mu, v^A / p> W_mu),   g = 1 for the winner only,

which drives every winning row toward unit length (weight normalization).
As in the reference implementation of Krotov & Hopfield (2019) that FlyVec
builds on, the summed update of a mini-batch is divided by its largest absolute
value and the learning rate decays linearly to zero.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .inputs import active_columns

K_CELLS = 400
HASH_LENGTHS = (4, 8, 16, 32, 64)

log = logging.getLogger(__name__)


class FlyVec:
    def __init__(self, vocab_size: int, n_cells: int = K_CELLS, seed: int = 0):
        self.vocab_size = vocab_size
        self.n_cells = n_cells
        self.seed = seed
        self.weights = None  # PN -> Kenyon cells, shape (n_cells, 2 * vocab_size)
        self.history: list[dict] = []

    # --- training -------------------------------------------------------------

    def fit(
        self,
        inputs,
        word_freq,
        epochs: int = 3,
        lr: float = 1.0,
        batch_size: int = 1024,
    ) -> FlyVec:
        """Winner-take-all update toward each input, weighted by inverse word frequency.

        `inputs` is the `(n_windows, w)` id array from `inputs.window_vectors`
        and `word_freq` the count (or probability) of each vocabulary word.
        `lr` is the largest change of any single weight in the first step; it
        decays linearly to zero over all steps. Appends
        `{"epoch", "energy", "seconds"}` to `self.history` after every epoch;
        `energy` is E divided by the number of windows.
        """
        inputs = np.asarray(inputs)
        word_freq = np.asarray(word_freq, dtype=np.float64)
        if word_freq.shape != (self.vocab_size,):
            raise ValueError(f"word_freq must have length {self.vocab_size}")
        p = np.maximum(word_freq / word_freq.sum(), np.finfo(np.float64).tiny)
        inv_p = np.tile(1.0 / p, 2).astype(np.float32)

        rng = np.random.default_rng(self.seed)
        if self.weights is None:
            w0 = rng.standard_normal((2 * self.vocab_size, self.n_cells), dtype=np.float32)
            w0 /= np.linalg.norm(w0, axis=0, keepdims=True)
            self.weights = w0.T
        # Train on the transpose so the rows gathered per window are contiguous.
        wt = np.ascontiguousarray(self.weights.T)
        update = np.empty_like(wt)

        n = len(inputs)
        n_batches = -(-n // batch_size)
        total_steps = max(epochs * n_batches, 1)
        step = 0
        for epoch in range(len(self.history), len(self.history) + epochs):
            start, energy = time.perf_counter(), 0.0
            order = rng.permutation(n)
            for b in range(n_batches):
                batch = order[b * batch_size : (b + 1) * batch_size]
                cols = active_columns(inputs[batch], self.vocab_size)
                active = cols >= 0
                cols = np.where(active, cols, 0)
                x = np.where(active, inv_p[cols], 0).astype(np.float32)  # (B, w)

                rows = wt[cols]  # (B, w, n_cells)
                winner = np.einsum("bw,bwk->bk", active.astype(np.float32), rows).argmax(axis=1)
                win_act = np.einsum("bw,bw->b", x, rows[np.arange(len(batch)), :, winner])
                norms = np.sqrt(np.einsum("ik,ik->k", wt, wt))
                energy -= float(np.sum(win_act / norms[winner]))

                # sum over the winner's inputs of v/p - <W, v/p> W, in place to
                # avoid temporaries the size of W
                decay = np.bincount(winner, weights=win_act, minlength=self.n_cells).astype(np.float32)
                np.multiply(wt, -decay, out=update)
                np.add.at(update, (cols[active], np.repeat(winner, cols.shape[1])[active.ravel()]), x[active])
                scale = max(float(update.max()), -float(update.min()), 1e-30)
                update *= lr * (1 - step / total_steps) / scale
                wt += update
                step += 1
            self.history.append(
                {"epoch": epoch, "energy": energy / max(n, 1), "seconds": time.perf_counter() - start}
            )
            log.info("epoch %d  energy %.4g  %.1fs", epoch, energy / max(n, 1), self.history[-1]["seconds"])
        self.weights = wt.T
        return self

    # --- hashing ----------------------------------------------------------------

    def _target_weights(self) -> np.ndarray:
        """(vocab_size, n_cells): the activation of each cell when only the target word is on."""
        if self.weights is None:
            raise RuntimeError("model is not trained")
        return self.weights[:, self.vocab_size :].T

    def hash(self, word_id: int, k: int):
        """Binary code of length n_cells with 1 at the top-k activated Kenyon cells.

        Only the target half of the input is on (docs/decisions.md D3). The
        activation W[:, vocab_size + word_id] / p is ranked the same without
        the positive factor 1 / p, so it is left out.
        """
        return self.hash_all(k, [word_id])[0]

    def hash_all(self, k: int, word_ids=None) -> np.ndarray:
        """Codes for many words at once: uint8 `(len(word_ids), n_cells)`, all words by default."""
        if not 1 <= k <= self.n_cells:
            raise ValueError(f"k must be between 1 and {self.n_cells}")
        act = self._target_weights()
        if word_ids is not None:
            act = act[np.asarray(word_ids)]
        top = np.argpartition(-act, k - 1, axis=1)[:, :k]
        codes = np.zeros(act.shape, dtype=np.uint8)
        np.put_along_axis(codes, top, 1, axis=1)
        return codes

    # --- persistence ------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=np.asarray(self.weights, dtype=np.float32),
            vocab_size=self.vocab_size,
            n_cells=self.n_cells,
            seed=self.seed,
            energy=np.array([h["energy"] for h in self.history]),
            seconds=np.array([h["seconds"] for h in self.history]),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FlyVec:
        data = np.load(path)
        model = cls(int(data["vocab_size"]), int(data["n_cells"]), int(data["seed"]))
        model.weights = data["weights"]
        model.history = [
            {"epoch": i, "energy": float(e), "seconds": float(s)}
            for i, (e, s) in enumerate(zip(data["energy"], data["seconds"]))
        ]
        return model
