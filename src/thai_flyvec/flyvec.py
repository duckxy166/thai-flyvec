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


class _ScaledColumns:
    """The weights during training, kept as W = V * s (V float32, one scale s per cell).

    The update of a mini-batch is dense, `-D_k W_ik` for every winning cell k,
    plus sparse terms at the active inputs. Applying it to all of W touched the
    whole (2 * vocab_size, n_cells) matrix several times per batch, which took
    most of the training time. Here the dense part only changes `s`, and V only
    changes at the active inputs, so a batch costs about the same as gathering
    its rows. The update is the same as before up to float rounding.

    Two quantities need the whole matrix and are kept up to date instead:
    the squared norm of every column of V (for the energy, recomputed exactly
    once per epoch) and, for the largest |update| that the step is divided by,
    the largest |V| of every column and its row.
    """

    # Fold s back into V when it drifts this far, so V stays in a safe float32 range.
    S_MIN, S_MAX = 1e-6, 1e6

    def __init__(self, wt: np.ndarray):
        self.v = wt  # (n_inputs, n_cells), float32, C order
        self.n_cells = wt.shape[1]
        self.s = np.ones(self.n_cells)
        self.sq_norms = np.zeros(self.n_cells)
        self.col_max = np.zeros(self.n_cells)  # max_i |V_ik|
        self.max_row = np.zeros(self.n_cells, dtype=np.int64)
        self.stale = np.ones(self.n_cells, dtype=bool)  # col_max / max_row need a recompute
        self.refresh_norms()

    def weights(self) -> np.ndarray:
        return (self.v * self.s).astype(np.float32)

    def refresh_norms(self) -> None:
        self.sq_norms = np.einsum("ik,ik->k", self.v, self.v, dtype=np.float64)

    def _refresh_column(self, k: int) -> None:
        column = np.abs(self.v[:, k])
        self.max_row[k] = int(np.argmax(column))
        self.col_max[k] = float(column[self.max_row[k]])
        self.stale[k] = False

    def step(self, cols, active, x, lr) -> float:
        """Apply one mini-batch; returns its (negative) contribution to the energy."""
        v, s, n_cells = self.v, self.s, self.n_cells
        rows = v[cols]  # (B, w, n_cells)
        winner = (np.einsum("bw,bwk->bk", active.astype(np.float32), rows) * s).argmax(axis=1)
        win_act = np.einsum("bw,bw->b", x, rows[np.arange(len(cols)), :, winner]) * s[winner]
        norms = np.abs(s) * np.sqrt(self.sq_norms)
        energy = -float(np.sum(win_act / norms[winner]))

        # The update is U = -D_k W_ik everywhere in a winning column k, plus the
        # summed inputs S at the active (row, k) pairs ("support").
        decay = np.bincount(winner, weights=win_act, minlength=n_cells)
        keys = cols[active].astype(np.int64) * n_cells + np.repeat(winner, cols.shape[1])[active.ravel()]
        keys, inverse = np.unique(keys, return_inverse=True)
        added = np.bincount(inverse, weights=x[active])
        sup_rows, sup_cols = np.divmod(keys, n_cells)
        w_sup = v[sup_rows, sup_cols] * s[sup_cols]  # W at the support, before the step
        u_sup = added - decay[sup_cols] * w_sup

        # Largest |U|: the support values, or |D_k s_k| times the largest |V| of
        # column k outside the support. The tracked column max is exact unless
        # its row is in the support; then that column is read directly.
        winners = np.unique(winner)
        for k in winners[self.stale[winners]]:
            self._refresh_column(int(k))
        largest = float(np.abs(u_sup).max())
        bound = np.abs(decay[winners] * s[winners]) * self.col_max[winners]
        in_support = np.isin(self.max_row[winners] * n_cells + winners, keys)
        for i in np.argsort(-bound, kind="stable"):
            if bound[i] <= largest:
                break
            if not in_support[i]:
                largest = float(bound[i])
                break
            k = int(winners[i])
            outside = np.ones(len(v), dtype=bool)
            outside[sup_rows[sup_cols == k]] = False
            if outside.any():
                exact = float(np.abs(v[outside, k]).max()) * abs(decay[k] * s[k])
                largest = max(largest, exact)
        coef = lr / max(largest, 1e-30)

        # W <- W + coef * U: scale the winning columns, then set the support.
        old_sq = v[sup_rows, sup_cols].astype(np.float64) ** 2
        s[winners] *= 1 - coef * decay[winners]
        fold = winners[(np.abs(s[winners]) < self.S_MIN) | (np.abs(s[winners]) > self.S_MAX)]
        for k in fold:
            v[:, k] = v[:, k] * s[k]
            s[k] = 1.0
        new_v = (w_sup + coef * u_sup) / s[sup_cols]
        v[sup_rows, sup_cols] = new_v

        # Keep the column norms and maxima up to date.
        self.sq_norms += np.bincount(sup_cols, weights=new_v.astype(np.float32).astype(np.float64) ** 2 - old_sq,
                                     minlength=n_cells)  # fmt: skip
        new_abs = np.abs(v[sup_rows, sup_cols]).astype(np.float64)
        sup_max = np.zeros(n_cells)
        np.maximum.at(sup_max, sup_cols, new_abs)
        took_over = winners[sup_max[winners] >= self.col_max[winners]]
        lost_max = winners[in_support & ~np.isin(winners, took_over)]
        self.stale[lost_max] = True
        is_new_max = (new_abs == sup_max[sup_cols]) & np.isin(sup_cols, took_over)
        self.max_row[sup_cols[is_new_max]] = sup_rows[is_new_max]
        self.col_max[took_over] = sup_max[took_over]
        for k in fold:
            self.sq_norms[k] = float(np.dot(v[:, k].astype(np.float64), v[:, k]))
            self._refresh_column(int(k))
        return energy


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
        state = _ScaledColumns(np.array(self.weights.T, dtype=np.float32, order="C"))

        n = len(inputs)
        n_batches = -(-n // batch_size)
        total_steps = max(epochs * n_batches, 1)
        step = 0
        for epoch in range(len(self.history), len(self.history) + epochs):
            start, energy = time.perf_counter(), 0.0
            state.refresh_norms()
            order = rng.permutation(n)
            for b in range(n_batches):
                batch = order[b * batch_size : (b + 1) * batch_size]
                cols = active_columns(inputs[batch], self.vocab_size)
                active = cols >= 0
                cols = np.where(active, cols, 0)
                x = np.where(active, inv_p[cols], 0).astype(np.float32)  # (B, w)
                energy += state.step(cols, active, x, lr * (1 - step / total_steps))
                step += 1
            self.history.append(
                {"epoch": epoch, "energy": energy / max(n, 1), "seconds": time.perf_counter() - start}
            )
            log.info("epoch %d  energy %.4g  %.1fs", epoch, energy / max(n, 1), self.history[-1]["seconds"])
        self.weights = state.weights().T
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
