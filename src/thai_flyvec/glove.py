"""GloVe word embeddings (Pennington, Socher & Manning, EMNLP 2014).

Builds the symmetric 1/distance word-word co-occurrence matrix within sentence
boundaries (dropping OOV tokens before windowing, like gensim's baselines) and
trains 300-dim vectors `W + W_tilde` with AdaGrad. Uses a small cached C/OpenMP
kernel via `gcc` + `ctypes` when available, with an identical NumPy fallback.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .inputs import OOV

log = logging.getLogger(__name__)

_C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

double glove_epoch(
    int64_t nnz, int dim, float lr,
    const int32_t *rows, const int32_t *cols,
    const float *fw, const float *logx,
    const int64_t *order,
    float *W, float *Wt, float *b, float *bt,
    float *gW, float *gWt, float *gb, float *gbt,
    int n_threads
) {
    double total_cost = 0.0;
    #pragma omp parallel for num_threads(n_threads) reduction(+:total_cost) schedule(static) if(n_threads > 1)
    for (int64_t idx = 0; idx < nnz; ++idx) {
        int64_t p = order[idx];
        int32_t i = rows[p];
        int32_t j = cols[p];
        float f = fw[p];
        float lx = logx[p];
        float *wi = W + (int64_t)i * dim;
        float *wj = Wt + (int64_t)j * dim;
        float *gwi = gW + (int64_t)i * dim;
        float *gwj = gWt + (int64_t)j * dim;
        float dot = b[i] + bt[j];
        for (int d = 0; d < dim; ++d) dot += wi[d] * wj[d];
        float diff = dot - lx;
        float fdiff = f * diff;
        total_cost += 0.5 * (double)fdiff * (double)diff;
        fdiff *= lr;
        for (int d = 0; d < dim; ++d) {
            float temp1 = fdiff * wj[d];
            float temp2 = fdiff * wi[d];
            wi[d] -= temp1 / sqrtf(gwi[d]);
            wj[d] -= temp2 / sqrtf(gwj[d]);
            gwi[d] += temp1 * temp1;
            gwj[d] += temp2 * temp2;
        }
        b[i] -= fdiff / sqrtf(gb[i]);
        bt[j] -= fdiff / sqrtf(gbt[j]);
        gb[i] += fdiff * fdiff;
        gbt[j] += fdiff * fdiff;
    }
    return total_cost;
}
"""

_C_LIB = None
_C_FUNC = None
_DLL_DIRS: list = []


def _load_c_kernel():
    """Compile (once, cached in temp) and return the C `glove_epoch` function, or None."""
    global _C_LIB, _C_FUNC
    if _C_FUNC is not None:
        return _C_FUNC

    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        return None

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _DLL_DIRS.append(os.add_dll_directory(str(Path(compiler).resolve().parent)))

    cache_dir = Path(tempfile.gettempdir()) / "thai_flyvec_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_C_SOURCE.encode("utf-8")).hexdigest()[:12]
    ext = ".dll" if os.name == "nt" else ".so"
    lib_path = cache_dir / f"glove_{digest}{ext}"

    if not lib_path.exists():
        c_tmp = cache_dir / f"glove_{digest}_{os.getpid()}.c"
        lib_tmp = cache_dir / f"glove_{digest}_{os.getpid()}{ext}"
        try:
            c_tmp.write_text(_C_SOURCE, encoding="utf-8")
            base_flags = ["-O3", "-shared"] + ([] if os.name == "nt" else ["-fPIC"])
            compiled = False
            for extra in (["-fopenmp"], []):
                cmd = [compiler, *base_flags, *extra, str(c_tmp), "-o", str(lib_tmp), "-lm"]
                res = subprocess.run(cmd, capture_output=True)
                if res.returncode == 0 and lib_tmp.exists():
                    compiled = True
                    break
            if not compiled:
                return None
            os.replace(lib_tmp, lib_path)
        except OSError:
            return None
        finally:
            c_tmp.unlink(missing_ok=True)
            lib_tmp.unlink(missing_ok=True)

    try:
        lib = ctypes.CDLL(str(lib_path))
        fn = lib.glove_epoch
        c_f = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        c_i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        c_i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        fn.restype = ctypes.c_double
        fn.argtypes = [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_float,
            c_i32,
            c_i32,
            c_f,
            c_f,
            c_i64,
            c_f,
            c_f,
            c_f,
            c_f,
            c_f,
            c_f,
            c_f,
            c_f,
            ctypes.c_int,
        ]
    except (OSError, AttributeError):
        return None

    _C_LIB = lib
    _C_FUNC = fn
    return _C_FUNC


def _epoch_numpy(nnz, dim, lr, rows, cols, fw, logx, order, W, Wt, b, bt, gW, gWt, gb, gbt) -> float:
    total_cost = 0.0
    lr32 = np.float32(lr)
    for idx in range(nnz):
        p = int(order[idx])
        i = int(rows[p])
        j = int(cols[p])
        f = fw[p]
        lx = logx[p]
        wi = W[i]
        wj = Wt[j]
        gwi = gW[i]
        gwj = gWt[j]
        dot = np.float32(b[i] + bt[j]) + np.dot(wi, wj)
        diff = np.float32(dot - lx)
        fdiff = np.float32(f * diff)
        total_cost += 0.5 * float(fdiff) * float(diff)
        fdiff = np.float32(fdiff * lr32)
        temp1 = fdiff * wj
        temp2 = fdiff * wi
        wi -= temp1 / np.sqrt(gwi)
        wj -= temp2 / np.sqrt(gwj)
        gwi += temp1 * temp1
        gwj += temp2 * temp2
        b[i] -= fdiff / np.sqrt(gb[i])
        bt[j] -= fdiff / np.sqrt(gbt[j])
        sq = fdiff * fdiff
        gb[i] += sq
        gbt[j] += sq
    return total_cost


def cooccurrence_matrix(
    corpus_or_sentences,
    window: int = 11,
    vocab_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric 1/distance word-word co-occurrence counts within sentences.

    Accepts a `preprocess.Corpus` or an iterable of 1D integer token-id arrays
    (with OOV = -1 and explicit `vocab_size`). OOV tokens are dropped before
    sliding the window of radius `window // 2`, matching gensim's baselines.
    Returns contiguous `(rows, cols, counts)` with dtypes `(int32, int32, float32)`.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be odd and at least 3, got {window}")
    half = window // 2

    if hasattr(corpus_or_sentences, "ids") and hasattr(corpus_or_sentences, "offsets"):
        ids = np.asarray(corpus_or_sentences.ids, dtype=np.int32)
        offsets = np.asarray(corpus_or_sentences.offsets, dtype=np.int64)
        if vocab_size is None:
            vocab_size = len(corpus_or_sentences.words)
        sent_id = np.repeat(np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets))
    else:
        if vocab_size is None or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer for sentence iterables")
        chunks, sids = [], []
        for sid, s in enumerate(corpus_or_sentences):
            arr = np.asarray(s, dtype=np.int32)
            if len(arr):
                chunks.append(arr)
                sids.append(np.full(len(arr), sid, dtype=np.int32))
        if not chunks:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        ids = np.concatenate(chunks)
        sent_id = np.concatenate(sids)

    keep = ids != OOV
    ids_k = ids[keep]
    sent_k = sent_id[keep]
    if len(ids_k) < 2:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    cooc = sparse.csr_matrix((vocab_size, vocab_size), dtype=np.float64)
    for d in range(1, half + 1):
        if d >= len(ids_k):
            break
        same = sent_k[:-d] == sent_k[d:]
        if not np.any(same):
            continue
        a = ids_k[:-d][same]
        b = ids_k[d:][same]
        w = np.full(len(a), 1.0 / d, dtype=np.float64)
        m = sparse.coo_matrix((w, (a, b)), shape=(vocab_size, vocab_size)).tocsr()
        cooc = cooc + m + m.T

    cooc.sum_duplicates()
    cooc.eliminate_zeros()
    coo = cooc.tocoo()
    return (
        np.ascontiguousarray(coo.row, dtype=np.int32),
        np.ascontiguousarray(coo.col, dtype=np.int32),
        np.ascontiguousarray(coo.data, dtype=np.float32),
    )


class GloVe:
    def __init__(self, vocab_size: int, dim: int = 300, seed: int = 0):
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.seed = int(seed)
        self.vectors: np.ndarray | None = None
        self.history: list[dict] = []
        self.x_max: float = 100.0
        self.alpha: float = 0.75
        self.lr: float = 0.05

    def fit(
        self,
        cooc: tuple[np.ndarray, np.ndarray, np.ndarray],
        epochs: int = 50,
        lr: float = 0.05,
        x_max: float = 100.0,
        alpha: float = 0.75,
        workers: int = 4,
        use_c: bool | None = None,
    ) -> GloVe:
        """Train GloVe with AdaGrad on `(rows, cols, counts)` from `cooccurrence_matrix`."""
        rows, cols, counts = cooc
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        counts = np.ascontiguousarray(counts, dtype=np.float32)

        self.x_max = float(x_max)
        self.alpha = float(alpha)
        self.lr = float(lr)

        V, D = self.vocab_size, self.dim
        rng = np.random.default_rng(self.seed)
        W = np.ascontiguousarray(rng.uniform(-0.5, 0.5, (V, D)) / D, dtype=np.float32)
        Wt = np.ascontiguousarray(rng.uniform(-0.5, 0.5, (V, D)) / D, dtype=np.float32)
        b = np.ascontiguousarray(rng.uniform(-0.5, 0.5, V) / D, dtype=np.float32)
        bt = np.ascontiguousarray(rng.uniform(-0.5, 0.5, V) / D, dtype=np.float32)
        gW = np.ones((V, D), dtype=np.float32)
        gWt = np.ones((V, D), dtype=np.float32)
        gb = np.ones(V, dtype=np.float32)
        gbt = np.ones(V, dtype=np.float32)

        nnz = len(counts)
        fw = np.ascontiguousarray(np.minimum((counts / x_max) ** alpha, 1.0), dtype=np.float32)
        logx = np.ascontiguousarray(np.log(counts), dtype=np.float32)

        c_func = None
        if use_c is not False:
            c_func = _load_c_kernel()
            if use_c is True and c_func is None:
                raise RuntimeError("C compiler (gcc) unavailable for GloVe kernel")

        self.history = []
        for ep in range(1, epochs + 1):
            start = time.perf_counter()
            order = np.ascontiguousarray(rng.permutation(nnz), dtype=np.int64)
            if nnz == 0:
                cost = 0.0
            elif c_func is not None:
                cost = float(
                    c_func(
                        nnz, D, float(lr),
                        rows, cols, fw, logx, order,
                        W, Wt, b, bt, gW, gWt, gb, gbt,
                        max(1, int(workers)),
                    )
                )
            else:
                cost = _epoch_numpy(nnz, D, lr, rows, cols, fw, logx, order, W, Wt, b, bt, gW, gWt, gb, gbt)
            elapsed = time.perf_counter() - start
            self.history.append({"epoch": ep, "loss": cost / max(nnz, 1), "seconds": elapsed})
            log.info("glove epoch %d  loss %.5f  %.1fs", ep, self.history[-1]["loss"], elapsed)

        self.vectors = np.ascontiguousarray(W + Wt, dtype=np.float32)
        return self

    def save(self, path: str | Path) -> Path:
        if self.vectors is None:
            raise RuntimeError("model is not trained")
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=np.asarray(self.vectors, dtype=np.float32),
            vocab_size=self.vocab_size,
            dim=self.dim,
            seed=self.seed,
            x_max=self.x_max,
            alpha=self.alpha,
            lr=self.lr,
            loss=np.array([h["loss"] for h in self.history]),
            seconds=np.array([h["seconds"] for h in self.history]),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> GloVe:
        with np.load(path) as data:
            model = cls(int(data["vocab_size"]), int(data["dim"]), int(data["seed"]))
            model.vectors = data["vectors"]
            model.x_max = float(data["x_max"])
            model.alpha = float(data["alpha"])
            model.lr = float(data["lr"])
            model.history = [
                {"epoch": i + 1, "loss": float(loss), "seconds": float(sec)}
                for i, (loss, sec) in enumerate(zip(data["loss"], data["seconds"]))
            ]
        return model
