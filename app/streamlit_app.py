"""Step 7: type a Thai word, see nearest neighbours from FlyVec and word2vec side by side,
plus the word's hash code.

    streamlit run app/streamlit_app.py

Needs a processed corpus (`data/processed/vocab.tsv`) and models trained with
`scripts/train_flyvec.py` and `scripts/train_baselines.py`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from thai_flyvec import pipeline  # noqa: E402
from thai_flyvec.evaluate import normalize_word  # noqa: E402
from thai_flyvec.flyvec import HASH_LENGTHS  # noqa: E402
from thai_flyvec.preprocess import load_vocab  # noqa: E402

MODELS_DIR = ROOT / "models"
VOCAB_PATH = ROOT / "data" / "processed" / "vocab.tsv"
MODEL_NAME = re.compile(r"flyvec_w(\d+)_s(\d+)\.npz")


def available_settings() -> list[tuple[int, int]]:
    """(w, seed) pairs that have both a FlyVec and a word2vec model."""
    settings = []
    for path in sorted(MODELS_DIR.glob("flyvec_w*_s*.npz")):
        match = MODEL_NAME.fullmatch(path.name)
        if match:
            w, seed = int(match[1]), int(match[2])
            if pipeline.model_path(MODELS_DIR, "word2vec", w, seed).exists():
                settings.append((w, seed))
    return settings


@st.cache_resource
def load_models(w: int, seed: int):
    words, _ = load_vocab(VOCAB_PATH)
    flyvec = pipeline.load_flyvec(MODELS_DIR, w, seed)
    w2v = pipeline.keyed_embedding(pipeline.load_keyed_vectors(MODELS_DIR, "word2vec", w, seed), words, "word2vec")
    return words, flyvec, w2v


def code_image(code: np.ndarray, width: int = 20) -> np.ndarray:
    """The 0/1 code as a grid of Kenyon cells, active cells dark, scaled up for display."""
    padded = np.zeros(-(-len(code) // width) * width, dtype=np.float32)
    padded[: len(code)] = code
    grid = 1 - padded.reshape(-1, width)
    return np.kron(grid, np.ones((12, 12), dtype=np.float32))


st.set_page_config(page_title="thai-flyvec", layout="wide")
st.title("แมลงวันอ่านภาษาไทยได้ไหม?")
st.caption("FlyVec: เวกเตอร์คำแบบไบนารีเบาบางจากสมองแมลงวันผลไม้ เทียบกับ word2vec")

settings = available_settings()
if not VOCAB_PATH.exists() or not settings:
    st.warning(
        "ยังไม่พบโมเดล — รัน `scripts/preprocess_corpus.py`, `scripts/train_flyvec.py` "
        "และ `scripts/train_baselines.py` ก่อน (ดู README)"
    )
    st.stop()

with st.sidebar:
    w, seed = st.selectbox("โมเดล (w, seed)", settings, format_func=lambda s: f"w = {s[0]}, seed = {s[1]}")
    k = st.select_slider("ความยาว hash (k)", options=list(HASH_LENGTHS), value=32)
    topn = st.slider("จำนวนคำใกล้เคียง", 5, 30, 10)

words, flyvec, w2v = load_models(w, seed)
fly = pipeline.flyvec_embedding(flyvec, words, k)

word = st.text_input("คำภาษาไทย", value="แมว")
query = normalize_word(word) if word.strip() else ""
if not query:
    st.stop()
if query not in fly:
    st.error(f"“{word}” ไม่อยู่ในคลังศัพท์ {len(words):,} คำ (OOV) — ลองคำอื่น")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader(f"FlyVec (k = {k})")
    neighbours = fly.nearest(query, topn)
    st.table({"คำ": [n for n, _ in neighbours], "cosine": [f"{s:.3f}" for _, s in neighbours]})
with right:
    st.subheader("word2vec (300 มิติ)")
    neighbours = w2v.nearest(query, topn)
    st.table({"คำ": [n for n, _ in neighbours], "cosine": [f"{s:.3f}" for _, s in neighbours]})

code = fly.get(query)
st.subheader(f"hash code ของ “{query}”")
st.caption(f"{flyvec.n_cells} Kenyon cells ทำงาน {int(code.sum())} เซลล์ (ช่องสีเข้ม)")
st.image(code_image(code), clamp=True)
st.code(" ".join(str(i) for i in np.flatnonzero(code)), language=None)
