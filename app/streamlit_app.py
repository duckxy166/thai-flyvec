"""Step 7: type a Thai word, see nearest neighbours from FlyVec and word2vec side by side,
plus the word's hash code."""

import streamlit as st

st.title("แมลงวันอ่านภาษาไทยได้ไหม?")
word = st.text_input("คำภาษาไทย")
# TODO: load models, show neighbours from both, draw the hash code
