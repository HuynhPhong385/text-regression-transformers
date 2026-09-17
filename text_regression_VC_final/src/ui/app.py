"""Streamlit demo UI."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Text Regression Demo", layout="centered")
st.title("Text Regression — Sentiment Score [0, 1]")

from pathlib import Path
from src.inference.predictor import Predictor

# Discover experiments with checkpoints
artifacts = sorted([p.parent.parent.name for p in Path("artifacts").glob("*/checkpoints/best.pt")])
if not artifacts:
    st.warning("No checkpoint found. Run training first: `conda run -n base python -m src.training.train --config configs/bert_finetune.yaml`")
    st.stop()

choice = st.selectbox("Model", artifacts)
text = st.text_area("Enter review", height=120, placeholder="The movie was excellent and very entertaining.")

if st.button("Predict"):
    if not text.strip():
        st.error("Please enter a review.")
    else:
        ckpt_dir = Path(f"artifacts/{choice}/checkpoints")
        pred = Predictor(ckpt_dir)
        score = pred.predict(text)
        st.metric("Sentiment score", f"{score:.3f}")
        st.caption(f"Model: {pred.model_name} | Strategy: {pred.strategy} | Experiment: {choice}")
        # Simple gauge
        st.progress(score)
