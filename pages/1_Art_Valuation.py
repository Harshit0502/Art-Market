import streamlit as st
import torch
from ultralytics import YOLO
import joblib
import numpy as np
import json
from pathlib import Path
from PIL import Image

# Load models once
@st.cache_resource
def load_models():
    clf = YOLO("best.pt")  # classifier
    price_model = joblib.load("rf_price.joblib")
    conformal = np.load("conformal_calib.npz")
    fallback = json.load(open("fallback_prices.json", "r"))
    return clf, price_model, conformal, fallback

clf, price_model, conformal, fallback_prices = load_models()

st.title("🖼️ Artwork Classification & Valuation")

uploaded = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Artwork", use_column_width=True)

    # 1. Classification
    results = clf(image)
    pred_class = results.names[int(results.probs.top1)]
    st.subheader(f"🖌️ Predicted Art Style: **{pred_class}**")

    # 2. Price Prediction
    try:
        features = np.random.rand(1, 10)  # TODO: Replace with your actual feature extractor
        base_price = price_model.predict(features)[0]

        # Conformal calibration
        lo, hi = conformal["lower"], conformal["upper"]
        interval = (base_price + lo.mean(), base_price + hi.mean())

        st.metric(
            label="💰 Predicted Price",
            value=f"₹{int(base_price):,}",
            delta=f"Confidence: {int(interval[0]):,} – {int(interval[1]):,}"
        )
    except Exception:
        st.warning("⚠️ Model failed — using fallback pricing.")
        st.success(f"Fallback Price: ₹{fallback_prices.get(pred_class, 20000):,}")
