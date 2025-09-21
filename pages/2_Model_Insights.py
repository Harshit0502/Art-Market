import streamlit as st
import json
import numpy as np

st.title("📊 Model Insights")

# Load fallback pricing
with open("fallback_prices.json", "r") as f:
    fallback = json.load(f)

st.subheader("Fallback Price Mapping")
st.json(fallback)

# Show conformal calibration intervals
conformal = np.load("conformal_calib.npz")
st.subheader("Conformal Calibration Intervals")
st.write("Lower bounds (mean):", float(conformal["lower"].mean()))
st.write("Upper bounds (mean):", float(conformal["upper"].mean()))

# Manual test
st.subheader("🔎 Test Price Model (Manual Features)")
features = st.text_input("Enter 10 comma-separated features (e.g., 0.2,0.5,...,0.1)")

if features:
    try:
        import joblib
        price_model = joblib.load("rf_price.joblib")

        vec = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
        price = price_model.predict(vec)[0]
        st.success(f"Predicted Price: ₹{int(price):,}")
    except Exception as e:
        st.error(f"Error: {e}")
