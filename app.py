import streamlit as st
from PIL import Image
import torch
import joblib
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms

# ----------------------------
# Set paths for model and resource files
# ----------------------------
MODEL_DIR = Path(".")
YOLO_MODEL_PATH = MODEL_DIR / "best.pt"
PRICE_MODEL_PATH = MODEL_DIR / "art_market_model.joblib"
RF_PRICE_MODEL_PATH = MODEL_DIR / "rf_price.joblib"
FALLBACK_PRICE_PATH = MODEL_DIR / "fallback_prices.json"
CONFORMAL_CALIB_PATH = MODEL_DIR / "conformal_calib.npz"

# ----------------------------
# Cached Loaders
# ----------------------------
@st.cache_resource
def load_yolo_model():
    if not YOLO_MODEL_PATH.exists():
        st.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
        return None
    return YOLO(str(YOLO_MODEL_PATH))

@st.cache_resource
def load_fallback_prices():
    if not FALLBACK_PRICE_PATH.exists():
        st.warning("Fallback prices file not found. Using defaults.")
        return {}
    with open(FALLBACK_PRICE_PATH, 'r') as f:
        return json.load(f)

@st.cache_resource
def load_price_models():
    try:
        ridge_model = joblib.load(PRICE_MODEL_PATH)
        rf_model = joblib.load(RF_PRICE_MODEL_PATH)
        return ridge_model, rf_model
    except Exception as e:
        st.error(f"Error loading price models: {e}")
        return None, None

@st.cache_resource
def load_conformal_calib():
    if not CONFORMAL_CALIB_PATH.exists():
        st.warning("Conformal calibration file not found.")
        return None
    return np.load(CONFORMAL_CALIB_PATH)

@st.cache_resource
def load_feature_extractor():
    import torchvision.models as models
    resnet = models.resnet18(pretrained=True)
    # Remove last classification layer
    modules = list(resnet.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules)
    feature_extractor.eval()
    return feature_extractor

# ----------------------------
# Image Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet means
                         std=[0.229, 0.224, 0.225])   # Imagenet stds
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0)

# ----------------------------
# Prediction Functions
# ----------------------------
def predict_art_style(image: Image.Image, yolo_model):
    if yolo_model is None:
        return "Unknown"

    results = yolo_model(image)
    if results and hasattr(results[0], "probs"):
        probs = results[0].probs
        class_idx = int(probs.top1)
        class_name = results[0].names[class_idx]
        return class_name
    return "Unknown"

def extract_features(image_tensor, feature_extractor):
    feature_extractor.eval()
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    # Flatten [1, 512, 1, 1] -> [1, 512]
    return features.squeeze().numpy().reshape(1, -1)

def predict_price(image: Image.Image, ridge_model, rf_model, fallback_prices, conformal_calib, yolo_model, feature_extractor):
    art_style = predict_art_style(image, yolo_model)

    image_tensor = preprocess_image(image)
    features = extract_features(image_tensor, feature_extractor)

    if ridge_model is None or rf_model is None:
        return fallback_prices.get(art_style, 1000), (500, 1500), art_style

    # Predict price
    ridge_pred = ridge_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    price_pred = (ridge_pred + rf_pred) / 2

    # Fallback if invalid
    if price_pred <= 0 or np.isnan(price_pred):
        price_pred = fallback_prices.get(art_style, 1000)

    # Conformal interval (placeholder logic)
    lower, upper = price_pred - 500, price_pred + 500
    return price_pred, (lower, upper), art_style

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="Art Style & Price Predictor", layout="centered")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Art Style Classification", "Price Prediction"])

    # Load models once
    yolo_model = load_yolo_model()
    fallback_prices = load_fallback_prices()
    ridge_model, rf_model = load_price_models()
    conformal_calib = load_conformal_calib()
    feature_extractor = load_feature_extractor()

    if page == "Home":
        st.title("Art Style and Price Prediction")
        st.write(
            """
            Welcome to the Art Style and Price Predictor app. 
            This app uses ML models to classify Indian art styles and predict artwork prices.
            Use the navigation pane to explore.
            """
        )

    elif page == "Art Style Classification":
        st.title("Art Style Classification")
        uploaded_file = st.file_uploader("Upload an artwork image", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict Art Style"):
                art_style = predict_art_style(image, yolo_model)
                st.success(f"Predicted Art Style: {art_style}")

    elif page == "Price Prediction":
        st.title("Artwork Price Prediction")
        uploaded_file = st.file_uploader("Upload an artwork image", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict Price and Explain Market"):
                price_pred, (lower, upper), art_style = predict_price(
                    image, ridge_model, rf_model, fallback_prices, conformal_calib, yolo_model, feature_extractor
                )
                st.success(f"Predicted Price: ₹{price_pred:,.2f}")
                st.info(f"Predicted Art Style: {art_style}")
                st.write(f"Estimated Price Range: ₹{lower:,.2f} - ₹{upper:,.2f}")

                # Example explanation
                st.write(f"The market for {art_style} style art is known to have high demand.")

if __name__ == "__main__":
    main()
