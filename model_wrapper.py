import joblib
from ultralytics import YOLO
from pathlib import Path

class ArtMarketModel:
    def __init__(self, clf_weights, price_model_path, fallback_prices, explanations):
        self.clf_weights = clf_weights
        self.price_model_path = price_model_path
        self.fallback_prices = fallback_prices
        self.explanations = explanations
        self._clf = None
        self._price_model = None

    def load(self):
        # Load classifier
        if self._clf is None:
            if Path(self.clf_weights).exists():
                self._clf = YOLO(self.clf_weights)
            else:
                print(f"[WARN] Weights not found at {self.clf_weights}, using pretrained yolov8n-cls.pt")
                self._clf = YOLO("yolov8n-cls.pt")

        # Load price model
        if self._price_model is None and Path(self.price_model_path).exists():
            self._price_model = joblib.load(self.price_model_path)

    def predict(self, img_path):
        # Classification
        res = self._clf.predict(source=img_path, imgsz=224, verbose=False)[0]
        probs = res.probs.data.cpu().numpy()
        top_idx = int(probs.argmax())
        cname = res.names[top_idx]
        conf = float(probs[top_idx])

        # Price
        if self._price_model is not None:
            # TODO: Replace with real feature extractor from training script
            price = float(self._price_model.predict([[0]*512])[0])  # placeholder
        else:
            price = self.fallback_prices.get(cname, 20000)

        expl = self.explanations.get(cname, "General art market info")
        return {
            "class": cname,
            "confidence": conf,
            "price": price,
            "market_explanation": expl,
        }
