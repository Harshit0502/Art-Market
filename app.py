import gradio as gr
import joblib
from model_wrapper import ArtMarketModel  # make sure this file is in the repo

# -----------------------
# Load Wrapper
# -----------------------
wrapper = joblib.load("art_market_model.joblib")
wrapper.load()   # IMPORTANT: load YOLO + price model

# -----------------------
# Predict Function
# -----------------------
def predict(img):
    result = wrapper.predict(img)
    return (
        result["class"],
        f"{result['confidence']:.2f}",
        f"{result['price']:,.0f}",
        result["market_explanation"],
    )

# -----------------------
# Gradio Interface
# -----------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload Artwork"),
    outputs=[
        gr.Textbox(label="Predicted Style"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Estimated Price (₹)"),
        gr.Textbox(label="Market Explanation"),
    ],
    title="Indian Art Market Classifier",
    description="Upload an artwork image → get style, price estimate, and cultural market context.",
)

if __name__ == "__main__":
    # Locally use share=True; on Hugging Face Spaces, just `demo.launch()`
    demo.launch()
