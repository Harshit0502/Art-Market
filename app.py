import streamlit as st

st.set_page_config(
    page_title="Art Market Prediction",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Indian Art Market Model")
st.markdown("""
This app helps **classify traditional Indian artworks** and estimate their **market price** using ML models.
""")

st.sidebar.success("Choose a page above 👆")
