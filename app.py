import streamlit as st
from pathlib import Path

from src.bundle_loader import load_bundle
from src.ui_tabs import render_tab_client_recommender


# -----------------------------
# Load Eleven CSS (UI only)
# -----------------------------
def load_css(path: str):
    css = Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="Eleven — Recommender System",
    layout="wide",
)

load_css("eleven_theme.css")

st.title("Eleven — Recommender System")
st.caption("Business UI • Client recommendations powered by your trained pipeline")

# -----------------------------
# Load bundle (unchanged)
# -----------------------------
bundle = load_bundle(".")

# -----------------------------
# Main business UI (unchanged)
# -----------------------------
render_tab_client_recommender(bundle)
