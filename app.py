import io
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_NAME = "sales_prediction_model.pkl"
FEATURES = ["youtube", "facebook", "newspaper"]
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Sales Predictor", page_icon="📈", layout="centered")


def find_model_file():
    """Look for the model next to app.py (and in the current folder), ignoring case."""
    for folder in (APP_DIR, Path.cwd()):
        for p in folder.iterdir():
            if p.is_file() and p.name.lower() == MODEL_NAME:
                return p
    return None


def load_from_bytes(data: bytes):
    # The model was saved with joblib, so try that first
    import joblib
    try:
        return joblib.load(io.BytesIO(data))
    except Exception:
        return pickle.loads(data)


@st.cache_resource
def load_model_from_path(path: str):
    return load_from_bytes(Path(path).read_bytes())


st.title("📈 Sales Prediction App")
st.write("Enter your advertising budgets to predict expected sales.")

model = None
model_path = find_model_file()

if model_path is not None:
    try:
        model = load_model_from_path(str(model_path))
    except Exception as e:
        st.error(f"Found '{model_path.name}' but could not load it: {e}")
        st.stop()
else:
    st.warning(
        f"Could not find '{MODEL_NAME}' in:\n\n- {APP_DIR}\n- {Path.cwd()}\n\n"
        "Move the file there, or upload it below."
    )
    uploaded = st.file_uploader("Upload your model file", type=["pkl", "PKL", "joblib"])
    if uploaded is None:
        st.stop()
    try:
        model = load_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not load the uploaded model: {e}")
        st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    youtube = st.number_input("YouTube budget", min_value=0.0, value=100.0, step=10.0)
with col2:
    facebook = st.number_input("Facebook budget", min_value=0.0, value=50.0, step=10.0)
with col3:
    newspaper = st.number_input("Newspaper budget", min_value=0.0, value=30.0, step=10.0)

if st.button("Predict Sales", type="primary"):
    input_df = pd.DataFrame([[youtube, facebook, newspaper]], columns=FEATURES)
    try:
        result = float(model.predict(input_df)[0])
        st.success(f"Predicted Sales: **{result:,.2f}**")
        st.subheader("Input summary")
        st.dataframe(input_df, hide_index=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

