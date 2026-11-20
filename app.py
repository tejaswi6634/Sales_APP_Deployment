
import io
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

MODEL_NAME = "sales_prediction_model.pkl"
FEATURES = ["youtube", "facebook", "newspaper"]
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AdVision | Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- MODEL LOADING ----------------

def find_model_file():
    for folder in (APP_DIR, Path.cwd()):
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and p.name.lower() == MODEL_NAME:
                    return p
    return None


def load_from_bytes(data):
    import joblib
    try:
        return joblib.load(io.BytesIO(data))
    except Exception:
        return pickle.loads(data)


@st.cache_resource
def load_model_from_path(path):
    return load_from_bytes(Path(path).read_bytes())


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #080B16;
    color: #F4F5FF;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1250px;
}

#MainMenu, footer {
    visibility: hidden;
}

/* HERO */

.hero {
    background: linear-gradient(120deg, #171A35, #24204A, #111D35);
    border: 1px solid rgba(145,130,255,0.25);
    border-radius: 24px;
    padding: 42px 45px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    background: #825BFF;
    filter: blur(110px);
    opacity: 0.35;
    right: 5%;
    top: -80px;
}

.hero-tag {
    color: #A99AFF;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 46px;
    font-weight: 700;
    color: white;
    margin: 12px 0;
}

.gradient-text {
    background: linear-gradient(90deg, #A78BFA, #55D6FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #B4B8D0;
    font-size: 15px;
    margin-bottom: 0;
}

/* SECTION HEADINGS */

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 600;
    margin: 30px 0 7px 0;
    color: #F4F5FF;
}

.section-caption {
    color: #9299B5;
    font-size: 13px;
    margin-bottom: 20px;
}

/* INPUT CARDS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111629;
    border: 1px solid #252D47;
    border-radius: 18px;
    padding: 15px;
}

div[data-testid="stNumberInput"] label {
    color: #D5D8E8 !important;
    font-weight: 600;
}

div[data-testid="stNumberInput"] input {
    background: #0B1020 !important;
    color: white !important;
    border: 1px solid #333D5C !important;
    border-radius: 10px !important;
}

/* BUTTON */

.stButton > button {
    background: linear-gradient(90deg, #7958F5, #527CFF);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 15px 25px;
    height: 52px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(111,91,255,0.35);
    color: white;
}

/* RESULT CARD */

.result-card {
    background: linear-gradient(135deg, #192846, #25204A);
    border: 1px solid #6255B9;
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
}

.result-label {
    color: #B9B4ED;
    font-size: 13px;
    letter-spacing: 2px;
    font-weight: 700;
    text-transform: uppercase;
}

.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 46px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 12px 0;
}

.result-subtitle {
    color: #B7C1DA;
    font-size: 13px;
}

/* FOOTER */

.footer {
    border-top: 1px solid #252D47;
    margin-top: 55px;
    padding-top: 20px;
    text-align: center;
    color: #747D9B;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ Marketing Intelligence Platform</div>
    <h1>Ad<span class="gradient-text">Vision.</span></h1>
    <p>
        Turn advertising investments into intelligent sales predictions.
        <br>
        Powered by Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------

model = None
model_path = find_model_file()

if model_path:
    try:
        model = load_model_from_path(str(model_path))
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

else:
    st.warning(
        f"Model file '{MODEL_NAME}' was not found. "
        "Upload your trained model to continue."
    )

    uploaded = st.file_uploader(
        "Upload trained model",
        type=["pkl", "joblib"]
    )

    if uploaded is None:
        st.stop()

    try:
        model = load_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()


# ---------------- INPUT SECTION ----------------

st.markdown(
    '<div class="section-heading">01 / Campaign Configuration</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">Configure your advertising budget across different marketing channels.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 📺 YouTube")
        st.caption("Video advertising")
        youtube = st.number_input(
            "YouTube Budget",
            min_value=0.0,
            value=100.0,
            step=10.0,
            key="youtube"
        )
        st.caption("Reach audiences through video campaigns.")

with col2:
    with st.container(border=True):
        st.markdown("### 📱 Facebook")
        st.caption("Social media advertising")
        facebook = st.number_input(
            "Facebook Budget",
            min_value=0.0,
            value=50.0,
            step=10.0,
            key="facebook"
        )
        st.caption("Engage audiences through social campaigns.")

with col3:
    with st.container(border=True):
        st.markdown("### 📰 Newspaper")
        st.caption("Print advertising")
        newspaper = st.number_input(
            "Newspaper Budget",
            min_value=0.0,
            value=30.0,
            step=10.0,
            key="newspaper"
        )
        st.caption("Reach readers through print campaigns.")


# ---------------- BUDGET SUMMARY ----------------

total_budget = youtube + facebook + newspaper

st.markdown(
    '<div class="section-heading">02 / Investment Overview</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Budget", f"{total_budget:,.2f}")
m2.metric("YouTube", f"{youtube:,.2f}")
m3.metric("Facebook", f"{facebook:,.2f}")
m4.metric("Newspaper", f"{newspaper:,.2f}")


# ---------------- PREDICTION ----------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button("✦  GENERATE SALES PREDICTION", type="primary"):

    input_df = pd.DataFrame(
        [[youtube, facebook, newspaper]],
        columns=FEATURES
    )

    try:
        result = float(model.predict(input_df)[0])

        st.markdown("""
        <div class="result-card">
            <div class="result-label">✦ Your Predicted Sales</div>
            <div class="result-value">"""+f"{result:,.2f}"+"""</div>
            <div class="result-subtitle">
                Machine Learning prediction based on your campaign investment
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-heading">03 / Campaign Breakdown</div>',
            unsafe_allow_html=True
        )

        # Budget distribution chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["YouTube", "Facebook", "Newspaper"],
                    values=[youtube, facebook, newspaper],
                    hole=0.68,
                    textinfo="label+percent",
                    marker=dict(
                        colors=["#9575FF", "#39C6E8", "#FFB86B"]
                    )
                )
            ]
        )

        fig.update_layout(
            title="Advertising Budget Distribution",
            title_font_color="white",
            font_color="#E5E7F5",
            paper_bgcolor="#111629",
            plot_bgcolor="#111629",
            showlegend=False,
            height=380,
            margin=dict(t=60, b=20, l=20, r=20),
            annotations=[
                dict(
                    text="TOTAL<br>"+f"{total_budget:,.0f}",
                    x=0.5,
                    y=0.5,
                    font_size=20,
                    font_color="white",
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Prediction Input Details")
        st.dataframe(input_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    ADVISION © 2026 &nbsp; | &nbsp;
    BUILT WITH MACHINE LEARNING & STREAMLIT
    <br><br>
    Turning Data Into Decisions.
</div>
""", unsafe_allow_html=True)

        st.subheader("Input summary")
        st.dataframe(input_df, hide_index=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
