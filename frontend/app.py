import base64
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
BACKGROUND = Path(__file__).parent / "assets" / "garden.jpg"

LEAF = "#3E5529"
GRASS = "#6E9E45"
PINK = "#EE9DB5"
PEACH = "#F3C47A"
LILAC = "#C9A7E0"

SPECIES = {
    "setosa": {"emoji": "🌷", "color": PINK, "label": "Iris setosa"},
    "versicolor": {"emoji": "🌼", "color": PEACH, "label": "Iris versicolor"},
    "virginica": {"emoji": "🌸", "color": LILAC, "label": "Iris virginica"},
}

st.set_page_config(page_title="Iris Garden", page_icon="🌷", layout="centered")


def background_css():
    if not BACKGROUND.exists():
        return "background: linear-gradient(180deg, #EAF4FC 0%, #F3F9EC 60%, #A9CF7C 100%);"
    encoded = base64.b64encode(BACKGROUND.read_bytes()).decode()
    return (
        f"background: url('data:image/jpeg;base64,{encoded}') center bottom / cover "
        "no-repeat fixed;"
    )


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&display=swap');

    html, body, [class*="css"], .stApp, button, input {{
        font-family: 'Nunito', sans-serif;
    }}

    /* Whole page: the garden picture */
    .stApp {{
        {background_css()}
    }}

    /* Transparent top bar */
    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Main content: soft white card, so text is readable over the picture */
    .block-container {{
        background: rgba(255, 255, 255, 0.82);
        border-radius: 28px;
        padding: 2.2rem 2.4rem 2.6rem;
        margin-top: 2.5rem;
        margin-bottom: 4rem;
        box-shadow: 0 10px 30px rgba(62, 85, 41, 0.15);
        backdrop-filter: blur(4px);
    }}

    h1, h2, h3 {{
        color: {LEAF} !important;
        font-weight: 800 !important;
    }}

    /* Sidebar: like a green garden fence */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #E4F0D6 0%, #CFE5B4 100%);
        border-right: 3px solid {GRASS};
    }}

    /* Predict button: green leaf, pink on hover */
    .stButton > button {{
        background: {GRASS};
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 2.2rem;
        font-weight: 800;
        font-size: 1.05rem;
        box-shadow: 0 4px 0 #557C34;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        background: {PINK};
        color: white;
        box-shadow: 0 4px 0 #D47896;
    }}
    .stButton > button:active {{
        transform: translateY(3px);
        box-shadow: 0 1px 0 #557C34;
    }}

    /* Result card */
    .result-card {{
        background: linear-gradient(135deg, #FFF5F8 0%, #FDF1DE 100%);
        border: 2px solid {PINK};
        border-radius: 22px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0 1.2rem;
        text-align: center;
    }}
    .result-card .flower {{ font-size: 3rem; line-height: 1; }}
    .result-card .name {{ font-size: 1.6rem; font-weight: 800; color: {LEAF}; }}
    .result-card .conf {{ color: #6B7F57; }}

    /* Probability bars: flower stems */
    .bar-row {{ margin: 0.45rem 0; }}
    .bar-label {{ display: flex; justify-content: space-between; color: {LEAF}; font-weight: 600; }}
    .bar-track {{ background: #E4F0D6; border-radius: 999px; height: 16px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}

    .footer {{ text-align: center; color: #6B7F57; font-size: 0.85rem; margin-top: 1.5rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🌷 Iris Garden")
st.write(
    "Move the sliders to describe your flower and press **Predict** — "
    "the model will tell you which iris grows in your garden."
)


with st.sidebar:
    st.header("🌱 Service status")
    st.caption(f"API: `{API_URL}`")
    try:
        requests.get(f"{API_URL}/", timeout=3).raise_for_status()
        st.success("API is running")
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        st.header("🪴 Model (MLflow)")
        st.write(f"**Name:** {info.get('model_name')}")
        st.write(f"**Version:** {info.get('model_version')} (alias `{info.get('alias')}`)")
        st.write(f"**Type:** {info.get('model_type')}")
        metrics = info.get("metrics", {})
        if metrics:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            col2.metric("F1", f"{metrics.get('f1_macro', 0):.3f}")
            st.metric("CV F1 (5-fold)", f"{metrics.get('cv_f1_mean', 0):.3f}")
    except requests.exceptions.RequestException:
        st.error("API is not available. Start it with `uvicorn main:app`.")


st.subheader("🌿 Flower measurements (cm)")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal width", 2.0, 4.5, 3.5, 0.1)
with col2:
    petal_length = st.slider("Petal length", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal width", 0.1, 2.5, 0.2, 0.1)

payload = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width,
}


def probability_bars(proba):
    rows = []
    for cls, p in sorted(proba.items(), key=lambda kv: -kv[1]):
        style = SPECIES.get(cls, {"emoji": "🌸", "color": PINK})
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label"><span>{style['emoji']} {cls}</span><span>{p:.1%}</span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{p * 100:.1f}%; background:{style['color']};"></div>
              </div>
            </div>
            """
        )
    return "".join(rows)


if st.button("🌷 Predict", type="primary"):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        species = result["prediction"]
        proba = result["probabilities"]
        style = SPECIES.get(species, {"emoji": "🌸", "label": species})

        st.markdown(
            f"""
            <div class="result-card">
              <div class="flower">{style['emoji']}</div>
              <div class="name">{style['label']}</div>
              <div class="conf">confidence {proba[species]:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("🌼 Class probabilities")
        st.markdown(probability_bars(proba), unsafe_allow_html=True)

        st.caption(f"Model: {result.get('model_name')} v{result.get('model_version')}")
        with st.expander("Request and response (JSON)"):
            st.write("Request:")
            st.json(payload)
            st.write("Response:")
            st.json(result)
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure FastAPI is running.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} {e.response.text}")

st.markdown(
    '<div class="footer">🌱 FastAPI · MLflow · Streamlit · Docker 🌱</div>',
    unsafe_allow_html=True,
)
