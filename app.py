import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

IMG_SIZE = (150, 150)                            
PREPROCESS = "rescale"                             
THRESHOLD = 0.5                                     
LOW_CONFIDENCE = 0.65               



LABELS = {
    "O": {"name": "Organic", "emoji": "🍃", "tip": "Compost it or put it in the organic waste bin."},
    "R": {"name": "Recyclable", "emoji": "♻️", "tip": "Rinse it and put it in the recycling bin."},
}

APP_TITLE = "Waste Classifier"
APP_SUBTITLE = "Upload or snap a photo, and the model will tell you whether it is organic or recyclable."
GITHUB_URL = ""  
# ══════════════════════════════════════════════════════════

st.set_page_config(page_title=APP_TITLE, page_icon="♻️", layout="centered")

# ── Styling (works in light & dark mode) ──
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 760px;}
    .hero {text-align: center; margin-bottom: 1.2rem;}
    .hero h1 {font-size: 2.4rem; margin-bottom: 0.2rem;}
    .hero p {opacity: 0.75; font-size: 1.05rem;}
    .result-card {
        border-radius: 18px; padding: 1.4rem 1.6rem; margin-top: 1rem;
        text-align: center; border: 1px solid rgba(128,128,128,0.25);
    }
    .result-O {background: rgba(46, 160, 67, 0.14);}
    .result-R {background: rgba(31, 111, 235, 0.14);}
    .result-card .emoji {font-size: 3rem; line-height: 1;}
    .result-card .label {font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0;}
    .result-card .tip {opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
       path = hf_hub_download(repo_id="amany30/waste_Classifier", filename="O_R_model_small.keras" )
       return tf.keras.models.load_model(path)


def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize(IMG_SIZE), dtype="float32")
    if PREPROCESS == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img: Image.Image):
    p = float(model.predict(preprocess(img), verbose=0)[0][0])  # probability of class R
    key = "R" if p >= THRESHOLD else "O"
    confidence = p if key == "R" else 1 - p
    return key, confidence


# ── Sidebar ──
with st.sidebar:
    st.header("About")
    st.write(
        "A binary image classifier built with **Transfer Learning** on **VGG16** "
        "(TensorFlow / Keras) to separate organic waste from recyclable waste."
    )
    st.subheader("How it works")
    st.markdown(
        "1. The image is resized and normalised\n"
        "2. VGG16 extracts visual features\n"
        "3. A small dense head outputs a probability\n"
        "4. Probability ≥ 0.5 → Recyclable, otherwise Organic"
    )
    st.subheader("Limitations")
    st.caption(
        "The model only knows two classes. If you upload something that is not waste "
        "(a cat, a car…), it will still answer O or R."
    )
    if GITHUB_URL:
        st.link_button("View on GitHub", GITHUB_URL)

# ── Header ──
st.markdown(
    f'<div class="hero"><h1>♻️ {APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>',
    unsafe_allow_html=True,
)

# ── Load model (with a friendly error) ──
try:
    model = load_model()
except Exception as e:
    st.error("Could not load the model from Hugging Face. Check the repo and file name in load_model().")
    st.caption(str(e))
    st.stop()

# ── Input ──
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Use camera"])
with tab_upload:
    uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with tab_camera:
    snapshot = st.camera_input("Take a photo", label_visibility="collapsed")

source = uploaded or snapshot

# ── Result ──
if source is not None:
    image = Image.open(source)
    st.image(image, use_container_width=True)

    with st.spinner("Analysing…"):
        key, conf = predict(model, image)

    info = LABELS[key]
    st.markdown(
        f"""
        <div class="result-card result-{key}">
            <div class="emoji">{info['emoji']}</div>
            <div class="label">{info['name']}</div>
            <div class="tip">{info['tip']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf:.1%}")

    if conf < LOW_CONFIDENCE:
        st.warning("The model is not very sure about this one. Try a clearer photo with a plain background.")
else:
    st.info("👆 Upload a photo or use your camera to get started.")

st.divider()
st.caption("Built with TensorFlow, VGG16 & Streamlit")
