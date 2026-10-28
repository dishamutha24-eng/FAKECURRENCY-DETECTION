from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "currency_resnet18.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Counterfeit", "Genuine"]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

st.set_page_config(
    page_title="CurrencyIQ | Fake Currency Detection",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# 2. CUSTOM INTERFACE
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #1a2a49 0%, transparent 40%),
        radial-gradient(ellipse at 90% 20%, #192b40 0%, transparent 35%),
        #0b1020;
    color: #f4f5fa;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #f4f5fa !important;
}

p, label {
    color: #e1e7f1;
}

.hero {
    padding: 38px;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    background: linear-gradient(125deg, #192744, #111827);
    margin-bottom: 28px;
}

.hero-label {
    color: #e9c46a;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-title {
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(32px, 5vw, 48px);
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 15px;
}

.hero-title span {
    color: #e9c46a;
}

.hero-subtitle {
    color: #b5c0d4;
    font-size: 16px;
    line-height: 1.7;
    max-width: 700px;
}

.feature-card {
    background: linear-gradient(145deg, #172139, #121a2b);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 22px;
    min-height: 135px;
    margin-bottom: 15px;
}

.feature-icon {
    font-size: 28px;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 17px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}

.feature-description {
    font-size: 13px;
    line-height: 1.6;
    color: #aab5ca;
}

.result-card {
    background: linear-gradient(135deg, #192744, #111827);
    border: 1px solid rgba(233,196,106,0.4);
    border-radius: 20px;
    padding: 25px;
    margin: 18px 0;
}

.result-label {
    color: #aab5ca;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-value {
    color: #e9c46a;
    font-family: 'Manrope', sans-serif;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

div.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: 1px solid #e9c46a;
    background: linear-gradient(100deg, #e9c46a, #f3d993);
    color: #101525;
    font-weight: 800;
}

div.stButton > button:hover {
    background: #f8e4a8;
    color: #101525;
    border-color: #ffffff;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.10);
}

.stTabs [data-baseweb="tab"] {
    background: #141c30;
    border-radius: 10px 10px 0 0;
    padding: 14px 22px;
    color: #c0cadb;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: #e9c46a !important;
    border-bottom: 2px solid #e9c46a !important;
}

[data-testid="stFileUploader"] {
    background: #141c30;
    border: 1px dashed rgba(233,196,106,0.4);
    border-radius: 15px;
    padding: 15px;
}

[data-testid="stMetric"] {
    background: #141c30;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #aab5ca !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

.footer {
    border-top: 1px solid rgba(255,255,255,0.10);
    margin-top: 40px;
    padding-top: 20px;
    color: #8491a8;
    text-align: center;
    font-size: 12px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 2),
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ==================================================
# 4. COMMON PREDICTION FUNCTION
# Both upload and webcam use this function.
# ==================================================

def predict_currency(image_rgb):
    image = Image.fromarray(image_rgb).convert("RGB")

    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    counterfeit_score = probabilities[0].item() * 100
    genuine_score = probabilities[1].item() * 100

    predicted_index = int(probabilities.argmax().item())
    prediction = CLASS_NAMES[predicted_index]

    return prediction, counterfeit_score, genuine_score


# ==================================================
# 5. DISPLAY PREDICTION RESULTS
# ==================================================

def show_result(prediction, counterfeit_score, genuine_score):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Analysis complete</div>
            <div class="result-value">{prediction.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Counterfeit score", f"{counterfeit_score:.2f}%")
        st.progress(
            min(max(counterfeit_score / 100, 0.0), 1.0)
        )

    with col2:
        st.metric("Genuine score", f"{genuine_score:.2f}%")
        st.progress(
            min(max(genuine_score / 100, 0.0), 1.0)
        )


# ==================================================
# 6. WEBCAM PROCESSOR
# ==================================================

class CurrencyCamera(VideoProcessorBase):

    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        self.latest_frame = image_bgr.copy()

        return av.VideoFrame.from_ndarray(
            image_bgr,
            format="bgr24",
        )


# ==================================================
# 7. LOAD MODEL AND SHOW HEADER
# ==================================================

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.markdown("""
<div class="hero">
    <div class="hero-label">Machine Learning · Computer Vision</div>
    <div class="hero-title">
        Currency<span>IQ</span><br>
        Fake Currency Detection
    </div>
    <div class="hero-subtitle">
        Analyze currency-note images using a trained ResNet18
        image classification model. Choose an image from your
        device or use your webcam.
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 8. FEATURE CARDS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🖼️</div>
        <div class="feature-title">Image Analysis</div>
        <div class="feature-description">
            Upload a saved currency image for classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📷</div>
        <div class="feature-title">Live Camera</div>
        <div class="feature-description">
            Use your webcam to capture a currency note.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Prediction Scores</div>
        <div class="feature-description">
            Compare the scores for both output classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## Analyze Your Currency")
st.caption("Choose an input method below.")


# ==================================================
# 9. UPLOAD AND CAMERA TABS
# ==================================================

upload_tab, camera_tab = st.tabs([
    "🖼️ Upload Image",
    "📷 Live Camera",
])


# --------------------------------------------------
# TAB 1: UPLOAD IMAGE
# --------------------------------------------------

with upload_tab:

    st.subheader("Upload a currency image")
    st.write(
        "Choose a JPG or PNG image. Use a clear image "
        "with the note fully visible."
    )

    uploaded_file = st.file_uploader(
        "Select image",
        type=["jpg", "jpeg", "png"],
        key="currency_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns([1.2, 1])

            with left:
                st.image(
                    uploaded_image,
                    caption="Selected currency image",
                    use_container_width=True,
                )

            with right:
                st.markdown("### Image ready")
                st.write(
                    f"Dimensions: {uploaded_image.width} × "
                    f"{uploaded_image.height}"
                )

                if st.button(
                    "✨ Analyze Image",
                    type="primary",
                    key="predict_upload",
                ):
                    image_rgb = np.array(uploaded_image)

                    with st.spinner("Analyzing image..."):
                        prediction, counterfeit_score, genuine_score = (
                            predict_currency(image_rgb)
                        )

                    show_result(
                        prediction,
                        counterfeit_score,
                        genuine_score,
                    )

        except Exception as error:
            st.error(f"Could not process the image: {error}")


# --------------------------------------------------
# TAB 2: WEBCAM
# --------------------------------------------------

with camera_tab:

    st.subheader("Analyze using your webcam")

    st.write(
        "Start the camera, position the note clearly, "
        "and wait for the image to become focused."
    )

    ctx = webrtc_streamer(
        key="currency-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CurrencyCamera,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor is not None:

        if st.button(
            "✨ Analyze Camera Image",
            type="primary",
            key="predict_camera",
        ):
            frame = ctx.video_processor.latest_frame

            if frame is None:
                st.warning(
                    "No camera frame received. Wait for the "
                    "camera feed and try again."
                )

            else:
                # Convert camera BGR frame to RGB.
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                st.image(
                    frame_rgb,
                    caption="Captured camera image",
                    use_container_width=True,
                )

                with st.spinner("Analyzing camera image..."):
                    prediction, counterfeit_score, genuine_score = (
                        predict_currency(frame_rgb)
                    )

                show_result(
                    prediction,
                    counterfeit_score,
                    genuine_score,
                )


# ==================================================
# 10. FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    CURRENCYIQ · RESNET18 IMAGE CLASSIFICATION
    <br><br>
    MACHINE LEARNING AND COMPUTER VISION PROJECT
</div>
""", unsafe_allow_html=True)