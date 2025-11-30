import os
import csv
from datetime import datetime

import numpy as np
import pytz
import streamlit as st
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, Multiply
import gdown
from fpdf import FPDF   # for PDF export

# -----------------------
# Gemini API Key
# -----------------------
# ⚠️ Replace with a NEW key from Google AI Studio. Don't push real key to GitHub.
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

def ask_gemini(question: str) -> str:
    if not GOOGLE_API_KEY:
        return "Gemini API key is not configured."

    headers = {"Content-Type": "application/json", "x-goog-api-key": GOOGLE_API_KEY}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You are an AI assistant helping with questions about blood groups, "
                            "blood smear analysis, and interpreting blood group prediction results. "
                            "Explain things in simple, clear language.\n\n"
                            f"User question: {question}"
                        )
                    }
                ]
            }
        ]
    }

    try:
        resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "No response from Gemini (no candidates)."
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return "No response from Gemini (no content parts)."
        return parts[0].get("text", "Gemini returned no text.")
    except Exception as e:
        return f"Error talking to Gemini API: {e}"


# -----------------------
# Streamlit Page Setup & Global Style
# -----------------------
st.set_page_config(page_title="Blood Group Detection", page_icon="🩸", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7fb;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.5rem;
    }
    .app-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #555;
        margin: 0;
    }
    .report-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        border: 1px solid #e6e6ef;
        margin-top: 1rem;
    }
    .chat-bubble-user {
        background-color: #e0f0ff;
        padding: 0.6rem 0.8rem;
        border-radius: 16px;
        margin-bottom: 0.4rem;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bubble-bot {
        background-color: #ffffff;
        padding: 0.6rem 0.8rem;
        border-radius: 16px;
        margin-bottom: 0.4rem;
        max-width: 80%;
        border: 1px solid #e4e4f0;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 4px;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Small logo + title (compact)
logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3004/3004458.png",
        width=70,
    )
with title_col:
    st.markdown('<div class="app-header"><div>', unsafe_allow_html=True)
    st.markdown('<p class="app-title">🩸 Blood Group Detection – Fusion CNN</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">ResNet50 + LeNet based model with AI assistant for explanations.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Model + Paths
# -----------------------
BASE_DIR = os.path.dirname(__file__)
GDRIVE_FILE_ID = "1MUeTJdagltmtkKV6ttdBzOcXsB3RiazU"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
MODEL_PATH = os.path.join(BASE_DIR, "model_blood_group_detection_fusion.h5")
HISTORY_CSV = os.path.join(BASE_DIR, "prediction_history.csv")

CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
RESNET_IMG_SIZE = (256, 256)
LENET_IMG_SIZE = (32, 32)

def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(input_tensor)
    se = Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)
    return Multiply()([input_tensor, se])

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model (first time only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"squeeze_excite_block": squeeze_excite_block},
    )

cnn_model = load_model()

def preprocess_resnet(pil_img):
    img = pil_img.convert("RGB").resize(RESNET_IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def preprocess_lenet(pil_img):
    img = pil_img.convert("RGB").resize(LENET_IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def log_prediction(user, timestamp, label, confidence_pct):
    file_exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user", "timestamp", "prediction", "confidence_percent"])
        writer.writerow([user, timestamp, label, confidence_pct])

def load_history():
    if not os.path.exists(HISTORY_CSV):
        return []
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# -----------------------
# Session state for chat & last prediction
# -----------------------
if "chat" not in st.session_state:
    st.session_state.chat = []   # [{"role": "user"/"bot", "text": "..."}]

if "last_report" not in st.session_state:
    st.session_state.last_report = None  # for PDF download

# -----------------------
# Tabs Layout
# -----------------------
tab_pred, tab_hist, tab_chat = st.tabs(["🩸 Prediction", "📜 History", "💬 AI Chat"])

# ---------- TAB 1: Prediction ----------
with tab_pred:
    st.markdown("### Upload image and predict blood group")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        username = st.text_input("User name", value="")
        uploaded_file = st.file_uploader("📤 Upload blood smear image", type=["jpg", "jpeg", "png", "bmp"])
    with col_right:
        st.info("Tips:\n- Use clear blood smear images\n- Supported: JPG, PNG, BMP")

    if uploaded_file:
        img_col, info_col = st.columns([3, 1])
        with img_col:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
        with info_col:
            st.markdown("**File details**")
            st.write(f"Name: `{uploaded_file.name}`")
            st.write(f"Size: {round(len(uploaded_file.getvalue())/1024, 2)} KB")
            st.write(f"Type: {uploaded_file.type}")

        if st.button("🔍 Predict Blood Group", use_container_width=True):
            with st.spinner("Running prediction..."):
                resnet_input = preprocess_resnet(img)
                lenet_input = preprocess_lenet(img)
                preds = cnn_model.predict([resnet_input, lenet_input])
                idx = int(np.argmax(preds))
                confidence = float(np.max(preds))

            predicted_label = CLASS_LABELS[idx]
            ist = pytz.timezone("Asia/Kolkata")
            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            display_name = username.strip() or "Anonymous"
            confidence_pct = round(confidence * 100, 2)

            # Save last report in session for PDF download
            st.session_state.last_report = {
                "user": display_name,
                "timestamp": timestamp,
                "prediction": predicted_label,
                "confidence": confidence_pct,
            }

            # Nice report card
            st.markdown(
                f"""
                <div class="report-card">
                    <h4>🩸 BloodPrint Prediction Report</h4>
                    <p><b>User:</b> {display_name}</p>
                    <p><b>Date/Time:</b> {timestamp}</p>
                    <p><b>Predicted Blood Group:</b> {predicted_label}</p>
                    <p><b>Model Confidence:</b> {confidence_pct:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            log_prediction(display_name, timestamp, predicted_label, confidence_pct)
            st.success("Prediction saved to history.")

    # PDF download button (only if we have a last_report)
    if st.session_state.last_report:
        rep = st.session_state.last_report

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "BloodPrint Prediction Report", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.ln(4)
        pdf.cell(0, 8, f"User: {rep['user']}", ln=True)
        pdf.cell(0, 8, f"Date/Time: {rep['timestamp']}", ln=True)
        pdf.cell(0, 8, f"Predicted Blood Group: {rep['prediction']}", ln=True)
        pdf.cell(0, 8, f"Model Confidence: {rep['confidence']:.2f}%", ln=True)

        pdf_out = pdf.output(dest="S")  # may be str or bytes
        if isinstance(pdf_out, str):
            pdf_bytes = pdf_out.encode("latin-1")
        else:
            pdf_bytes = pdf_out

        st.download_button(
            label="📄 Download Prediction as PDF",
            data=pdf_bytes,
            file_name="bloodprint_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ---------- TAB 2: History ----------
with tab_hist:
    st.markdown("### Previous Predictions")
    history = load_history()
    if not history:
        st.info("No predictions have been made yet.")
    else:
        st.write(f"Total predictions: **{len(history)}**")
        st.dataframe(history, use_container_width=True)

# ---------- TAB 3: Chat ----------
with tab_chat:
    st.markdown("### Chat with AI about blood groups or your results")
    st.caption("Examples: *What is the difference between A+ and A- blood?*")

    # Show chat history with bubbles
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot">{msg["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    user_question = st.text_area("Type your question here:", height=80)

    if st.button("Ask AI", use_container_width=True):
        if not user_question.strip():
            st.warning("Please enter a question before asking.")
        else:
            st.session_state.chat.append({"role": "user", "text": user_question})
            with st.spinner("Thinking..."):
                answer = ask_gemini(user_question)
            st.session_state.chat.append({"role": "bot", "text": answer})
            st.experimental_rerun()
