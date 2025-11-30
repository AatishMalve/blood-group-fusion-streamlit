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
from fpdf import FPDF

# ==========================
# GEMINI API CONFIG
# ==========================
GOOGLE_API_KEY = "AIzaSyAkcqpRvFiT46L4BG7WGqTDWsv1CdUuVOc"   # Replace with your new valid key

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
                    {"text": "Explain in simple words:\n" + question}
                ]
            }
        ]
    }

    try:
        resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error talking to Gemini API: {e}"


# ==========================
# STREAMLIT PAGE CONFIG
# ==========================
st.set_page_config(page_title="Blood Group Detection", page_icon="🩸", layout="wide")

st.markdown("""
<style>
.report-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.07);
    border: 1px solid #ececec;
}
.chat-bubble-user {
    background-color: #d8ebff;
    padding: 8px 12px;
    border-radius: 14px;
    margin-bottom: 6px;
    max-width: 80%;
    margin-left: auto;
}
.chat-bubble-bot {
    background-color: #ffffff;
    padding: 8px 12px;
    border-radius: 14px;
    margin-bottom: 6px;
    max-width: 80%;
    border: 1px solid #e4e4ef;
}
.chat-box {
    max-height: 350px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ==========================
# MODEL SETUP
# ==========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_blood_group_detection_fusion.h5")
GDRIVE_FILE_ID = "1MUeTJdagltmtkKV6ttdBzOcXsB3RiazU"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

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
        with st.spinner("Downloading model (first time only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"squeeze_excite_block": squeeze_excite_block})

cnn_model = load_model()

def preprocess_resnet(img):
    arr = np.array(img.convert("RGB").resize(RESNET_IMG_SIZE)).astype("float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def preprocess_lenet(img):
    arr = np.array(img.convert("RGB").resize(LENET_IMG_SIZE)).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def log_history(user, timestamp, label, confidence):
    file_exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user", "timestamp", "prediction", "confidence"])
        writer.writerow([user, timestamp, label, confidence])

def load_history():
    if not os.path.exists(HISTORY_CSV):
        return []
    with open(HISTORY_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ==========================
# SESSION STATE
# ==========================
st.session_state.setdefault("chat", [])
st.session_state.setdefault("last_report", None)


# ==========================
# TABS
# ==========================
tab_predict, tab_history, tab_chat = st.tabs(["🔍 Prediction", "📜 History", "💬 AI Chat"])

# ---------- Prediction Tab ----------
with tab_predict:

    user = st.text_input("User Name", "")
    uploaded = st.file_uploader("Upload blood smear image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        col1, col2 = st.columns([3, 1])
        with col1:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", width=350)

        if st.button("Predict"):
            with st.spinner("Detecting blood group..."):
                pr1 = preprocess_resnet(img)
                pr2 = preprocess_lenet(img)
                preds = cnn_model.predict([pr1, pr2])
                idx = int(np.argmax(preds))
                conf = round(float(np.max(preds)) * 100, 2)

            label = CLASS_LABELS[idx]
            timeIST = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
            username = user.strip() or "Anonymous"

            st.session_state["last_report"] = {
                "user": username,
                "timestamp": timeIST,
                "label": label,
                "confidence": conf,
            }

            log_history(username, timeIST, label, conf)

            st.markdown(
                f"""
                <div class="report-card">
                <h4>🩸 Prediction Report</h4>
                <b>User:</b> {username}<br>
                <b>Time:</b> {timeIST}<br>
                <b>Prediction:</b> {label}<br>
                <b>Confidence:</b> {conf}%</div>
                """,
                unsafe_allow_html=True
            )

        if st.session_state["last_report"]:
        rep = st.session_state["last_report"]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "BloodPrint Prediction Report", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.ln(4)
        pdf.cell(0, 8, f"User: {rep['user']}", ln=True)
        pdf.cell(0, 8, f"Date/Time: {rep['timestamp']}", ln=True)
        pdf.cell(0, 8, f"Prediction: {rep['label']}", ln=True)
        pdf.cell(0, 8, f"Confidence: {rep['confidence']}%", ln=True)

        pdf_out = pdf.output(dest="S")
        if isinstance(pdf_out, str):
            pdf_bytes = pdf_out.encode("latin-1")
        else:
            pdf_bytes = bytes(pdf_out)

        st.download_button(
            "📄 Download PDF Report",
            pdf_bytes,
            "report.pdf",
            "application/pdf"
        )



# ---------- History Tab ----------
with tab_history:
    hist = load_history()
    if not hist:
        st.info("No previous predictions available.")
    else:
        st.dataframe(hist, use_container_width=True)


# ---------- Chat Tab ----------
with tab_chat:

    st.caption("Example: What is the difference between A+ and A- blood?")

    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for m in st.session_state["chat"]:
        bubble = "chat-bubble-user" if m["role"] == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{bubble}">{m["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    question = st.text_input("Ask AI:")

    if st.button("Ask", use_container_width=True):
        if question.strip():
            st.session_state["chat"].append({"role": "user", "text": question})
            with st.spinner("Thinking..."):
                reply = ask_gemini(question)
            st.session_state["chat"].append({"role": "bot", "text": reply})
            st.rerun()
        else:
            st.warning("Enter a question first!")


