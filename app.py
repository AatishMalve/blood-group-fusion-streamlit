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
import json
import re
from difflib import get_close_matches

# ==========================
# Load predefined Q&A (JSON)
# ==========================

def clean_text(t: str) -> str:
    """Lowercase, remove punctuation, trim spaces."""
    return re.sub(r"[^\w\s]", "", t.lower()).strip()

try:
    with open("qa_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        # normalize keys to cleaned form
        QA_DATA = {clean_text(k): v for k, v in raw_data.items()}
except (FileNotFoundError, json.JSONDecodeError):
    QA_DATA = {}


def find_local_answer(question: str):
    """Return answer from local JSON if exact or slightly similar; else None."""
    q = clean_text(question)

    # Exact match
    if q in QA_DATA:
        return QA_DATA[q]

    # Fuzzy close match – similar but not too loose
    keys = list(QA_DATA.keys())
    matches = get_close_matches(q, keys, n=1, cutoff=0.7)  # 0.7 ~ “little bit similar”
    if matches:
        return QA_DATA[matches[0]]

    return None


# ==========================
# GEMINI API CONFIG
# ==========================
# ⚠️ Put your own valid Gemini API key here. Don't commit real keys to public repos.
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash-latest:generateContent"
)


def ask_gemini(question: str) -> str:
    """Fallback: call Gemini API if local JSON has no answer."""
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
                            "fingerprint-based prediction models, and interpreting results. "
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
    except Exception:
        # Graceful fallback if Gemini fails
        return (
            "I couldn’t contact the Gemini service right now. "
            "Please try again later or ask another question from the knowledge base."
        )


# ==========================
# STREAMLIT PAGE STYLE
# ==========================
st.set_page_config(page_title="Blood Group Detection", page_icon="🩸", layout="wide")

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Small logo + title
logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=70)
with title_col:
    st.markdown("### 🩸 Blood Group Detection – Fusion CNN")
    st.caption("ResNet50 + LeNet based model with AI assistant for explanations.")

st.markdown("---")


# ==========================
# MODEL SETUP
# ==========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_blood_group_detection_fusion.h5")
GDRIVE_FILE_ID = "1MUeTJdagltmtkKV6ttdBzOcXsB3RiazU"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

HISTORY_CSV = os.path.join(BASE_DIR, "prediction_history.csv")
CLASS_LABELS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
RESNET_IMG_SIZE = (256, 256)
LENET_IMG_SIZE = (32, 32)


def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(input_tensor)
    se = Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    return Multiply()([input_tensor, se])


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"squeeze_excite_block": squeeze_excite_block}
    )


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
    st.markdown("### Upload image and predict blood group")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        username_input = st.text_input("User Name", "")
        uploaded_file = st.file_uploader(
            "Upload blood smear image", type=["jpg", "jpeg", "png", "bmp"]
        )
    with col_right:
        st.info("Tips:\n- Use clear blood smear images\n- Supported: JPG, JPEG, PNG, BMP")

    if uploaded_file:
        c1, c2 = st.columns([3, 1])
        with c1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=350)
        with c2:
            st.write("**File details**")
            st.write(f"Name: `{uploaded_file.name}`")
            st.write(f"Size: {round(len(uploaded_file.getvalue()) / 1024, 2)} KB")
            st.write(f"Type: {uploaded_file.type}")

        if st.button("Predict", use_container_width=True):
            with st.spinner("Detecting blood group..."):
                res_in = preprocess_resnet(img)
                len_in = preprocess_lenet(img)
                preds = cnn_model.predict([res_in, len_in])
                idx = int(np.argmax(preds))
                conf = round(float(np.max(preds)) * 100, 2)

            label = CLASS_LABELS[idx]
            ist = pytz.timezone("Asia/Kolkata")
            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            display_name = username_input.strip() or "Anonymous"

            st.session_state["last_report"] = {
                "user": display_name,
                "timestamp": timestamp,
                "label": label,
                "confidence": conf,
            }

            log_history(display_name, timestamp, label, conf)

            st.markdown(
                f"""
                <div class="report-card">
                    <h4>🩸 Prediction Report</h4>
                    <b>User:</b> {display_name}<br>
                    <b>Date/Time:</b> {timestamp}<br>
                    <b>Prediction:</b> {label}<br>
                    <b>Confidence:</b> {conf}%<br>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Simple text report download
    if st.session_state["last_report"]:
        rep = st.session_state["last_report"]
        report_text = (
            "BloodPrint Prediction Report\n"
            "----------------------------\n"
            f"User: {rep['user']}\n"
            f"Date/Time: {rep['timestamp']}\n"
            f"Prediction: {rep['label']}\n"
            f"Confidence: {rep['confidence']}%\n"
        )
        st.download_button(
            "📄 Download Report (TXT)",
            report_text.encode("utf-8"),
            "bloodprint_report.txt",
            "text/plain",
            use_container_width=True,
        )


# ---------- History Tab ----------
with tab_history:
    st.markdown("### Previous Predictions")
    history = load_history()
    if not history:
        st.info("No previous predictions available.")
    else:
        st.dataframe(history, use_container_width=True)


# ---------- Chat Tab ----------
with tab_chat:
    st.markdown("### Chat with AI about blood groups or your results")
    st.caption("Example: *What is the difference between A+ and A- blood?*")

    # Show chat history (bubbles)
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state["chat"]:
        cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{cls}">{msg["text"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    question = st.text_input("Ask AI:")

    if st.button("Ask", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            # add user message to history
            st.session_state["chat"].append({"role": "user", "text": question})

            # think...
            with st.spinner("Thinking..."):
                local_answer = find_local_answer(question)

                if local_answer:
                    reply = local_answer + "\n\n(Answer from built-in knowledge base)"
                else:
                    reply = ask_gemini(question)

            # add bot reply to history
            st.session_state["chat"].append({"role": "bot", "text": reply})
            st.rerun()
