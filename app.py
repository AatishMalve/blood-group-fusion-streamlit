import os
import csv
from datetime import datetime

import numpy as np
import pytz
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, Multiply
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
    matches = get_close_matches(q, keys, n=1, cutoff=0.7)
    if matches:
        return QA_DATA[matches[0]]

    return None


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
    # Local logo: put heart.png in same folder as app.py
    local_logo = os.path.join(os.path.dirname(__file__), "heart.png")
    if os.path.exists(local_logo):
        st.image(local_logo, width=70)

with title_col:
    st.markdown("### 🩸 Blood Group Detection – Fusion CNN")
    st.markdown(
        "<p style='font-size:20px; font-weight:600;'>Right blood. Right time. Saves life.</p>",
        unsafe_allow_html=True,
    )
    st.caption("ResNet50 + LeNet based model with AI assistant for explanations.")

st.markdown("---")


# ==========================
# MODEL SETUP (OFFLINE)
# ==========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_blood_group_detection_fusion.h5")

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
    # Offline: model must already exist locally
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found.\n\n"
            "Please place 'model_blood_group_detection_fusion.h5' in the "
            "same folder as app.py, then restart the app."
        )
        st.stop()
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
# ASK BUTTON CALLBACK (OFFLINE)
# ==========================
def handle_ask():
    question = st.session_state.get("question_input", "").strip()

    if not question:
        st.warning("Enter a question first.")
        return

    # add user message to history
    st.session_state["chat"].append({"role": "user", "text": question})

    q_lower = question.lower()

    # 1️⃣ SHOW ALL PREDICTIONS / HISTORY
    if ("show" in q_lower and "prediction" in q_lower and "all" in q_lower) or \
       ("show" in q_lower and "history" in q_lower) or \
       ("list" in q_lower and "prediction" in q_lower):

        try:
            history = load_history()
            if history:
                reply = "📜 **All Previous Predictions:**\n\n"
                for row in history:
                    reply += (
                        f"👤 {row['user']} | 🕒 {row['timestamp']} | "
                        f"🩸 {row['prediction']} | 🎯 {row['confidence']}%\n"
                    )
            else:
                reply = "Sorry, no earlier predictions, please predict first."
        except Exception:
            reply = "Error retrieving history."

        st.session_state["chat"].append({"role": "bot", "text": reply})
        st.session_state["question_input"] = ""
        return

    # 2️⃣ SHOW ONLY LAST / EARLIER PREDICTION
    if ("show" in q_lower and "prediction" in q_lower and
        ("earlier" in q_lower or "last" in q_lower or "recent" in q_lower)):

        last = st.session_state.get("last_report")
        if last:
            reply = (
                "Here is your last prediction:\n\n"
                f"User: {last['user']}\n"
                f"Date/Time: {last['timestamp']}\n"
                f"Prediction: {last['label']}\n"
                f"Confidence: {last['confidence']}%\n"
            )
        else:
            reply = "Sorry, no earlier predictions, please predict first."

        st.session_state["chat"].append({"role": "bot", "text": reply})
        st.session_state["question_input"] = ""
        return

    # 3️⃣ OFFLINE Q&A ONLY
    local_answer = find_local_answer(question)
    if local_answer:
        reply = local_answer
    else:
        reply = (
            "I am running in offline mode and couldn't find this question in the "
            "saved Q&A.\n\n"
            "Please try rephrasing or ask something about blood groups, Rh factor, "
            "fingerprint-based prediction, or this application."
        )

    st.session_state["chat"].append({"role": "bot", "text": reply})
    st.session_state["question_input"] = ""


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
    st.markdown("### Chat with AI (offline Q&A)")
    st.caption("Uses saved Q&A from qa_data.json (no internet required).")

    # Show chat history (bubbles)
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state["chat"]:
        cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{cls}">{msg["text"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.text_input("Ask AI:", key="question_input")
    st.button("Ask", use_container_width=True, on_click=handle_ask)
