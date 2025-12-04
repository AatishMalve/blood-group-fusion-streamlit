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
from typing import List

# ==========================
# LOCAL Q&A LOADING (NO INTERNET)
# ==========================

def clean_text(t: str) -> str:
    """Lowercase, remove punctuation, trim spaces."""
    return re.sub(r"[^\w\s]", "", t.lower()).strip()

try:
    with open("qa_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        QA_DATA = {clean_text(k): v for k, v in raw_data.items()}
except (FileNotFoundError, json.JSONDecodeError):
    QA_DATA = {}


def find_local_answer(question: str):
    """Return answer from local JSON if exact or slightly similar; else None."""
    q = clean_text(question)

    if q in QA_DATA:
        return QA_DATA[q]

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

/* Page background */
.stApp {
    background: radial-gradient(circle at top left, #ffe3ec 0, #ffffff 40%, #f3f6ff 100%);
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Hide default Streamlit header/footer */
header[data-testid="stHeader"] {background: rgba(0,0,0,0); }
footer {visibility: hidden;}

/* Section titles */
h2, h3, h4 {
    font-weight: 700;
    color: #1f2933;
}

/* Generic card */
.app-card {
    background: #ffffff;
    padding: 1.2rem 1.5rem;
    border-radius: 18px;
    border: 1px solid #edf0ff;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}

/* Prediction report card */
.report-card {
    background: linear-gradient(135deg, #ffe5ec, #ffffff);
    padding: 1.3rem 1.5rem;
    border-radius: 18px;
    box-shadow: 0 14px 35px rgba(220, 38, 38, 0.20);
    border: 1px solid rgba(248, 113, 113, 0.4);
}

/* Tabs */
[data-baseweb="tab-list"] {
    gap: 0.4rem;
}
button[role="tab"] {
    border-radius: 999px !important;
    padding: 0.45rem 1.4rem !important;
    font-weight: 600 !important;
    border: 1px solid transparent !important;
}
button[role="tab"][aria-selected="true"] {
    background: #ef4444 !important;
    color: #ffffff !important;
}
button[role="tab"][aria-selected="false"] {
    background: #ffffff !important;
    color: #4b5563 !important;
    border-color: #e5e7eb !important;
}

/* Inputs as cards */
.block-container label {
    font-weight: 600;
    color: #374151;
}

/* Primary buttons */
.stButton>button {
    width: 100%;
    border-radius: 999px;
    background: linear-gradient(135deg, #ef4444, #f97316);
    color: white;
    font-weight: 600;
    border: none;
    padding: 0.55rem 0;
    box-shadow: 0 10px 25px rgba(239,68,68,0.45);
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 40px rgba(239,68,68,0.55);
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    border-radius: 16px;
    padding: 0.6rem 0.8rem 1.2rem 0.8rem;
    background: #f9fafb;
    border: 1px dashed #e5e7eb;
}

/* Chat bubbles */
.chat-bubble-user {
    background: #fee2e2;
    padding: 10px 14px;
    border-radius: 14px;
    margin-bottom: 6px;
    max-width: 80%;
    margin-left: auto;
    color: #111827;
}
.chat-bubble-bot {
    background: #ffffff;
    padding: 10px 14px;
    border-radius: 14px;
    margin-bottom: 6px;
    max-width: 80%;
    border: 1px solid #e5e7eb;
}
.chat-box {
    max-height: 350px;
    overflow-y: auto;
    padding-right: 0.5rem;
}

/* Hero subtitle badge */
.hero-subtitle {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(248, 250, 252, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.4);
    font-size: 0.78rem;
    color: #4b5563;
}

/* File details list */
.file-details {
    font-size: 0.9rem;
    color: #4b5563;
}

/* History table */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 10px 26px rgba(15,23,42,0.06);
}

/* Download button */
.stDownloadButton>button {
    border-radius: 999px;
    background: #111827;
    color: #f9fafb;
    font-weight: 600;
}

/* Info box override for Tips */
.app-tip {
    background: #f0f9ff;
    border-radius: 14px;
    padding: 0.9rem 1.0rem;
    border: 1px solid #bae6fd;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# Hero header (uses local emoji icon only)
logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.markdown("<div style='font-size:60px;'>🩸</div>", unsafe_allow_html=True)

with title_col:
    st.markdown(
        """
        <div style="margin-bottom:0.2rem;">
            <span class="hero-subtitle">
                ⚕️ Smart Hematology Assistant
            </span>
        </div>
        <h2 style="margin-bottom:0.2rem;">Blood Group Detection – <span style="color:#ef4444;">Fusion CNN</span></h2>
        <p style="font-size:1.05rem; color:#4b5563; margin-bottom:0.2rem;">
            Right blood. Right time. Saves life.
        </p>
        <p style="font-size:0.9rem; color:#6b7280;">
            ResNet50 + LeNet fusion model with an offline AI assistant for explanations.
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ==========================
# MODEL SETUP (NO DOWNLOAD)
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
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Please place 'model_blood_group_detection_fusion.h5' "
            "in the same folder as app.py."
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
# OFFLINE CHAT CALLBACK
# ==========================
def handle_ask():
    question = st.session_state.get("question_input", "").strip()

    if not question:
        st.warning("Enter a question first.")
        return

    st.session_state["chat"].append({"role": "user", "text": question})
    q_lower = question.lower()

    # show all predictions / history
    if ("show" in q_lower and "prediction" in q_lower and "all" in q_lower) or \
       ("show" in q_lower and "history" in q_lower) or \
       ("list" in q_lower and "prediction" in q_lower):

        try:
            history = load_history()
            if history:
                reply_lines = ["📜 All Previous Predictions:\n"]
                for row in history:
                    reply_lines.append(
                        f"👤 {row['user']} | 🕒 {row['timestamp']} | "
                        f"🩸 {row['prediction']} | 🎯 {row['confidence']}%"
                    )
                reply = "\n".join(reply_lines)
            else:
                reply = "No earlier predictions found, please make a prediction first."
        except Exception:
            reply = "Error retrieving history."

        st.session_state["chat"].append({"role": "bot", "text": reply})
        st.session_state["question_input"] = ""
        return

    # show last prediction
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
            reply = "No earlier predictions found, please make a prediction first."

        st.session_state["chat"].append({"role": "bot", "text": reply})
        st.session_state["question_input"] = ""
        return

    # normal Q&A from local JSON
    local_answer = find_local_answer(question)
    if local_answer:
        reply = local_answer
    else:
        reply = (
            "Offline assistant could not find an exact answer.\n\n"
            "Try asking about:\n"
            "• Blood groups and Rh factor\n"
            "• Meaning of your prediction result\n"
            "• Basic transfusion rules\n\n"
            "You can also add more Q&A to 'qa_data.json' for richer offline explanations."
        )

    st.session_state["chat"].append({"role": "bot", "text": reply})
    st.session_state["question_input"] = ""


# ==========================
# TABS
# ==========================
tab_predict, tab_history, tab_chat = st.tabs(["🔍 Prediction", "📜 History", "💬 AI Chat"])

# ---------- Prediction Tab ----------
with tab_predict:
    st.markdown("### 🔍 Upload image and predict blood group")

    # Input card
    with st.container():
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        col_left, col_right = st.columns([2.2, 1.1])

        with col_left:
            username_input = st.text_input("User Name", "", placeholder="Enter patient / user name")
            uploaded_file = st.file_uploader(
                "Upload blood smear image",
                type=["jpg", "jpeg", "png", "bmp"]
            )

        with col_right:
            st.markdown(
                """
                <div class="app-tip">
                    <b>Tips for best results</b><br/>
                    • Use clear, focused smear images<br/>
                    • Avoid glare and heavy artifacts<br/>
                    • Supported formats: JPG, JPEG, PNG, BMP
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        st.markdown("")
        c1, c2 = st.columns([2.2, 1.0])
        with c1:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            img = Image.open(uploaded_file)
            st.image(img, caption="Preview of uploaded smear image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown("**File details**", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="file-details">
                    • Name: <code>{uploaded_file.name}</code><br/>
                    • Size: {round(len(uploaded_file.getvalue()) / 1024, 2)} KB<br/>
                    • Type: {uploaded_file.type}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔬 Predict blood group", use_container_width=True):
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
    st.markdown("### 📜 Previous predictions")
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    history = load_history()
    if not history:
        st.info("No previous predictions available.")
    else:
        st.dataframe(history, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Chat Tab ----------
with tab_chat:
    st.markdown("### 💬 AI chat (offline)")
    st.caption("Ask about blood groups, Rh factor or your prediction result. Answers use local 'qa_data.json'.")
    st.markdown('<div class="app-card">', unsafe_allow_html=True)

    # Show chat history (bubbles)
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state["chat"]:
        cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{cls}">{msg["text"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Text input
    st.text_input("Ask AI:", key="question_input", placeholder="Type your question here…")

    # Button uses callback
    st.button("Ask", use_container_width=True, on_click=handle_ask)

    st.markdown('</div>', unsafe_allow_html=True)
