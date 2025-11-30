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

# -----------------------
# Gemini API Key
# -----------------------
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # <-- Put your key here safely

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
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error talking to Gemini API: {e}"


# -----------------------
# Streamlit Page Setup & UI Style
# -----------------------
st.set_page_config(page_title="Blood Group Detection", page_icon="🩸", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f7f7fb; }
    .report-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #eee;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header image
st.image("https://images.pexels.com/photos/5207097/pexels-photo-5207097.jpeg", use_column_width=True)
st.markdown("## 🩸 Blood Group Detection – ResNet50 + LeNet Fusion")
st.write("Upload a blood smear image below to predict the **blood group** using the AI model.")

# -----------------------
# Model Load + Paths
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

    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"squeeze_excite_block": squeeze_excite_block})

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
# UI - Upload + Prediction
# -----------------------
username = st.text_input("User name", value="")
uploaded_file = st.file_uploader("📤 Upload blood smear image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    col1, col2 = st.columns([3, 1])
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.write("### File Info")
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {round(len(uploaded_file.getvalue())/1024, 2)} KB")

    if st.button("🔍 Predict Blood Group"):
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

        st.markdown(
            f"""
            <div class="report-card">
                <h3>🩸 BloodPrint Prediction Report</h3>
                <p><b>User:</b> {display_name}</p>
                <p><b>Date/Time:</b> {timestamp}</p>
                <p><b>Predicted Blood Group:</b> {predicted_label}</p>
                <p><b>Model Confidence:</b> {confidence_pct:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        log_prediction(display_name, timestamp, predicted_label, confidence_pct)
        st.success("Prediction saved to history!")

# -----------------------
# History Section
# -----------------------
if st.button("📜 Show Prediction History"):
    history = load_history()
    if not history:
        st.info("No past predictions yet.")
    else:
        st.table(history)

# -----------------------
# Chatbot
# -----------------------
st.markdown("---")
st.subheader("💬 Ask AI about blood groups or results")
st.caption("Example: *What is the difference between A+ and A- blood?*")

user_question = st.text_area("Type your question here:", height=70)

if st.button("Ask AI"):
    with st.spinner("Thinking..."):
        response = ask_gemini(user_question)
    st.write(response)
