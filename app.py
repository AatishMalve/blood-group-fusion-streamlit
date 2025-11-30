import os
import csv
from datetime import datetime

import numpy as np
import pytz
import streamlit as st
import requests  # <-- use REST API instead of google-generativeai
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, Multiply
import gdown

# -----------------------
# Gemini / Google API Key
# -----------------------
# ⛔ IMPORTANT: regenerate a new key in Google AI Studio
# and DO NOT commit it to GitHub or share it.
GOOGLE_API_KEY = "AIzaSyAkcqpRvFiT46L4BG7WGqTDWsv1CdUuVOc"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


def ask_gemini(question: str) -> str:
    """Call Gemini API using HTTP POST and return the text reply."""
    if not GOOGLE_API_KEY:
        return "Gemini API key is not configured."

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GOOGLE_API_KEY,
    }

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
# Paths and Google Drive model
# -----------------------
BASE_DIR = os.path.dirname(__file__)

# your Google Drive file ID
GDRIVE_FILE_ID = "1MUeTJdagltmtkKV6ttdBzOcXsB3RiazU"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
MODEL_PATH = os.path.join(BASE_DIR, "model_blood_group_detection_fusion.h5")

# history file
HISTORY_CSV = os.path.join(BASE_DIR, "prediction_history.csv")

# class labels
CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
RESNET_IMG_SIZE = (256, 256)
LENET_IMG_SIZE = (32, 32)

# -----------------------
# custom block used in model
# -----------------------
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = Dense(filters // ratio, activation="relu",
               kernel_initializer="he_normal", use_bias=False)(input_tensor)
    se = Dense(filters, activation="sigmoid",
               kernel_initializer="he_normal", use_bias=False)(se)
    x = Multiply()([input_tensor, se])
    return x

@st.cache_resource
def load_model():
    # download once if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only, please wait)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"squeeze_excite_block": squeeze_excite_block},
    )
    return model

cnn_model = load_model()

# -----------------------
# Preprocessing helpers
# -----------------------
def preprocess_resnet(pil_img):
    img = pil_img.convert("RGB").resize(RESNET_IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def preprocess_lenet(pil_img):
    img = pil_img.convert("RGB").resize(LENET_IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------
# History helpers
# -----------------------
def log_prediction(user, timestamp, label, confidence_pct):
    """Append one prediction entry to a CSV file."""
    file_exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user", "timestamp", "prediction", "confidence_percent"])
        writer.writerow([user, timestamp, label, confidence_pct])

def load_history():
    """Read prediction history from CSV as list of dicts."""
    if not os.path.exists(HISTORY_CSV):
        return []
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Blood Group Detection", page_icon="🩸")
st.title("🩸 Blood Group Detection – ResNet50 + LeNet Fusion")

username = st.text_input("User name", value="")

st.write("Upload a blood smear image to predict the **blood group** using the fusion CNN model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict blood group"):
        with st.spinner("Running prediction..."):
            resnet_input = preprocess_resnet(image)
            lenet_input = preprocess_lenet(image)

            # model expects [resnet_batch, lenet_batch]
            preds = cnn_model.predict([resnet_input, lenet_input])
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))

        predicted_label = CLASS_LABELS[idx]
        ist = pytz.timezone("Asia/Kolkata")
timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
        display_name = username.strip() or "Anonymous"
        confidence_pct = round(confidence * 100, 2)

        # show report
        st.subheader("BloodPrint Prediction Report")
        st.write(f"**User:** {display_name}")
        st.write(f"**Date/Time:** {timestamp}")
        st.write(f"**Prediction:** {predicted_label}")
        st.write(f"**Confidence:** {confidence_pct:.2f}%")

        # save to history
        log_prediction(display_name, timestamp, predicted_label, confidence_pct)
        st.info("✅ Prediction saved to history.")

# -----------------------
# History button
# -----------------------
if st.button("Show prediction history"):
    history = load_history()
    if not history:
        st.warning("No predictions have been made yet.")
    else:
        st.subheader("Prediction History")
        st.write(f"Total predictions: **{len(history)}**")
        st.table(history)

# -----------------------
# Gemini Chatbot Section
# -----------------------
st.markdown("---")
st.subheader("💬 Ask AI about blood groups or results")

user_question = st.text_input("Type your question here:")

if st.button("Get AI Response"):
    if not user_question.strip():
        st.warning("Please enter a question before asking.")
    else:
        with st.spinner("Contacting Gemini..."):
            answer = ask_gemini(user_question)
        st.write(answer)



