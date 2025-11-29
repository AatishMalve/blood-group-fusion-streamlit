import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, Multiply

# -----------------------
# Paths and constants
# -----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_blood_group_detection_fusion.h5")

# If you prefer to read from classes_fusion.txt, you can,
# but hard-coding is simple and safe:
CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

RESNET_IMG_SIZE = (256, 256)
LENET_IMG_SIZE = (32, 32)

# -----------------------
# Custom block for loading
# -----------------------
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(input_tensor)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([input_tensor, se])
    return x

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"squeeze_excite_block": squeeze_excite_block}
    )
    return model

model = load_model()

# -----------------------
# Preprocessing helpers
# -----------------------
def preprocess_resnet(pil_img):
    img = pil_img.convert("RGB").resize(RESNET_IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)          # ResNet preprocessing
    arr = np.expand_dims(arr, axis=0)    # shape (1, 256, 256, 3)
    return arr

def preprocess_lenet(pil_img):
    img = pil_img.convert("RGB").resize(LENET_IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)    # shape (1, 32, 32, 3)
    return arr

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Blood Group Detection", page_icon="🩸")
st.title("🩸 Blood Group Detection – ResNet50 + LeNet Fusion")

st.write("Upload a blood smear image to predict the **blood group** using your fusion model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict blood group"):
        with st.spinner("Running prediction..."):
            resnet_input = preprocess_resnet(image)
            lenet_input = preprocess_lenet(image)

            # model expects [resnet_batch, lenet_batch]
            preds = model.predict([resnet_input, lenet_input])
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))

        predicted_label = CLASS_LABELS[idx]
        st.success(f"Predicted blood group: **{predicted_label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
