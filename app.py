import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ====== PAGE CONFIG ======
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")

st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image and let the model predict its species!")

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/final_best_model_finetuned.h5")  # or .keras
    return model

model = load_model()

# ====== CLASS LABELS ======
labels = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# ====== UPLOAD IMAGE ======
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")
    st.success(f"**Predicted Species:** {pred_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Show all class probabilities
    with st.expander("See class probabilities"):
        for lbl, prob in zip(labels, prediction[0]):
            st.write(f"{lbl:35s} → {prob*100:.2f}%")

st.markdown("---")
st.caption("Model: MobileNetV2 fine-tuned")
