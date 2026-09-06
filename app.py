import numpy as np
import streamlit as st
from PIL import Image

from model_loader import load_bird_model

st.set_page_config(page_title="Bird Species Recognition", page_icon="🦜")
st.title("🦜 Bird Species Recognition App")


@st.cache_resource
def get_model():
    return load_bird_model()


session, id2label = get_model()

uploaded_file = st.file_uploader(
    "Upload a bird image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Bird Image", use_container_width=True)
    st.write("Analyzing... 🔍")

    # The exported ONNX model expects NCHW input with 224x224 pixels.
    image_array = np.asarray(image.resize((224, 224))).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.47853944, 0.4732864, 0.47434163], dtype=np.float32)

    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})
    logits = outputs[0][0]

    probabilities = np.exp(logits - np.max(logits))
    probabilities = probabilities / probabilities.sum()

    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    label = id2label[str(predicted_idx)]

    st.success(f"✅ Predicted species: **{label}**")
    st.caption(f"Confidence: {confidence * 100:.2f}%")
else:
    st.info("📸 Upload an image to start recognition.")
