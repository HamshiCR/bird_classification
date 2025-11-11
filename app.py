import streamlit as st
from PIL import Image
import torch
from model_loader import load_bird_model

# Load model & processor
processor, model = load_bird_model()

st.set_page_config(page_title="Bird Species Recognition", page_icon="🦜")
st.title("🦜 Bird Species Recognition App")

uploaded_file = st.file_uploader("Upload a bird image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Bird Image", use_column_width=True)
    st.write("Analyzing... 🔍")

    # Preprocess using processor instead of extractor
    inputs = processor(images=image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_idx].item()
        label = model.config.id2label[predicted_idx]

    st.success(f"✅ Predicted species: **{label}**")
    st.caption(f"Confidence: {confidence*100:.2f}%")
else:
    st.info("📸 Upload an image to start recognition.")
