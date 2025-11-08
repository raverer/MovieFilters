# ==========================================
# 🎥 MOVIE FILTER STUDIO v1.0_final_base + Oppenheimer
# ==========================================
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import os
import requests
from io import BytesIO

st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")

# -----------------------
# LUT Loader
# -----------------------
@st.cache_resource
def load_cube_file(path):
    lut = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or "LUT_3D_SIZE" in line:
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                lut.append([float(x) for x in parts])
    size = int(round(len(lut) ** (1/3)))
    return np.array(lut).reshape((size, size, size, 3))

@st.cache_resource
def load_all_luts(folder="luts"):
    luts = {}
    for file in os.listdir(folder):
        if file.endswith(".cube"):
            name = file.replace(".cube", "")
            luts[name] = load_cube_file(os.path.join(folder, file))
    return luts

def apply_lut(image, lut):
    img = np.array(image).astype(np.float32) / 255.0
    idx = np.clip((img * (lut.shape[0] - 1)).astype(int), 0, lut.shape[0] - 1)
    result = lut[idx[..., 0], idx[..., 1], idx[..., 2]]
    return (result * 255).astype(np.uint8)

# -----------------------
# Load and generate previews
# -----------------------
@st.cache_resource
def load_base_preview(path="preview/base.jpg", resize_to=(500, 350)):
    if os.path.exists(path):
        base = Image.open(path).convert("RGB")
    else:
        url = "https://images.unsplash.com/photo-1612690119274-8819a81c13a2?auto=format&fit=crop&w=1200&q=80"
        resp = requests.get(url, timeout=10)
        base = Image.open(BytesIO(resp.content)).convert("RGB")
    return base.resize(resize_to, Image.LANCZOS)

@st.cache_resource
def generate_previews(base_image, luts):
    previews = {}
    for name, lut in luts.items():
        previews[name] = apply_lut(base_image, lut)
    return previews

# -----------------------
# Streamlit UI
# -----------------------
st.title("🎞️ Movie Filter Studio")
st.caption("Now featuring cinematic tones from **Oldboy**, **Dune**, **Grand Budapest Hotel**, and **Oppenheimer**.")

luts = load_all_luts()
base = load_base_preview()
previews = generate_previews(base, luts)

# Preview gallery
st.markdown("### 🎞 Filter Previews")
cols = st.columns(min(4, len(previews)))
for i, (name, img) in enumerate(previews.items()):
    with cols[i % len(cols)]:
        st.image(img, caption=name, use_column_width=True)

# Filter selection
filter_choice = st.selectbox("🎬 Choose your cinematic look:", list(luts.keys()))
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner(f"Applying {filter_choice}..."):
        result = apply_lut(image, luts[filter_choice])
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(result, caption=f"{filter_choice} Look", use_column_width=True)
        buf = io.BytesIO()
        Image.fromarray(result).save(buf, format="JPEG", quality=98)
        st.download_button("📥 Download", buf.getvalue(), f"{filter_choice}.jpg", "image/jpeg")
else:
    st.info("Upload an image to apply your cinematic filter.")
