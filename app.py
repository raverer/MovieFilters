import streamlit as st
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans

st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")

# -----------------------
# LUT loading utilities
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
def load_all_luts(lut_folder="luts"):
    luts = {}
    for file in os.listdir(lut_folder):
        if file.endswith(".cube"):
            name = file.replace(".cube", "")
            luts[name] = load_cube_file(os.path.join(lut_folder, file))
    return luts

def apply_lut(image, lut):
    img = np.array(image).astype(np.float32) / 255.0
    img = np.clip(img, 0, 1)
    idx = np.clip((img * (lut.shape[0]-1)).astype(int), 0, lut.shape[0]-1)
    result = lut[idx[...,0], idx[...,1], idx[...,2]]
    return (result * 255).astype(np.uint8)

# -----------------------
# Generate filter previews
# -----------------------
@st.cache_resource
def generate_previews(luts, base_image_path="preview/base.jpg"):
    import requests
    from io import BytesIO

    # If no local file exists, use a fallback placeholder image
    if not os.path.exists(base_image_path):
        st.warning("No preview/base.jpg found — using placeholder image.")
        url = "https://placehold.co/800x600/888/FFF?text=Movie+Preview"
        base = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    else:
        base = Image.open(base_image_path).convert("RGB")

    base = base.resize((350, 250))
    previews = {}
    for name, lut in luts.items():
        previews[name] = apply_lut(base, lut)
    return previews

# -----------------------
# AI mood analyzer
# -----------------------
def analyze_image_mood(image):
    img = np.array(image.resize((100, 100))) / 255.0
    flat = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=5).fit(flat)
    dominant = kmeans.cluster_centers_.mean(axis=0)
    r, g, b = dominant

    # heuristic for mood classification
    if r > 0.6 and g < 0.4:
        return "warm & dramatic", "oldboy"
    elif b > 0.55:
        return "cool & sci-fi", "dune"
    elif r > 0.5 and g > 0.5:
        return "vibrant & playful", "wes_anderson"
    else:
        return "moody & nostalgic", "wongkarwai"

# -----------------------
# Streamlit UI
# -----------------------
st.title("🎥 Movie Filter Studio")
st.caption("Apply cinematic looks inspired by legendary films.")

luts = load_all_luts()
previews = generate_previews(luts)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

# Big preview gallery
st.subheader("🎞 Choose Your Cinematic Look")
cols = st.columns(3)
for i, (name, img) in enumerate(previews.items()):
    with cols[i % 3]:
        st.image(img, caption=name, use_container_width=True)
        if st.button(f"Apply {name}"):
            st.session_state["selected_filter"] = name

# AI suggestion
if uploaded_file:
    st.markdown("### 🤖 AI Filter Suggestion")
    image = Image.open(uploaded_file).convert("RGB")
    mood, suggestion = analyze_image_mood(image)
    st.info(f"AI detects this photo is **{mood}**, suggested filter: **{suggestion}** 🎬")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        if "selected_filter" in st.session_state:
            selected_filter = st.session_state["selected_filter"]
        else:
            selected_filter = suggestion
        result = apply_lut(image, luts[selected_filter])
        st.image(result, caption=f"Applied: {selected_filter}", use_container_width=True)

        st.download_button(
            "📥 Download Image",
            data=Image.fromarray(result).tobytes(),
            file_name=f"{selected_filter}_filtered.png",
            mime="image/png"
        )
else:
    st.info("Upload an image to get AI-suggested cinematic filters.")
