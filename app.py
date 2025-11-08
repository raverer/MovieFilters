import streamlit as st
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
import requests
from io import BytesIO

# -----------------------
# Streamlit page setup
# -----------------------
st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")
st.markdown(
    """
    <style>
    /* App-wide aesthetic styles */
    body {background-color: #0e1117;}
    .stApp {background-color: #0e1117;}
    h1, h2, h3, h4, h5, h6, p, span {color: #404040 !important;}
    .thumbnail {
        border-radius: 18px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 0px 15px rgba(255,255,255,0.08);
    }
    .thumbnail:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 25px rgba(255,255,255,0.2);
        border: 2px solid #ff5c5c;
    }
    .filter-name {
        text-align: center;
        font-weight: 600;
        margin-top: 0.5rem;
        color: #a3a3a3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
# Generate filter previews (HD-safe)
# -----------------------
@st.cache_resource
def generate_previews(luts, base_image_path="preview/base.jpg"):
    if os.path.exists(base_image_path):
        base = Image.open(base_image_path).convert("RGB")
    else:
        url = "https://images.unsplash.com/photo-1612690119274-8819a81c13a2?auto=format&fit=crop&w=1200&q=80"
        st.warning("⚠️ No preview/base.jpg found — using fallback neutral image.")
        response = requests.get(url)
        base = Image.open(BytesIO(response.content)).convert("RGB")

    base = base.resize((500, 350))
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
st.caption("Transform your photos with cinematic filters inspired by iconic films.")

luts = load_all_luts()
previews = generate_previews(luts)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------
# Filter Gallery (Grid)
# -----------------------
st.markdown("### 🎞️ Choose Your Cinematic Look")
cols = st.columns(4)

for i, (name, img) in enumerate(previews.items()):
    col = cols[i % 4]
    with col:
        st.image(img, use_container_width=True, caption=None, output_format="PNG", channels="RGB")
        st.markdown(f"<div class='filter-name'>{name.title()}</div>", unsafe_allow_html=True)
        if st.button(f"✨ Apply {name.title()}", key=f"btn_{name}"):
            st.session_state["selected_filter"] = name

st.markdown("---")

# -----------------------
# Apply Filter + AI Suggestion
# -----------------------
if uploaded_file:
    st.markdown("### 🤖 AI Filter Suggestion")
    image = Image.open(uploaded_file).convert("RGB")
    mood, suggestion = analyze_image_mood(image)
    st.info(f"AI detects this photo is **{mood}**, suggested filter: **{suggestion.title()}** 🎬")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        selected_filter = st.session_state.get("selected_filter", suggestion)
        result = apply_lut(image, luts[selected_filter])
        st.image(result, caption=f"🎬 Applied: {selected_filter.title()}", use_container_width=True)

        # Save for download
        output = Image.fromarray(result)
        buf = BytesIO()
        output.save(buf, format="PNG")
        st.download_button(
            "📥 Download Filtered Image",
            data=buf.getvalue(),
            file_name=f"{selected_filter}_filtered.png",
            mime="image/png",
        )
else:
    st.info("Upload an image to try cinematic filters and AI suggestions.")
