# ==========================================
# 🎥 MOVIE FILTER STUDIO (Oldboy + Dune)
# ==========================================
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from sklearn.cluster import KMeans

st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")

# -----------------------
# 🎞 LUT loading utilities
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
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)
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
# 🎬 Oldboy Filter Definition
# -----------------------
def apply_cinetone_curve(img, contrast=1.15, pivot=0.45):
    img = np.clip(img, 0, 1)
    return np.clip((img - pivot) * contrast + pivot, 0, 1)

def oldboy_fight_scene_effect_hd(img):
    img = img.astype(np.float32) / 255.0
    shadow_tint = np.array([0.00, 0.04, 0.12])
    highlight_tint = np.array([0.10, 0.05, -0.02])
    luminance = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    luminance = np.expand_dims(luminance, 2)
    shadow_mask = np.clip(1.0 - 2.0 * luminance, 0, 1)
    highlight_mask = np.clip(2.0 * (luminance - 0.5), 0, 1)
    img += shadow_tint * shadow_mask
    img += highlight_tint * highlight_mask
    img = apply_cinetone_curve(img, contrast=1.25)
    img[..., 2] *= 0.95
    img[..., 1] *= 1.05
    img[..., 0] *= 1.08
    img = np.clip(img, 0, 1)
    # Film grain
    h, w, _ = img.shape
    grain_strength = 0.015
    noise = np.random.normal(0, 1, (h, w, 1))
    grain = (noise - noise.min()) / (noise.max() - noise.min())
    grain = (grain - 0.5) * 2.0
    img = np.clip(img + grain * grain_strength * (0.3 + luminance), 0, 1)
    # Vignette
    kernel_x = cv2.getGaussianKernel(w, w / 1.8)
    kernel_y = cv2.getGaussianKernel(h, h / 1.8)
    vignette = kernel_y * kernel_x.T
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    img *= (0.7 + 0.3 * vignette)
    # Sharpen
    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    sharp = np.clip(img + (img - img_blur) * 0.8, 0, 1)
    return (sharp * 255).astype(np.uint8)

# -----------------------
# 🟢 Dune cinematic filter
# -----------------------
def apply_dune_filter(image):
    img = np.array(image).astype(np.float32) / 255.0
    orange_tone = np.array([1.15, 1.05, 0.85])
    img = np.clip(img * orange_tone, 0, 1)
    shadows = np.minimum(img, 0.5)
    img = img - shadows * 0.08 * np.array([0.1, -0.1, -0.2])
    img = np.clip((img - 0.1) * 1.2 + 0.05, 0, 1)
    rows, cols = img.shape[:2]
    X_kernel = cv2.getGaussianKernel(cols, 250)
    Y_kernel = cv2.getGaussianKernel(rows, 250)
    kernel = Y_kernel * X_kernel.T
    mask = kernel / kernel.max()
    vignette = img * (0.5 * mask[..., np.newaxis] + 0.5)
    vignette = np.clip(vignette * 1.1, 0, 1)
    vignette[:, :, 0] *= 1.05
    vignette[:, :, 2] *= 0.95
    return (vignette * 255).astype(np.uint8)

# -----------------------
# Generate previews
# -----------------------
@st.cache_resource
def generate_previews(luts, base_image_path="preview/base.jpg"):
    import requests
    from io import BytesIO
    if not os.path.exists(base_image_path):
        url = "https://picsum.photos/800/600"
        base = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    else:
        base = Image.open(base_image_path).convert("RGB")
    base = base.resize((400, 280))
    previews = {}
    for name, lut in luts.items():
        previews[name] = apply_lut(base, lut)
    previews["oldboy"] = oldboy_fight_scene_effect_hd(np.array(base))
    previews["dune"] = apply_dune_filter(base)
    return previews

# -----------------------
# Mood analyzer
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
st.caption("Apply cinematic looks inspired by legendary films.")

luts = load_all_luts()
previews = generate_previews(luts)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

# 🎞 Preview gallery
st.subheader("🎞 Choose Your Cinematic Look")
cols = st.columns(3)
for i, (name, img) in enumerate(previews.items()):
    with cols[i % 3]:
        st.image(img, caption=name.title(), use_container_width=True)
        if st.button(f"Apply {name.title()}"):
            st.session_state["selected_filter"] = name

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    mood, suggestion = analyze_image_mood(image)
    st.info(f"AI detects this photo is **{mood}**, suggested filter: **{suggestion}** 🎬")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        selected_filter = st.session_state.get("selected_filter", suggestion)
        if selected_filter == "oldboy":
            result = oldboy_fight_scene_effect_hd(np.array(image))
        elif selected_filter == "dune":
            result = apply_dune_filter(image)
        else:
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
