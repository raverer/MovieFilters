# ==========================================
# 🎥 MOVIE FILTER STUDIO (Oldboy + Dune 2021)
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
# Utility: cine tone curve
# -----------------------
def apply_cinetone_curve(img, contrast=1.15, pivot=0.45):
    img = np.clip(img, 0, 1)
    return np.clip((img - pivot) * contrast + pivot, 0, 1)

# ==========================================
# 🎬 Oldboy Filter (with subtle green tint)
# ==========================================
def oldboy_fight_scene_effect_hd(img):
    img = img.astype(np.float32) / 255.0

    shadow_tint = np.array([0.00, 0.04, 0.12])
    highlight_tint = np.array([0.10, 0.05, -0.02])

    luminance = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    luminance = np.expand_dims(luminance, 2)
    shadow_mask = np.clip(1.0 - 2.0 * luminance, 0, 1)
    highlight_mask = np.clip(2.0 * (luminance - 0.5), 0, 1)

    img = img + shadow_tint * shadow_mask + highlight_tint * highlight_mask
    img = apply_cinetone_curve(img, contrast=1.25)

    # --- 🎨 Subtle green tint ---
    green_tint = np.array([0.0, 0.03, 0.0])
    img = np.clip(img + green_tint * 0.4, 0, 1)

    # color balance
    img[..., 2] *= 0.95
    img[..., 1] *= 1.05
    img[..., 0] *= 1.08
    img = np.clip(img, 0, 1)

    # Film grain (subtle, luminance-weighted)
    h, w, _ = img.shape
    grain_strength = 0.015
    noise = np.random.normal(0, 1, (h, w, 1)).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = (noise - 0.5) * 2.0
    img = np.clip(img + noise * grain_strength * (0.3 + luminance), 0, 1)

    # Vignette
    kernel_x = cv2.getGaussianKernel(w, w / 1.8)
    kernel_y = cv2.getGaussianKernel(h, h / 1.8)
    vignette = (kernel_y * kernel_x.T)
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    img = img * (0.7 + 0.3 * vignette)

    # Sharpen (adaptive, mild)
    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    sharp = np.clip(img + (img - img_blur) * 0.8, 0, 1)

    return (sharp * 255).astype(np.uint8)

# ==========================================
# 🌵 Dune (2021) Filter — cinematic teal-orange
# ==========================================
def dune_teal_orange_filter(image):
    img = np.array(image).astype(np.float32) / 255.0

    # Slight filmic gamma
    img = np.power(img, 0.97)

    # Shadows → teal
    shadows = np.clip(1.0 - (img * 2.0), 0, 1)
    teal_tint = np.array([0.0, 0.10, 0.22])
    img = img + teal_tint * shadows * 0.40

    # Highlights → soft ochre/orange warmth
    highlights = np.clip((img - 0.45) * 2.2, 0, 1)
    orange_tint = np.array([0.20, 0.10, -0.02])
    img = img + orange_tint * highlights * 0.45

    # Tone curve
    img = apply_cinetone_curve(img, contrast=1.25, pivot=0.42)

    # Gentle color rebalance
    img = np.clip(img * np.array([1.04, 1.00, 0.96]), 0, 1)

    # Atmospheric haze
    img = np.clip(img * 0.97 + 0.03, 0, 1)

    # Clarity (mild sharpening)
    sharp = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    clarity = np.clip(img + (img - sharp) * 0.8, 0, 1)

    # Subtle fine grain
    h, w, _ = clarity.shape
    noise = np.random.normal(0, 1, (h, w, 1)).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    clarity = np.clip(clarity + (noise - 0.5) * 0.008, 0, 1)

    # Soft vignette (half strength)
    kernel_x = cv2.getGaussianKernel(w, w / 2.2)
    kernel_y = cv2.getGaussianKernel(h, h / 2.2)
    vignette = (kernel_y * kernel_x.T)
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    clarity = clarity * (0.85 + 0.15 * vignette)

    final = np.clip(clarity, 0, 1)
    return (final * 255).astype(np.uint8)

# -----------------------
# Previews & UI
# -----------------------
@st.cache_resource
def load_base_preview(path="preview/base.jpg", resize_to=(500, 350)):
    if os.path.exists(path):
        base = Image.open(path).convert("RGB")
    else:
        url = "https://images.unsplash.com/photo-1612690119274-8819a81c13a2?auto=format&fit=crop&w=1200&q=80"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        base = Image.open(BytesIO(resp.content)).convert("RGB")
    base = base.resize(resize_to, Image.LANCZOS)
    return base

@st.cache_resource
def generate_previews(base_image):
    previews = {}
    previews["Oldboy"] = oldboy_fight_scene_effect_hd(np.array(base_image))
    previews["Dune(2021)"] = dune_teal_orange_filter(base_image)
    return previews

# -----------------------
# 🎨 Interface
# -----------------------
st.title("🎞️ Movie Filter Studio")
st.caption("Cinematic filters: High fidelity, filmic texture.")

base = load_base_preview()
previews = generate_previews(base)

st.markdown("### 🎞 Filter previews")
cols = st.columns(2)
i = 0
for name, img in previews.items():
    with cols[i % 2]:
        st.image(img, use_column_width=True, caption=name, output_format="PNG")
        if st.button(f"Apply {name}", key=f"apply_{name}"):
            st.session_state["selected_filter"] = name
    i += 1

st.markdown("---")
filter_choice = st.radio("Or pick a filter:", ["Oldboy", "Dune(2021)"],
                         index=0 if st.session_state.get("selected_filter", "Oldboy") == "Oldboy" else 1,
                         horizontal=True)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_arr = np.array(image)

    h, w = img_arr.shape[:2]
    if max(h, w) > 4000:
        scale = 4000 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img_arr = cv2.resize(img_arr, new_size, interpolation=cv2.INTER_AREA)
        st.info(f"Image resized to {img_arr.shape[1]}x{img_arr.shape[0]} for performance.")

    selected = st.session_state.get("selected_filter", filter_choice)
    selected = filter_choice if filter_choice else selected

    with st.spinner(f"Applying {selected}..."):
        if selected == "Oldboy":
            out = oldboy_fight_scene_effect_hd(img_arr)
        else:
            out = dune_teal_orange_filter(img_arr)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_arr, caption="Original", use_column_width=True)
    with col2:
        st.image(out, caption=f"{selected} Look", use_column_width=True)

        out_pil = Image.fromarray(out)
        buf = BytesIO()
        out_pil.save(buf, format="JPEG", quality=98)
        buf.seek(0)
        st.download_button(
            "📥 Download Filtered Image",
            data=buf.getvalue(),
            file_name=f"{selected.lower().replace(' ', '_')}.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Upload an image to apply a cinematic filter.")
