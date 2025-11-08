# ==========================================
# 🎥 MOVIE FILTER STUDIO (Oldboy + Dune Teal-Orange)
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
# 🎬 Oldboy Filter (keeps filmic texture)
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
# 🟠 Dune Teal-Orange Filter (preserve crispness)
# ==========================================
def dune_teal_orange_filter(image):
    img = np.array(image).astype(np.float32) / 255.0

    # slight filmic gamma for depth
    img = np.power(img, 0.95)

    # shadows -> teal
    shadows = np.clip(1.0 - (img * 2.2), 0, 1)
    teal_tint = np.array([0.0, 0.10, 0.26])
    img = img + teal_tint * shadows * 0.35

    # highlights -> orange
    highlights = np.clip((img - 0.5) * 2.0, 0, 1)
    orange_tint = np.array([0.22, 0.12, -0.03])
    img = img + orange_tint * highlights * 0.55

    # stronger cinematic contrast for Dune look
    img = apply_cinetone_curve(img, contrast=1.35, pivot=0.45)
    img = np.clip(img * np.array([1.05, 1.00, 0.95]), 0, 1)

    # clarity: small unsharp mask (keeps details)
    sharp = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
    clarity = np.clip(img + (img - sharp) * 0.9, 0, 1)

    # subtle grain (very low)
    h, w, _ = clarity.shape
    noise = np.random.normal(0, 1, (h, w, 1)).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    clarity = np.clip(clarity + (noise - 0.5) * 0.01, 0, 1)

    # vignette (soft)
    kernel_x = cv2.getGaussianKernel(w, w / 2.0)
    kernel_y = cv2.getGaussianKernel(h, h / 2.0)
    vignette = (kernel_y * kernel_x.T)
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    clarity = clarity * (0.8 + 0.2 * vignette)

    final = np.clip(clarity, 0, 1)
    return (final * 255).astype(np.uint8)

# -----------------------
# Previews: load base image (local first, fallback to Unsplash)
# -----------------------
@st.cache_resource
def load_base_preview(path="preview/base.jpg", resize_to=(500, 350)):
    if os.path.exists(path):
        base = Image.open(path).convert("RGB")
    else:
        # reliable fallback
        url = "https://images.unsplash.com/photo-1612690119274-8819a81c13a2?auto=format&fit=crop&w=1200&q=80"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        base = Image.open(BytesIO(resp.content)).convert("RGB")
    base = base.resize(resize_to, Image.LANCZOS)
    return base

@st.cache_resource
def generate_previews(base_image):
    # apply both filters to the base preview
    previews = {}
    previews["Oldboy"] = oldboy_fight_scene_effect_hd(np.array(base_image))
    previews["Dune Teal-Orange"] = dune_teal_orange_filter(base_image)
    return previews

# -----------------------
# UI
# -----------------------
st.title("🎞️ Movie Filter Studio")
st.caption("Cinematic filters: Oldboy and Dune (Teal–Orange). High fidelity, filmic texture.")

# load base + previews
base = load_base_preview()
previews = generate_previews(base)

# Preview gallery (two large cards)
st.markdown("### 🎞 Filter previews")
cols = st.columns(2)
i = 0
for name, img in previews.items():
    with cols[i % 2]:
        st.image(img, use_column_width=True, caption=name, output_format="PNG")
        if st.button(f"Apply {name}", key=f"apply_{name}"):
            st.session_state["selected_filter"] = name
    i += 1

# radio fallback/select
st.markdown("---")
filter_choice = st.radio("Or pick a filter:", ["Oldboy", "Dune Teal-Orange"],
                         index=0 if st.session_state.get("selected_filter", "Oldboy") == "Oldboy" else 1,
                         horizontal=True)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_arr = np.array(image)

    # resize extremely large images for performance but keep detail
    h, w = img_arr.shape[:2]
    if max(h, w) > 4000:
        scale = 4000 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img_arr = cv2.resize(img_arr, new_size, interpolation=cv2.INTER_AREA)
        st.info(f"Image resized to {img_arr.shape[1]}x{img_arr.shape[0]} for performance.")

    selected = st.session_state.get("selected_filter", filter_choice)
    # ensure radio selection overrides if user changed it directly
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

        # Download prepared
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
