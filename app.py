# ==========================================
# 🎥 MOVIE FILTER STUDIO v3.0_final_base
# Oldboy + Dune + Grand Budapest + Oppenheimer + Lord Of The Rings
# ==========================================
import streamlit as st
import numpy as np
from PIL import Image
import cv2
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
# 🎬 Oldboy Filter
# ==========================================
def oldboy_fight_scene_effect_hd(img):
    img = img.astype(np.float32) / 255.0
    shadow_tint = np.array([0.00, 0.06, 0.10])
    highlight_tint = np.array([0.08, 0.07, -0.02])

    luminance = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    luminance = np.expand_dims(luminance, 2)
    shadow_mask = np.clip(1.0 - 2.0 * luminance, 0, 1)
    highlight_mask = np.clip(2.0 * (luminance - 0.5), 0, 1)

    img = img + shadow_tint * shadow_mask + highlight_tint * highlight_mask
    img = apply_cinetone_curve(img, contrast=1.25)

    img[..., 2] *= 0.95
    img[..., 1] *= 1.05
    img[..., 0] *= 1.08
    img = np.clip(img, 0, 1)

    h, w, _ = img.shape
    grain_strength = 0.015
    noise = np.random.normal(0, 1, (h, w, 1)).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = (noise - 0.5) * 2.0
    img = np.clip(img + noise * grain_strength * (0.3 + luminance), 0, 1)

    kernel_x = cv2.getGaussianKernel(w, w / 1.8)
    kernel_y = cv2.getGaussianKernel(h, h / 1.8)
    vignette = (kernel_y * kernel_x.T)
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    img = img * (0.7 + 0.3 * vignette)

    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    sharp = np.clip(img + (img - img_blur) * 0.8, 0, 1)
    return (sharp * 255).astype(np.uint8)

# ==========================================
# 🟠 Dune Teal-Orange Filter
# ==========================================
def dune_teal_orange_filter(image):
    img = np.array(image).astype(np.float32) / 255.0
    img = np.power(img, 0.95)

    shadows = np.clip(1.0 - (img * 2.2), 0, 1)
    teal_tint = np.array([0.0, 0.10, 0.26])
    img = img + teal_tint * shadows * 0.35

    highlights = np.clip((img - 0.5) * 2.0, 0, 1)
    orange_tint = np.array([0.22, 0.12, -0.03])
    img = img + orange_tint * highlights * 0.55

    img = apply_cinetone_curve(img, contrast=1.35, pivot=0.45)
    img = np.clip(img * np.array([1.05, 1.00, 0.95]), 0, 1)

    sharp = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
    clarity = np.clip(img + (img - sharp) * 0.9, 0, 1)

    h, w, _ = clarity.shape
    noise = np.random.normal(0, 1, (h, w, 1)).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    clarity = np.clip(clarity + (noise - 0.5) * 0.01, 0, 1)

    kernel_x = cv2.getGaussianKernel(w, w / 2.0)
    kernel_y = cv2.getGaussianKernel(h, h / 2.0)
    vignette = (kernel_y * kernel_x.T)
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    clarity = clarity * (0.8 + 0.2 * vignette)

    final = np.clip(clarity, 0, 1)
    return (final * 255).astype(np.uint8)

# ==========================================
# 🎀 Grand Budapest Hotel Filter
# ==========================================
def apply_grand_budapest_filmic(image):
    img = image.astype(np.float32) / 255.0
    pastel_tint = np.array([1.05, 0.96, 1.10])
    img = np.clip(img * pastel_tint, 0, 1)
    img = np.clip((img - 0.05) * 1.12 + 0.05, 0, 1)

    shadows = np.power(img, 1.1)
    highlights = np.power(img, 0.9)
    img = np.clip(0.6 * shadows + 0.4 * highlights, 0, 1)
    img = np.clip(img * np.array([1.05, 1.02, 0.96]), 0, 1)

    detail = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = np.clip(img + (img - detail) * 0.6, 0, 1)

    h, w, _ = img.shape
    noise = np.random.normal(0, 0.02, (h, w, 3))
    img = np.clip(img + noise, 0, 1)

    X_kernel = cv2.getGaussianKernel(w, w / 1.8)
    Y_kernel = cv2.getGaussianKernel(h, h / 1.8)
    kernel = Y_kernel * X_kernel.T
    vignette = kernel / kernel.max()
    vignette = np.dstack([vignette] * 3)
    img = np.clip(img * (0.9 + 0.1 * vignette), 0, 1)

    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharp = np.clip(img + (img - blur) * 0.5, 0, 1)
    return (sharp * 255).astype(np.uint8)

# ==========================================
# ⚛️ Oppenheimer Filter
# ==========================================
def oppenheimer_filter(image):
    img = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    img = img * 0.8 + np.expand_dims(gray, 2) * 0.2

    warm_tone = np.array([1.15, 1.05, 0.90])
    cool_tone = np.array([0.95, 1.00, 1.05])
    luminance = np.expand_dims(gray, 2)
    img = img * (cool_tone * (1 - luminance) + warm_tone * luminance)

    img = apply_cinetone_curve(img, contrast=1.25, pivot=0.45)

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=4)
    img = np.clip(img + blur * 0.05, 0, 1)

    h, w, _ = img.shape
    noise = np.random.normal(0, 0.015, (h, w, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 1)

    X_kernel = cv2.getGaussianKernel(w, w / 1.5)
    Y_kernel = cv2.getGaussianKernel(h, h / 1.5)
    vignette = Y_kernel * X_kernel.T
    vignette = np.dstack([vignette / vignette.max()] * 3)
    img *= (0.8 + 0.2 * vignette)

    blur_small = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = np.clip(img + (img - blur_small) * 0.8, 0, 1)
    return (img * 255).astype(np.uint8)

# ==========================================
# 🌅 Rivendell Sunrise Filter
# ==========================================
def rivendell_sunrise(image):
    img = image.astype(np.float32) / 255.0
    img = np.clip(img ** 0.95, 0, 1)
    warm_tone = np.array([1.10, 1.03, 0.92])
    img *= warm_tone

    soft_blend = np.full_like(img, [1.05, 0.97, 0.95])
    luminance = np.expand_dims(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0, 2)
    img = img * (1 - 0.25 * luminance) + soft_blend * (0.25 * luminance)

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    img = np.clip(img + blur * 0.12, 0, 1)

    img = np.clip(img * 0.96 + 0.04, 0, 1)
    shadow_lift = np.power(img, 0.9)
    img = np.clip(shadow_lift, 0, 1)

    h, w, _ = img.shape
    X_kernel = cv2.getGaussianKernel(w, w / 1.8)
    Y_kernel = cv2.getGaussianKernel(h, h / 1.8)
    vignette = Y_kernel * X_kernel.T
    vignette = np.dstack([vignette / vignette.max()] * 3)
    img *= (0.9 + 0.1 * vignette)

    blur_small = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharp = np.clip(img + (img - blur_small) * 0.45, 0, 1)
    return (sharp * 255).astype(np.uint8)

# -----------------------
# Preview setup
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
def generate_previews(base_image):
    previews = {
        "Oldboy": oldboy_fight_scene_effect_hd(np.array(base_image)),
        "Dune": dune_teal_orange_filter(base_image),
        "Grand Budapest Hotel": apply_grand_budapest_filmic(np.array(base_image)),
        "Oppenheimer": oppenheimer_filter(np.array(base_image)),
        "Lord Of The Rings": rivendell_sunrise(np.array(base_image))
    }
    return previews

# -----------------------
# Streamlit UI
# -----------------------
st.title("🎞️ Movie Filter Studio")
st.caption("Cinematic filters")

base = load_base_preview()
previews = generate_previews(base)

st.markdown("### 🎞 Filter previews")
cols = st.columns(5)
for i, (name, img) in enumerate(previews.items()):
    with cols[i]:
        st.image(img, use_column_width=True, caption=name)

st.markdown("---")
filter_choice = st.selectbox(
    "🎬 Choose your cinematic filter:",
    ["Oldboy", "Dune", "Grand Budapest Hotel", "Oppenheimer", "Lord Of The Rings"]
)

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

    with st.spinner(f"Applying {filter_choice}..."):
        if filter_choice == "Oldboy":
            out = oldboy_fight_scene_effect_hd(img_arr)
        elif filter_choice == "Dune":
            out = dune_teal_orange_filter(img_arr)
        elif filter_choice == "Grand Budapest Hotel":
            out = apply_grand_budapest_filmic(img_arr)
        elif filter_choice == "Oppenheimer":
            out = oppenheimer_filter(img_arr)
        else:
            out = rivendell_sunrise(img_arr)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_arr, caption="Original", use_column_width=True)
    with col2:
        st.image(out, caption=f"{filter_choice} Look", use_column_width=True)

        out_pil = Image.fromarray(out)
        buf = BytesIO()
        out_pil.save(buf, format="JPEG", quality=98)
        buf.seek(0)
        st.download_button(
            "📥 Download Filtered Image",
            data=buf.getvalue(),
            file_name=f"{filter_choice.lower().replace(' ', '_')}.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Upload an image to apply a cinematic filter.")
