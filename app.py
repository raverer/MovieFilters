# ==========================================
# 🎬 Movie Filter Studio (Oldboy + Dune + Grand Budapest + Oppenheimer + Rivendell Sunrise)
# ==========================================
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import requests
from io import BytesIO

st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")
st.title("🎞️ Movie Filter Studio")
st.caption("Apply cinematic color grades inspired by iconic films.")

# -----------------------
# Helpers
# -----------------------
def to_float32(img):
    return img.astype(np.float32) / 255.0

def to_uint8(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

# -----------------------
# Filters (as before)
# -----------------------
def oldboy_filter(img):
    img = img.copy()
    img = img * np.array([1.05, 1.02, 0.95])
    img = np.clip(img ** 1.05, 0, 1)
    h, w = img.shape[:2]
    vign = cv2.getGaussianKernel(w, w / 1.8) * cv2.getGaussianKernel(h, h / 1.8).T
    vign = np.dstack([vign / vign.max()] * 3)
    img *= (0.85 + 0.15 * vign)
    return np.clip(img, 0, 1)

def dune_filter(img):
    img = img.copy()
    warm = np.array([1.15, 1.07, 0.9])
    img *= warm
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    img = np.clip(img + blur * 0.05, 0, 1)
    h, w = img.shape[:2]
    vign = cv2.getGaussianKernel(w, w / 1.6) * cv2.getGaussianKernel(h, h / 1.6).T
    vign = np.dstack([vign / vign.max()] * 3)
    img *= (0.85 + 0.15 * vign)
    return np.clip(img, 0, 1)

def grand_budapest_filter(img):
    img = img.copy()
    pink_tone = np.array([1.1, 0.95, 1.05])
    img *= pink_tone
    img = np.clip(img ** 0.9, 0, 1)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
    img = np.clip(img + blur * 0.1, 0, 1)
    return np.clip(img, 0, 1)

def oppenheimer_filter(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    img = img * 0.8 + np.expand_dims(gray, 2) * 0.2
    warm_tone = np.array([1.15, 1.05, 0.90])
    cool_tone = np.array([0.95, 1.00, 1.05])
    luminance = np.expand_dims(gray, 2)
    img = img * (cool_tone * (1 - luminance) + warm_tone * luminance)
    img = np.clip((img - 0.45) * 1.25 + 0.45, 0, 1)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=4)
    img = np.clip(img + blur * 0.05, 0, 1)
    noise = np.random.normal(0, 0.015, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 1)
    return np.clip(img, 0, 1)

def rivendell_sunrise(img):
    img = np.clip(img ** 0.95, 0, 1)
    warm_tone = np.array([1.10, 1.03, 0.92])
    img *= warm_tone
    soft_blend = np.full_like(img, [1.05, 0.97, 0.95])
    luminance = np.expand_dims(cv2.cvtColor((img*255).astype(np.uint8),
                                           cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0, 2)
    img = img * (1 - 0.25 * luminance) + soft_blend * (0.25 * luminance)
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=3)
    img = np.clip(img + blur * 0.15, 0, 1)
    img = np.clip(img * 0.96 + 0.04, 0, 1)
    img = np.clip(np.power(img, 0.9), 0, 1)  # shadow lift / gentle contrast
    h, w, _ = img.shape
    vign = cv2.getGaussianKernel(w, w / 1.8) * cv2.getGaussianKernel(h, h / 1.8).T
    vign = np.dstack([vign / vign.max()] * 3)
    img *= (0.9 + 0.1 * vign)
    blur_small = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    img = np.clip(img + (img - blur_small) * 0.4, 0, 1)
    return np.clip(img, 0, 1)

# -----------------------
# Preview helpers
# -----------------------
@st.cache_resource
def load_base_preview(path="preview/base.jpg", resize_to=(420, 280)):
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
    # Generate preview thumbnails (numpy uint8)
    arr = np.array(base_image)
    base_f = to_float32(arr)
    previews = {
        "Oldboy": to_uint8(oldboy_filter(base_f)),
        "Dune 2021": to_uint8(dune_filter(base_f)),
        "Grand Budapest": to_uint8(grand_budapest_filter(base_f)),
        "Oppenheimer": to_uint8(oppenheimer_filter(base_f)),
        "Rivendell Sunrise": to_uint8(rivendell_sunrise(base_f))
    }
    return previews

# -----------------------
# UI layout: previews + controls + uploader
# -----------------------
base = load_base_preview()
previews = generate_previews(base)

st.markdown("### 🎞 Filter previews")
cols = st.columns(5)
names = list(previews.keys())
for i, name in enumerate(names):
    with cols[i]:
        st.image(previews[name], use_column_width=True, caption=name)

st.markdown("---")
filter_choice = st.selectbox(
    "🎥 Choose your cinematic look:",
    ["Oldboy", "Dune 2021", "Grand Budapest", "Oppenheimer", "Rivendell Sunrise"]
)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_f = to_float32(img_np)

    h, w = img_np.shape[:2]
    if max(h, w) > 4000:
        scale = 4000 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img_np = cv2.resize(img_np, new_size, interpolation=cv2.INTER_AREA)
        img_f = to_float32(img_np)
        st.info(f"Image resized to {img_np.shape[1]}x{img_np.shape[0]} for performance.")

    with st.spinner(f"Applying {filter_choice}..."):
        if filter_choice == "Oldboy":
            result_f = oldboy_filter(img_f)
        elif filter_choice == "Dune 2021":
            result_f = dune_filter(img_f)
        elif filter_choice == "Grand Budapest":
            result_f = grand_budapest_filter(img_f)
        elif filter_choice == "Oppenheimer":
            result_f = oppenheimer_filter(img_f)
        else:
            result_f = rivendell_sunrise(img_f)

    col1, col2 = st.columns(2)
    with col1:
        st.image(to_uint8(img_f), caption="Original", use_column_width=True)
    with col2:
        st.image(to_uint8(result_f), caption=f"{filter_choice} Look", use_column_width=True)

        # prepare download
        out_pil = Image.fromarray(to_uint8(result_f))
        buf = BytesIO()
        out_pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        st.download_button(
            "📥 Download Filtered Image",
            data=buf.getvalue(),
            file_name=f"{filter_choice.lower().replace(' ', '_')}.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Upload an image to apply a cinematic filter.")
