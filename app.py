# 🎬 Movie Filter Studio (Oldboy + Dune + Grand Budapest + Oppenheimer + Rivendell Sunrise)
# by Rana Moly + GPT-5

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="🎬 Movie Filter Studio", layout="centered")
st.title("🎞️ Movie Filter Studio")
st.caption("Apply cinematic color grades inspired by iconic films.")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
filter_choice = st.selectbox(
    "🎥 Choose your cinematic look:",
    ["Oldboy", "Dune 2021", "Grand Budapest", "Oppenheimer", "Rivendell Sunrise"]
)

def to_float32(img):
    return img.astype(np.float32) / 255.0

def to_uint8(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)

# --- Filters ---
def oldboy_filter(img):
    img = img.copy()
    img = img * [1.05, 1.02, 0.95]  # warmer tone with slight green lift
    img = np.clip(img ** 1.05, 0, 1)
    vignette = cv2.getGaussianKernel(img.shape[1], img.shape[1] / 1.8) * cv2.getGaussianKernel(img.shape[0], img.shape[0] / 1.8).T
    vignette = np.dstack([vignette / vignette.max()] * 3)
    img *= (0.85 + 0.15 * vignette)
    return np.clip(img, 0, 1)

def dune_filter(img):
    img = img.copy()
    warm = np.array([1.15, 1.07, 0.9])
    img *= warm
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    img = np.clip(img + blur * 0.05, 0, 1)
    vignette = cv2.getGaussianKernel(img.shape[1], img.shape[1] / 1.6) * cv2.getGaussianKernel(img.shape[0], img.shape[0] / 1.6).T
    vignette = np.dstack([vignette / vignette.max()] * 3)
    img *= (0.85 + 0.15 * vignette)
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
    luminance = np.expand_dims(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0, 2)
    img = img * (1 - 0.25 * luminance) + soft_blend * (0.25 * luminance)
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=3)
    img = np.clip(img + blur * 0.15, 0, 1)
    img = np.clip(img * 0.96 + 0.04, 0, 1)
    shadow_lift = np.power(img, 0.9)
    img = np.clip(shadow_lift, 0, 1)
    h, w, _ = img.shape
    vignette = cv2.getGaussianKernel(w, w / 1.8) * cv2.getGaussianKernel(h, h / 1.8).T
    vignette = np.dstack([vignette / vignette.max()] * 3)
    img *= (0.9 + 0.1 * vignette)
    blur_small = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    img = np.clip(img + (img - blur_small) * 0.4, 0, 1)
    return np.clip(img, 0, 1)

# --- Apply selected filter ---
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    img = to_float32(image)

    if filter_choice == "Oldboy":
        result = oldboy_filter(img)
    elif filter_choice == "Dune 2021":
        result = dune_filter(img)
    elif filter_choice == "Grand Budapest":
        result = grand_budapest_filter(img)
    elif filter_choice == "Oppenheimer":
        result = oppenheimer_filter(img)
    elif filter_choice == "Rivendell Sunrise":
        result = rivendell_sunrise(img)

    st.image([to_uint8(img), to_uint8(result)], caption=["Original", filter_choice], width=350)
    
    result_img = Image.fromarray(to_uint8(result))
    st.download_button("⬇️ Download Filtered Image", data=result_img.tobytes(), file_name=f"{filter_choice.lower().replace(' ', '_')}.png")
