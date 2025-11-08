# ==========================================
# 🎥 MOVIE FILTER STUDIO (Oldboy + Dune Teal-Orange)
# ==========================================
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io

st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")

# ==========================================
# 🎬 OLD BOY FILTER
# ==========================================
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


# ==========================================
# 🟠 DUNE TEAL-ORANGE FILTER
# ==========================================
def dune_teal_orange_filter(image):
    img = np.array(image).astype(np.float32) / 255.0

    # 1️⃣ Create a cinematic contrast curve
    img = np.power(img, 0.95)  # preserve shadow depth

    # 2️⃣ Push shadows toward teal
    shadows = np.clip(1.0 - (img * 2.2), 0, 1)
    teal_tint = np.array([0.0, 0.1, 0.25])
    img += teal_tint * shadows * 0.35

    # 3️⃣ Push highlights toward orange
    highlights = np.clip((img - 0.5) * 2.0, 0, 1)
    orange_tint = np.array([0.25, 0.15, -0.05])
    img += orange_tint * highlights * 0.55

    # 4️⃣ Adjust color balance and contrast
    img = apply_cinetone_curve(img, contrast=1.35, pivot=0.45)
    img = np.clip(img * np.array([1.05, 1.0, 0.95]), 0, 1)

    # 5️⃣ Add clarity and preserve crisp details
    sharp = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
    clarity = np.clip(img + (img - sharp) * 0.9, 0, 1)

    # 6️⃣ Film grain + subtle vignette
    h, w, _ = clarity.shape
    noise = np.random.normal(0, 1, (h, w, 1))
    grain = (noise - noise.min()) / (noise.max() - noise.min())
    clarity = np.clip(clarity + grain * 0.01, 0, 1)

    # vignette
    kernel_x = cv2.getGaussianKernel(w, w / 2.0)
    kernel_y = cv2.getGaussianKernel(h, h / 2.0)
    vignette = kernel_y * kernel_x.T
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    clarity *= (0.8 + 0.2 * vignette)

    # Preserve tonal richness
    final = np.clip(clarity, 0, 1)
    return (final * 255).astype(np.uint8)


# ==========================================
# 🧠 Streamlit Interface
# ==========================================
st.title("🎞️ Movie Filter Studio")
st.caption("Cinematic filters inspired by *Oldboy* and *Dune* — sharp, textured, and filmic.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

filter_choice = st.radio(
    "🎬 Choose a cinematic tone",
    ["Oldboy", "Dune Teal-Orange"],
    horizontal=True
)

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # Resize large images to 4000px max
    h, w = image.shape[:2]
    if max(h, w) > 4000:
        scale = 4000 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        st.info(f"Image resized to {image.shape[1]}x{image.shape[0]} for performance optimization.")

    # Apply selected filter
    with st.spinner(f"🎥 Applying {filter_choice} cinematic tone..."):
        if filter_choice == "Oldboy":
            filtered_img = oldboy_fight_scene_effect_hd(image)
        else:
            filtered_img = dune_teal_orange_filter(image)

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(filtered_img, caption=f"{filter_choice} Look", use_container_width=True)

    # Download Button
    img_pil = Image.fromarray(filtered_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=98)
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Download Cinematic Image",
        data=byte_im,
        file_name=f"{filter_choice.lower().replace(' ', '_')}.jpg",
        mime="image/jpeg"
    )

else:
    st.info("👆 Upload your image to apply the cinematic filter.")
