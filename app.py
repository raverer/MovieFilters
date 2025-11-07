import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# ===========================
# 🎬 Oldboy Filter Definition
# ===========================
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


# ===========================
# 🧠 Streamlit UI
# ===========================
st.set_page_config(page_title="🎬 Movie Filter Lab", layout="wide")

st.title("🎥 Oldboy Fight Scene Filter – AI Movie Color Lab")
st.markdown("Upload your image and experience the **cinematic tones** inspired by Park Chan-wook’s *Oldboy (2003)*.")

uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    h, w = image.shape[:2]
    if max(h, w) > 4000:
        scale = 4000 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        st.info(f"Image resized to {image.shape[1]}x{image.shape[0]} for optimal processing.")

    # Apply filter
    with st.spinner("🎞️ Applying Oldboy cinematic tone..."):
        filtered_img = oldboy_fight_scene_effect_hd(image)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(filtered_img, caption="Oldboy Fight Scene Look", use_container_width=True)

    # Download button
    img_pil = Image.fromarray(filtered_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=98)
    byte_im = buf.getvalue()
    st.download_button(
        label="💾 Download Cinematic Image",
        data=byte_im,
        file_name="oldboy_fight_scene_result.jpg",
        mime="image/jpeg"
    )

else:
    st.info("👆 Upload an image to apply the filter.")
