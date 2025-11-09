import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# ------------------------------------------
# Utility Functions
# ------------------------------------------
def to_float32(img):
    return np.clip(img.astype(np.float32) / 255.0, 0, 1)

def to_uint8(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def make_vignette(h, w, strength=0.15, radius_scale=1.8):
    """
    Creates a vignette mask (h x w x 3).
    strength: vignette intensity (0–1)
    radius_scale: softness of falloff
    """
    kernel_y = cv2.getGaussianKernel(h, h / radius_scale)
    kernel_x = cv2.getGaussianKernel(w, w / radius_scale)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vign = (1.0 - strength) + strength * mask[..., np.newaxis]
    return vign.astype(np.float32)

# ------------------------------------------
# Filter Functions
# ------------------------------------------

def oldboy_filter(img):
    img = img.copy()
    img[..., 0] *= 1.1   # Red
    img[..., 1] *= 0.95  # Green
    img[..., 2] *= 0.9   # Blue
    img = np.power(img, 1.1)
    h, w = img.shape[:2]
    vign = make_vignette(h, w, strength=0.3, radius_scale=1.8)
    img *= vign
    return np.clip(img, 0, 1)

def dune_filter(img):
    img = img.copy()
    img[..., 0] *= 1.2
    img[..., 1] *= 1.05
    img[..., 2] *= 0.85
    img = np.power(img, 1.05)
    h, w = img.shape[:2]
    vign = make_vignette(h, w, strength=0.2, radius_scale=2.0)
    img *= vign
    return np.clip(img, 0, 1)

def grand_budapest_filter(img):
    img = img.copy()
    img[..., 0] *= 1.05
    img[..., 1] *= 0.95
    img[..., 2] *= 1.1
    img = np.power(img, 0.95)
    h, w = img.shape[:2]
    vign = make_vignette(h, w, strength=0.1, radius_scale=1.8)
    img *= vign
    return np.clip(img, 0, 1)

def oppenheimer_filter(img):
    gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY) / 255.0
    img = np.stack([gray * 1.05, gray * 1.0, gray * 0.95], axis=-1)
    img = np.power(img, 1.1)
    h, w = img.shape[:2]
    vign = make_vignette(h, w, strength=0.2, radius_scale=1.5)
    img *= vign
    return np.clip(img, 0, 1)

def rivendell_filter(img):
    """Rivendell Sunrise – soft dreamlike tone, warm highlights, cool shadows"""
    img = img.copy()
    # Gentle warm-pink highlights & cool shadows
    img[..., 0] *= 1.1   # R
    img[..., 1] *= 1.05  # G
    img[..., 2] *= 1.15  # B
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.power(img, 0.95)
    h, w = img.shape[:2]
    vign = make_vignette(h, w, strength=0.1, radius_scale=1.8)
    img *= vign
    # soft warm glow overlay
    warm = np.array([1.1, 1.05, 1.0])
    img = img * 0.9 + warm * 0.1
    return np.clip(img, 0, 1)

# ------------------------------------------
# LUT (for export)
# ------------------------------------------
def create_lut(filter_func, size=33):
    """Generate LUT cube from a filter function"""
    levels = np.linspace(0, 1, size)
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    for r in range(size):
        for g in range(size):
            for b in range(size):
                color = np.array([levels[r], levels[g], levels[b]])
                color_img = np.ones((1, 1, 3), dtype=np.float32) * color
                filtered = filter_func(color_img)
                lut[r, g, b] = filtered
    return lut

def save_lut_cube(lut, filename="filter.cube"):
    """Save LUT as .cube file"""
    size = lut.shape[0]
    with open(filename, "w") as f:
        f.write(f"LUT_3D_SIZE {size}\n")
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    vals = lut[r, g, b]
                    f.write(f"{vals[0]} {vals[1]} {vals[2]}\n")
    return filename

# ------------------------------------------
# Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")
st.title("🎬 Movie Filter Studio (Oldboy + Dune + Grand Budapest + Oppenheimer + Rivendell Sunrise)")

filters = {
    "Oldboy": oldboy_filter,
    "Dune 2021": dune_filter,
    "Grand Budapest": grand_budapest_filter,
    "Oppenheimer": oppenheimer_filter,
    "Rivendell Sunrise": rivendell_filter
}

# Load base preview
@st.cache_data
def load_base_preview():
    img = np.zeros((280, 420, 3), dtype=np.uint8)
    img[..., :] = [190, 200, 210]  # neutral gray-blue preview
    return Image.fromarray(img)

@st.cache_data
def generate_previews(base_image):
    arr = np.array(base_image)
    base_f = to_float32(arr)
    previews = {name: to_uint8(func(base_f)) for name, func in filters.items()}
    return previews

base = load_base_preview()
previews = generate_previews(base)

st.markdown("### 🎞 Filter Previews")
cols = st.columns(len(previews))
for i, (name, preview) in enumerate(previews.items()):
    with cols[i]:
        st.image(preview, caption=name, use_container_width=True)

st.markdown("---")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
selected_filter = st.selectbox("Choose a movie filter:", list(filters.keys()))

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    img_f = to_float32(img_np)
    result = filters[selected_filter](img_f)

    st.image(
        [to_uint8(img_np), to_uint8(result)],
        caption=["Original", f"{selected_filter} Look"],
        use_container_width=True
    )

    # Download LUT button
    if st.button("🎨 Generate LUT (.cube)"):
        lut = create_lut(filters[selected_filter])
        filename = f"{selected_filter.replace(' ', '_').lower()}_filter.cube"
        save_lut_cube(lut, filename)
        with open(filename, "rb") as f:
            st.download_button(
                label="⬇️ Download LUT",
                data=f,
                file_name=filename,
                mime="application/octet-stream"
            )
