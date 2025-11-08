import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="🎬 Movie Filter Studio", layout="wide")

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
# Streamlit UI
# -----------------------
st.title("🎥 Movie Filter Studio")
st.caption("Apply cinematic looks inspired by iconic movies.")

# Load LUTs
luts = load_all_luts()
filter_names = list(luts.keys())

# Layout
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎞 Choose Filter")
    selected_filter = st.radio("Movie Look", filter_names)

    st.markdown("**Preview Samples:**")
    preview_cols = st.columns(3)
    for i, name in enumerate(filter_names[:3]):
        with preview_cols[i]:
            st.image(f"https://placehold.co/150x100?text={name}", caption=name)

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        if st.button("✨ Apply Filter"):
            lut = luts[selected_filter]
            result = apply_lut(image, lut)
            st.image(result, caption=f"Filtered: {selected_filter}", use_container_width=True)
            st.download_button(
                "📥 Download Image",
                data=Image.fromarray(result).tobytes(),
                file_name=f"{selected_filter}_filtered.png",
                mime="image/png"
            )
    else:
        st.info("Upload an image to start filtering.")

st.markdown("---")
st.caption("© 2025 Movie Filter Studio — Cinematic Looks for Everyone.")
