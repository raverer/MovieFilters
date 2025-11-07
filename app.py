import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

# ==========================================
# 🎞️ Oldboy Fight Scene Filter (True HD v4.0)
# ==========================================
def apply_cinetone_curve(img, contrast=1.15, pivot=0.45):
    """Apply a smooth S-curve contrast boost for cinematic tone mapping."""
    img = np.clip(img, 0, 1)
    return np.clip((img - pivot) * contrast + pivot, 0, 1)

def oldboy_fight_scene_effect_hd(img):
    img = img.astype(np.float32) / 255.0

    # --- Step 1: Split-tone (cool shadows / warm highlights) ---
    shadow_tint = np.array([0.00, 0.04, 0.12], dtype=np.float32)
    highlight_tint = np.array([0.10, 0.05, -0.02], dtype=np.float32)

    luminance = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    luminance = np.expand_dims(luminance, 2)
    shadow_mask = np.clip(1.0 - 2.0 * luminance, 0, 1)
    highlight_mask = np.clip(2.0 * (luminance - 0.5), 0, 1)

    img += shadow_tint * shadow_mask
    img += highlight_tint * highlight_mask

    # --- Step 2: Apply cinematic tone curve ---
    img = apply_cinetone_curve(img, contrast=1.25)

    # --- Step 3: Slightly crushed blacks and teal-mids ---
    img[..., 2] *= 0.95   # reduce reds
    img[..., 1] *= 1.05   # enhance greens
    img[..., 0] *= 1.08   # cooler blues
    img = np.clip(img, 0, 1)

    # --- Step 4: Film grain using luminance modulation ---
    h, w, _ = img.shape
    grain_strength = 0.015  # subtle filmic grain
    noise = np.random.normal(0, 1, (h, w, 1))
    grain = (noise - noise.min()) / (noise.max() - noise.min())
    grain = (grain - 0.5) * 2.0
    img = np.clip(img + grain * grain_strength * (0.3 + luminance), 0, 1)

    # --- Step 5: Vignette ---
    kernel_x = cv2.getGaussianKernel(w, w / 1.8)
    kernel_y = cv2.getGaussianKernel(h, h / 1.8)
    vignette = kernel_y * kernel_x.T
    vignette = vignette / vignette.max()
    vignette = np.dstack([vignette] * 3)
    img *= (0.7 + 0.3 * vignette)

    # --- Step 6: Adaptive sharpening ---
    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    sharp = np.clip(img + (img - img_blur) * 0.8, 0, 1)

    return (sharp * 255).astype(np.uint8)

# ==========================================
# 📏 Resize helper
# ==========================================
def resize_if_needed(img, max_dim=4000):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        print(f"🔄 Resized to {img.shape[1]}x{img.shape[0]}")
    else:
        print(f"✅ Using full native resolution ({w}x{h})")
    return img

# ==========================================
# 🖼️ Upload + Apply Filter
# ==========================================
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = np.array(Image.open(img_path).convert("RGB"))
img = resize_if_needed(img, max_dim=4000)

filtered_img = oldboy_fight_scene_effect_hd(img)

# ==========================================
# 💾 Save + Display
# ==========================================
output_path = "oldboy_fight_scene_hd_output.jpg"
Image.fromarray(filtered_img).save(output_path, quality=98)

plt.figure(figsize=(14, 10))
plt.imshow(filtered_img)
plt.axis("off")
plt.title("🎬 Oldboy Fight Scene Filter – True HD Output (v4.0)", fontsize=14)
plt.show()

print(f"✅ Saved HD output: {output_path}")
