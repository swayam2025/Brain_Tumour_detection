import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection System")

st.write("""
Upload an MRI image to:
- Detect tumor
- Identify tumor type
- Segment tumor region
- Calculate tumor area
""")

# ---------------- CUSTOM LOSS (Segmentation) ----------------
def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    cls_model = tf.keras.models.load_model("models/classification.h5")
    seg_model = tf.keras.models.load_model(
        "models/segmentation.h5",
        custom_objects={
            "dice_loss": dice_loss,
            "dice_coef": dice_coef
        }
    )
    return cls_model, seg_model

cls_model, seg_model = load_models()
st.success("Models loaded successfully ✅")

# ---------------- CLASS LABELS ----------------
class_names = ['notumor', 'pituitary', 'meningioma', 'glioma']

# ---------------- PREPROCESS FUNCTIONS ----------------
def preprocess_classification(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_segmentation(image):
    img = image.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- SEGMENTATION VISUALIZATION ----------------
def overlay_mask(original, mask):
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    overlay = original.copy()
    overlay[mask == 1] = [255, 0, 0]  # red tumor region

    blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
    return blended, mask

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # ---------------- CLASSIFICATION ----------------
    input_cls = preprocess_classification(image)
    preds = cls_model.predict(input_cls)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.subheader("🧪 Classification Result")
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # ---------------- SEGMENTATION (ONLY IF TUMOR) ----------------
    if pred_class != "notumor":
        st.subheader("🎯 Tumor Segmentation Result")

        input_seg = preprocess_segmentation(image)
        mask_pred = seg_model.predict(input_seg)[0, :, :, 0]

        original_np = np.array(image)
        overlay_img, binary_mask = overlay_mask(original_np, mask_pred)

        # Tumor area (pixel count)
        tumor_pixels = np.sum(binary_mask)
        st.write(f"**Tumor Area (in pixels):** {tumor_pixels}")

        st.image(
            overlay_img,
            caption="Tumor Segmentation Overlay",
            use_container_width=True
        )

    else:
        st.info("✅ No tumor detected, segmentation skipped.")
