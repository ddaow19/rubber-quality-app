import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="ระบบจำแนกคุณภาพยางพาราแผ่นดิบ",
    page_icon="🌿",
    layout="centered"
)

# ─── Class Labels ─────────────────────────────────────────
# ⚠️ แก้ให้ตรงกับ label ที่ใช้ตอน train
CLASS_NAMES = ["3", "4", "5", "B"]
IMG_SIZE = (224, 224)  # VGG16 standard input size

# ─── Load Model (cache ไว้ไม่ให้โหลดซ้ำ) ─────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("vgg16_best.keras")
    return model

model = load_model()

# ─── Preprocessing Function ───────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    return img_array

# ─── UI Header ────────────────────────────────────────────
st.title("🌿 ระบบจำแนกคุณภาพยางพาราแผ่นดิบ")
st.markdown("อัปโหลดภาพยางพาราแผ่นดิบ เพื่อให้ AI วิเคราะห์ระดับคุณภาพ")
st.divider()

# ─── Upload Image ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพยางพารา",
    type=["jpg", "jpeg", "png"]
)

# ─── Prediction ───────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🖼️ ภาพที่อัปโหลด")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("#### 📊 ผลการวิเคราะห์")
        with st.spinner("🔍 กำลังวิเคราะห์..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0])) * 100

        # แสดงผลหลัก
        st.success(f"✅ **{CLASS_NAMES[predicted_class]}**")
        st.metric(label="🎯 ความมั่นใจ", value=f"{confidence:.2f}%")

        st.divider()

        # แสดง probability bar ทุกคลาส
        st.markdown("#### 📈 ความน่าจะเป็นแต่ละเกรด")
        for cls, prob in zip(CLASS_NAMES, predictions[0]):
            st.progress(float(prob), text=f"{cls} : {prob*100:.1f}%")

else:
    # แสดง placeholder ตอนยังไม่อัปโหลด
    st.info("👆 กรุณาอัปโหลดภาพยางพาราเพื่อเริ่มการวิเคราะห์")