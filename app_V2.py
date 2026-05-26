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
CLASS_NAMES = ["3", "4", "5", "B"]
IMG_SIZE = (224, 224)

# ─── Class Details ────────────────────────────────────────
CLASS_DETAILS = {
    "3": {
        "emoji": "🟢",
        "title": "ยางชั้น 3 — คุณภาพดี",
        "color": "green",
        "badge": "✅ คุณภาพดี",
        "details": [
            ("🎨 สี", "น้ำตาลอ่อนหรือเหลืองทองสม่ำเสมอ"),
            ("💪 เนื้อยาง", "มีความยืดหยุ่นดี ความใสและความสะอาดสูง"),
            ("🔵 รูพรุน", "น้อยมากหรือแทบไม่มีเลย"),
            ("⚠️ ตำหนิที่อนุโลม", "ฟองอากาศขนาดเล็กมาก (Pinhead size) ได้เพียงเล็กน้อย"),
            ("❌ ตำหนิที่ไม่อนุโลม", "ห้ามมีสิ่งเจือปนหรือคราบเชื้อราที่มองเห็นได้ชัดเจน"),
        ]
    },
    "4": {
        "emoji": "🟡",
        "title": "ยางชั้น 4 — คุณภาพปานกลาง",
        "color": "orange",
        "badge": "🔶 คุณภาพปานกลาง",
        "details": [
            ("🎨 สี", "เข้มกว่าชั้น 3 เล็กน้อย หรือมีสีไม่สม่ำเสมอในบางจุด (ความด่าง)"),
            ("💪 เนื้อยาง", "มาตรฐานทั่วไป พบมากที่สุดในตลาด"),
            ("🔵 รูพรุน", "พบฟองอากาศได้บ้างเล็กน้อย"),
            ("⚠️ ตำหนิที่อนุโลม", "คราบแห้งของน้ำยาง หรือจุดด่างดำขนาดเล็กจากฝุ่นละออง"),
            ("❌ ตำหนิที่ไม่อนุโลม", "ห้ามมีรอยไหม้หรือคราบราที่เป็นอันตรายต่อเนื้อยาง"),
        ]
    },
    "5": {
        "emoji": "🟠",
        "title": "ยางชั้น 5 — คุณภาพต่ำ",
        "color": "red",
        "badge": "🔴 คุณภาพต่ำ",
        "details": [
            ("🎨 สี", "สีเข้มหรือสีคล้ำ (Opaque)"),
            ("💪 เนื้อยาง", "อาจมีความกระด้างหรือยืดหยุ่นลดลง"),
            ("🔵 รูพรุน", "พบฟองอากาศขนาดใหญ่ขึ้น"),
            ("⚠️ ตำหนิที่อนุโลม", "จุดตำหนิจากสิ่งสกปรก คราบเขม่า หรือรอยเปื้อนจากการตากยางที่ไม่ดี"),
            ("❌ เงื่อนไข", "ยังต้องอยู่ในเกณฑ์ที่สามารถนำไปแปรรูปในอุตสาหกรรมได้"),
        ]
    },
    "B": {
        "emoji": "🔴",
        "title": "Class B — ยางมีตำหนิ / ไม่ผ่านมาตรฐาน",
        "color": "red",
        "badge": "⛔ ไม่ผ่านมาตรฐาน",
        "details": [
            ("🎨 สภาพ", "แผ่นยางมีความชื้นสูงเกินไป หรืออาการยางเหนียว (Sticky)"),
            ("💪 เนื้อยาง", "แผ่นยางติดกันเป็นปึก หรือเสียหายอย่างชัดเจน"),
            ("🔵 รูพรุน", "มีรอยไหม้อย่างรุนแรง"),
            ("❌ ตำหนิหลัก", "พบเชื้อรา (Mold) อย่างชัดเจน หรือสิ่งปลอมปนขนาดใหญ่ เช่น เปลือกไม้ ก้อนหิน"),
            ("💰 ผลกระทบ", "ถูกหักราคาจากสหกรณ์ค่อนข้างมาก"),
        ]
    },
}

# ─── Load Model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("vgg16_best.keras")
    return model

model = load_model()

# ─── Preprocessing ────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ─── UI Header ────────────────────────────────────────────
st.title("🌿 ระบบจำแนกคุณภาพยางพาราแผ่นดิบ")
st.caption("สหกรณ์กองทุนสวนยางแคนดง จำกัด | มาตรฐาน Air Dried Sheets (ADS)")
st.divider()

# ─── Upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพยางพาราแผ่นดิบ",
    type=["jpg", "jpeg", "png"]
)

# ─── Prediction & Display ─────────────────────────────────
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

        predicted_label = CLASS_NAMES[predicted_class]
        info = CLASS_DETAILS[predicted_label]

        # แสดงผลหลัก
        st.markdown(f"## {info['emoji']} {predicted_label}")
        st.markdown(f"**{info['title']}**")
        st.metric(label="🎯 ความมั่นใจ", value=f"{confidence:.2f}%")

        st.divider()

        # probability bar ทุกคลาส
        st.markdown("#### 📈 ความน่าจะเป็นแต่ละคลาส")
        for cls, prob in zip(CLASS_NAMES, predictions[0]):
            st.progress(float(prob), text=f"{cls} : {prob*100:.1f}%")

    st.divider()

    # ─── Class Detail Card ────────────────────────────────
    st.markdown(f"### {info['emoji']} รายละเอียด {predicted_label} — {info['title']}")

    for icon_label, desc in info["details"]:
        st.markdown(f"- **{icon_label}:** {desc}")

    # ─── คำแนะนำเพิ่มเติม ─────────────────────────────────
    st.divider()
    st.markdown("#### 💡 คำแนะนำ")

    if predicted_label == "3":
        st.success("ยางคุณภาพดี เหมาะสำหรับการจำหน่ายในราคาสูงสุด ควรรักษาคุณภาพในกระบวนการผลิตต่อไป")
    elif predicted_label == "4":
        st.info("ยางคุณภาพมาตรฐาน สามารถจำหน่ายได้ในราคาปกติ ควรปรับปรุงกระบวนการรีดและตากยางเพื่อยกระดับเป็น Class 3")
    elif predicted_label == "5":
        st.warning("ยางคุณภาพต่ำ ราคาจำหน่ายจะต่ำกว่ามาตรฐาน ควรตรวจสอบกระบวนการตากยางและความสะอาดในการผลิต")
    elif predicted_label == "B":
        st.error("ยางไม่ผ่านมาตรฐาน จะถูกหักราคาอย่างมาก ควรตรวจสอบความชื้น เชื้อรา และสิ่งปลอมปนในกระบวนการผลิตทันที")

else:
    st.info("👆 กรุณาอัปโหลดภาพยางพาราแผ่นดิบเพื่อเริ่มการวิเคราะห์")

    # ─── แสดงตารางมาตรฐานคลาส ─────────────────────────────
    st.divider()
    st.markdown("### 📋 มาตรฐานการจัดชั้นคุณภาพยางแผ่นดิบ (ADS)")
    st.markdown("ตามเกณฑ์ **สหกรณ์กองทุนสวนยางแคนดง จำกัด** และมาตรฐาน Green Book")

    for cls, info in CLASS_DETAILS.items():
        with st.expander(f"{info['emoji']} {cls} — {info['title']}"):
            for icon_label, desc in info["details"]:
                st.markdown(f"- **{icon_label}:** {desc}")