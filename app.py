import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta


# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="ระบบจำแนกคุณภาพยางพาราแผ่นดิบ",
    page_icon="🌿",
    layout="wide"
)

# ─── Constants ────────────────────────────────────────────
CLASS_NAMES = ["3", "4", "5", "B"]
IMG_SIZE    = (224, 224)
CSV_FILE    = "predictions.csv"

# ─── Class Details ────────────────────────────────────────
CLASS_DETAILS = {
    "3": {
        "emoji": "🟢", "title": "ยางชั้น 3 — คุณภาพดี",
        "details": [
            ("🎨 สี",               "น้ำตาลอ่อนหรือเหลืองทองสม่ำเสมอ"),
            ("💪 เนื้อยาง",         "มีความยืดหยุ่นดี ความใสและความสะอาดสูง"),
            ("🔵 รูพรุน",           "น้อยมากหรือแทบไม่มีเลย"),
            ("⚠️ ตำหนิที่อนุโลม",  "ฟองอากาศขนาดเล็กมาก (Pinhead size) ได้เพียงเล็กน้อย"),
            ("❌ ตำหนิที่ไม่อนุโลม","ห้ามมีสิ่งเจือปนหรือคราบเชื้อราที่มองเห็นได้ชัดเจน"),
        ]
    },
    "4": {
        "emoji": "🟡", "title": "ยางชั้น 4 — คุณภาพปานกลาง",
        "details": [
            ("🎨 สี",               "เข้มกว่าชั้น 3 เล็กน้อย หรือมีสีไม่สม่ำเสมอในบางจุด"),
            ("💪 เนื้อยาง",         "มาตรฐานทั่วไป พบมากที่สุดในตลาด"),
            ("🔵 รูพรุน",           "พบฟองอากาศได้บ้างเล็กน้อย"),
            ("⚠️ ตำหนิที่อนุโลม",  "คราบแห้งของน้ำยาง หรือจุดด่างดำขนาดเล็กจากฝุ่นละออง"),
            ("❌ ตำหนิที่ไม่อนุโลม","ห้ามมีรอยไหม้หรือคราบราที่เป็นอันตรายต่อเนื้อยาง"),
        ]
    },
    "5": {
        "emoji": "🟠", "title": "ยางชั้น 5 — คุณภาพต่ำ",
        "details": [
            ("🎨 สี",               "สีเข้มหรือสีคล้ำ (Opaque)"),
            ("💪 เนื้อยาง",         "อาจมีความกระด้างหรือยืดหยุ่นลดลง"),
            ("🔵 รูพรุน",           "พบฟองอากาศขนาดใหญ่ขึ้น"),
            ("⚠️ ตำหนิที่อนุโลม",  "จุดตำหนิจากสิ่งสกปรก คราบเขม่า หรือรอยเปื้อน"),
            ("❌ เงื่อนไข",         "ยังต้องอยู่ในเกณฑ์ที่สามารถนำไปแปรรูปในอุตสาหกรรมได้"),
        ]
    },
    "B": {
        "emoji": "🔴", "title": "Class B — ยางมีตำหนิ / ไม่ผ่านมาตรฐาน",
        "details": [
            ("🎨 สภาพ",             "แผ่นยางมีความชื้นสูงเกินไป หรืออาการยางเหนียว (Sticky)"),
            ("💪 เนื้อยาง",         "แผ่นยางติดกันเป็นปึก หรือเสียหายอย่างชัดเจน"),
            ("🔵 รูพรุน",           "มีรอยไหม้อย่างรุนแรง"),
            ("❌ ตำหนิหลัก",        "พบเชื้อรา (Mold) หรือสิ่งปลอมปนขนาดใหญ่ เช่น เปลือกไม้ ก้อนหิน"),
            ("💰 ผลกระทบ",          "ถูกหักราคาจากสหกรณ์ค่อนข้างมาก"),
        ]
    },
}

# ─── Load Model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vgg16_best.keras", compile=False)

model = load_model()

# ─── Helpers ──────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def load_csv() -> pd.DataFrame:
    cols = [
        "timestamp", "session_id", "filename",
        "predicted_class", "confidence",
        "prob_class3", "prob_class4", "prob_class5", "prob_classB"
    ]

    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            for col in cols:
                if col not in df.columns:
                    df[col] = np.nan
            return df[cols]
        except Exception:
            return pd.DataFrame(columns=cols)

    return pd.DataFrame(columns=cols)

def append_csv(rows: list[dict]):
    df_old = load_csv()
    df_new = pd.DataFrame(rows)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(CSV_FILE, index=False)

def advice(label):
    mapping = {
        "3": ("success", "ยางคุณภาพดี เหมาะสำหรับการจำหน่ายในราคาสูงสุด ควรรักษาคุณภาพในกระบวนการผลิตต่อไป"),
        "4": ("info",    "ยางคุณภาพมาตรฐาน ควรปรับปรุงกระบวนการรีดและตากยางเพื่อยกระดับเป็น Class 3"),
        "5": ("warning", "ยางคุณภาพต่ำ ควรตรวจสอบกระบวนการตากยางและความสะอาดในการผลิต"),
        "B": ("error",   "ยางไม่ผ่านมาตรฐาน ควรตรวจสอบความชื้น เชื้อรา และสิ่งปลอมปนทันที"),
    }
    return mapping.get(label, ("info", ""))

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 เมนูหลัก")
    st.divider()
    menu = st.radio(
        label="เลือกเมนู",
        options=[
            "🏢 รายละเอียดเกี่ยวกับหน่วยงาน",
            "📖 คำชี้แจงการใช้งาน",
            "🔍 ทำนายคุณภาพชั้นยางพารา",
            "📊 รายงานสถิติ",
        ],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("© สหกรณ์กองทุนสวนยางแคนดง จำกัด")
    st.caption("พัฒนาโดย: โครงการวิจัย AI ยางพารา")

# ══════════════════════════════════════════════════════════
# PAGE: รายละเอียดเกี่ยวกับหน่วยงาน
# ══════════════════════════════════════════════════════════
if menu == "🏢 รายละเอียดเกี่ยวกับหน่วยงาน":
    st.title("🏢 รายละเอียดเกี่ยวกับหน่วยงาน")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### สหกรณ์กองทุนสวนยางแคนดง จำกัด")
        st.markdown("""
        สหกรณ์กองทุนสวนยางแคนดง จำกัด เป็นองค์กรที่ดำเนินงานด้านการรวบรวม
        และจำหน่ายผลผลิตยางพาราของเกษตรกรในพื้นที่ โดยมีบทบาทสำคัญในการ
        กำหนดมาตรฐานคุณภาพยางแผ่นดิบ (Air Dried Sheets: ADS) เพื่อให้เกษตรกร
        ได้รับราคาที่เป็นธรรมตามคุณภาพที่แท้จริงของผลผลิต
        """)
    with col2:
        st.metric("📦 มาตรฐานที่ใช้", "Green Book")
        st.metric("🌿 ประเภทยาง", "ADS")

    st.divider()
    st.markdown("### 🎯 พันธกิจ")
    c1, c2 = st.columns(2)
    with c1:
        st.info("🤝 รวบรวมและจำหน่ายผลผลิตยางพาราของสมาชิกในราคาที่เป็นธรรม")
        st.info("📊 กำหนดมาตรฐานคุณภาพยางแผ่นดิบตามเกณฑ์ Green Book")
    with c2:
        st.info("🔬 ส่งเสริมการใช้เทคโนโลยี AI ในการตรวจสอบคุณภาพยาง")
        st.info("👨‍🌾 พัฒนาความรู้และทักษะให้แก่เกษตรกรสมาชิก")

    st.divider()
    st.markdown("### 📋 มาตรฐานการจัดชั้นคุณภาพยางแผ่นดิบ (ADS)")
    for cls, info in CLASS_DETAILS.items():
        with st.expander(f"{info['emoji']} {cls} — {info['title']}"):
            for lbl, desc in info["details"]:
                st.markdown(f"- **{lbl}:** {desc}")

# ══════════════════════════════════════════════════════════
# PAGE: คำชี้แจงการใช้งาน
# ══════════════════════════════════════════════════════════
elif menu == "📖 คำชี้แจงการใช้งาน":
    st.title("📖 คำชี้แจงการใช้งาน")
    st.divider()

    st.markdown("### 🤖 เกี่ยวกับระบบ AI")
    st.markdown("""
    ระบบนี้ใช้โมเดล **VGG16** ที่ผ่านการฝึกสอน (Fine-tuning) ด้วยภาพยางพาราแผ่นดิบ
    จากสหกรณ์กองทุนสวนยางแคนดง จำกัด เพื่อจำแนกคุณภาพยางออกเป็น 4 คลาส
    ได้แก่ Class 3, Class 4, Class 5 และ Class B
    """)
    st.divider()

    st.markdown("### 📸 วิธีการใช้งาน")
    for step in [
        ("1️⃣", "เลือกเมนู **ทำนายคุณภาพชั้นยางพารา** จากแถบด้านซ้าย"),
        ("2️⃣", "อัปโหลดภาพยางพารา **1 ภาพหรือมากกว่า** (JPG / PNG)"),
        ("3️⃣", "กดปุ่ม **🚀 เริ่มทำนาย** เพื่อประมวลผล"),
        ("4️⃣", "ดูผลรายภาพ ค่าเฉลี่ย และบันทึกลง CSV อัตโนมัติ"),
        ("5️⃣", "ดูสถิติย้อนหลังได้ที่เมนู **📊 รายงานสถิติ**"),
    ]:
        st.markdown(f"{step[0]} {step[1]}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1: st.success("✅ .JPG / .JPEG")
    with c2: st.success("✅ .PNG")
    with c3: st.warning("⚠️ ขนาดแนะนำ ≥ 224×224 px")

    st.divider()
    st.warning("""
    **⚠️ ข้อควรระวัง**
    - ผลการทำนายจาก AI เป็นเพียงการช่วยสนับสนุนการตัดสินใจเท่านั้น
    - ควรใช้ร่วมกับการตรวจสอบโดยผู้เชี่ยวชาญเสมอ
    - ภาพที่ถ่ายในสภาพแสงไม่เพียงพออาจส่งผลต่อความแม่นยำ
    """)

# ══════════════════════════════════════════════════════════
# PAGE: ทำนายคุณภาพชั้นยางพารา
# ══════════════════════════════════════════════════════════
elif menu == "🔍 ทำนายคุณภาพชั้นยางพารา":
    st.title("🔍 ทำนายคุณภาพชั้นยางพารา")
    st.caption("สหกรณ์กองทุนสวนยางแคนดง จำกัด | มาตรฐาน Air Dried Sheets (ADS)")
    st.divider()

    uploaded_files = st.file_uploader(
        "📤 อัปโหลดภาพยางพาราแผ่นดิบ (เลือกได้หลายภาพ)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📂 อัปโหลดแล้ว **{len(uploaded_files)} ภาพ** — กดปุ่มด้านล่างเพื่อเริ่มทำนาย")

        if st.button("🚀 เริ่มทำนาย", type="primary", use_container_width=True):
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            rows_to_save = []
            results = []

            st.divider()
            st.markdown("### 📋 ผลการทำนายรายภาพ")

            for i, f in enumerate(uploaded_files):
                image = Image.open(f)
                arr   = preprocess_image(image)

                with st.spinner(f"กำลังวิเคราะห์ภาพที่ {i+1}/{len(uploaded_files)} ..."):
                    preds = model.predict(arr, verbose=0)

                pred_idx   = int(np.argmax(preds[0]))
                pred_label = CLASS_NAMES[pred_idx]
                conf       = float(np.max(preds[0])) * 100
                info       = CLASS_DETAILS[pred_label]

                results.append({
                    "filename":        f.name,
                    "predicted_class": pred_label,
                    "confidence":      conf,
                    "probs":           preds[0],
                })

                # ── แสดงผลรายภาพ ──────────────────────────
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        st.image(image, caption=f.name, use_column_width=True)
                    with c2:
                        st.markdown(f"#### {info['emoji']} {pred_label}")
                        st.markdown(f"**{info['title']}**")
                        st.metric("🎯 ความมั่นใจ", f"{conf:.2f}%")
                    with c3:
                        st.markdown("**📈 ความน่าจะเป็นแต่ละคลาส**")
                        for cls, p in zip(CLASS_NAMES, preds[0]):
                            st.progress(float(p), text=f"{cls}: {p*100:.1f}%")

                        adv_type, adv_msg = advice(pred_label)
                        getattr(st, adv_type)(adv_msg)

                # ── เตรียมบันทึก CSV ──────────────────────
                rows_to_save.append({
                    "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id":     session_id,
                    "filename":       f.name,
                    "predicted_class":pred_label,
                    "confidence":     round(conf, 4),
                    "prob_class3":    round(float(preds[0][0]), 4),
                    "prob_class4":    round(float(preds[0][1]), 4),
                    "prob_class5":    round(float(preds[0][2]), 4),
                    "prob_classB":    round(float(preds[0][3]), 4),
                })

            # ── สรุปค่าเฉลี่ย ─────────────────────────────
            st.divider()
            st.markdown("### 📊 สรุปผลการทำนายทั้งหมด")

            df_result = pd.DataFrame(results)
            avg_conf  = df_result["confidence"].mean()
            class_counts = df_result["predicted_class"].value_counts()

            m_cols = st.columns(len(CLASS_NAMES) + 1)
            m_cols[0].metric("🖼️ ภาพทั้งหมด", len(results))
            for j, cls in enumerate(CLASS_NAMES):
                m_cols[j+1].metric(
                    f"{CLASS_DETAILS[cls]['emoji']} {cls}",
                    class_counts.get(cls, 0)
                )

            st.metric("🎯 ความมั่นใจเฉลี่ย", f"{avg_conf:.2f}%")

            # ── ตารางสรุป ─────────────────────────────────
            df_show = df_result[["filename","predicted_class","confidence"]].copy()
            df_show.columns = ["ชื่อไฟล์", "คลาสที่ทำนาย", "ความมั่นใจ (%)"]
            df_show["ความมั่นใจ (%)"] = df_show["ความมั่นใจ (%)"].round(2)
            st.dataframe(df_show, use_container_width=True)

            # ── บันทึก CSV ────────────────────────────────
            append_csv(rows_to_save)
            st.success(f"✅ บันทึกผลลงไฟล์ **{CSV_FILE}** เรียบร้อยแล้ว ({len(rows_to_save)} แถว)")

    else:
        st.info("👆 กรุณาอัปโหลดภาพยางพาราแผ่นดิบเพื่อเริ่มการวิเคราะห์")
        st.markdown("""
        ##### 💡 เคล็ดลับการถ่ายภาพ
        - 📷 ถ่ายภาพในที่แสงสว่างเพียงพอ
        - 🎯 ถ่ายให้เห็นเนื้อยางชัดเจน ไม่มีวัตถุบดบัง
        - 📐 ระยะห่างจากแผ่นยางประมาณ 30–50 ซม.
        - 🚫 หลีกเลี่ยงภาพเบลอหรือมีแสงสะท้อน
        """)

# ══════════════════════════════════════════════════════════
# PAGE: รายงานสถิติ
# ══════════════════════════════════════════════════════════
elif menu == "📊 รายงานสถิติ":
    st.title("📊 รายงานสถิติการทำนาย")
    st.divider()

    df = load_csv()

    if df.empty:
        st.warning("⚠️ ยังไม่มีข้อมูลการทำนาย กรุณาทำนายภาพก่อน")
    else:
        # df["timestamp"] = pd.to_datetime(df["timestamp"])
        # df["date"]      = df["timestamp"].dt.date
        # df["week"]      = df["timestamp"].dt.to_period("W").astype(str)
        # df["year"]      = df["timestamp"].dt.year
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="mixed",
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"]).copy()

        if df.empty:
            st.warning("⚠️ ไม่สามารถอ่านข้อมูลวันที่ในไฟล์ CSV ได้")
            st.stop()

        df["date"] = df["timestamp"].dt.date
        df["week"] = df["timestamp"].dt.to_period("W").astype(str)
        df["year"] = df["timestamp"].dt.year

        # ── Filter ────────────────────────────────────────
        period = st.radio(
            "📅 เลือกช่วงเวลา",
            ["รายวัน", "รายสัปดาห์", "รายปี"],
            horizontal=True
        )

        if period == "รายวัน":
            dates     = sorted(df["date"].unique(), reverse=True)
            sel_date  = st.selectbox("เลือกวันที่", dates)
            df_filter = df[df["date"] == sel_date]
            period_label = str(sel_date)

        elif period == "รายสัปดาห์":
            weeks     = sorted(df["week"].unique(), reverse=True)
            sel_week  = st.selectbox("เลือกสัปดาห์", weeks)
            df_filter = df[df["week"] == sel_week]
            period_label = sel_week

        else:  # รายปี
            years     = sorted(df["year"].unique(), reverse=True)
            sel_year  = st.selectbox("เลือกปี", years)
            df_filter = df[df["year"] == sel_year]
            period_label = str(sel_year)

        st.markdown(f"#### 📌 ช่วงเวลา: {period_label} | ทั้งหมด {len(df_filter)} รายการ")
        st.divider()

        if df_filter.empty:
            st.info("ไม่มีข้อมูลในช่วงเวลาที่เลือก")
        else:
            # ── Metrics ───────────────────────────────────
            class_counts = df_filter["predicted_class"].value_counts()
            avg_conf     = df_filter["confidence"].mean()

            m_cols = st.columns(len(CLASS_NAMES) + 2)
            m_cols[0].metric("🖼️ ทั้งหมด", len(df_filter))
            for j, cls in enumerate(CLASS_NAMES):
                m_cols[j+1].metric(
                    f"{CLASS_DETAILS[cls]['emoji']} {cls}",
                    class_counts.get(cls, 0)
                )
            m_cols[-1].metric("🎯 ความมั่นใจเฉลี่ย", f"{avg_conf:.2f}%")

            st.divider()

            # ── Charts ────────────────────────────────────
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### 🥧 สัดส่วนคลาส")
                pie_data = pd.DataFrame({
                    "คลาส":    [CLASS_DETAILS[c]["emoji"] + " " + c for c in CLASS_NAMES],
                    "จำนวน":   [class_counts.get(c, 0) for c in CLASS_NAMES],
                })
                pie_data = pie_data[pie_data["จำนวน"] > 0]
                st.bar_chart(pie_data.set_index("คลาส"))

            with c2:
                st.markdown("#### 📈 ความมั่นใจเฉลี่ยแต่ละคลาส")
                avg_by_class = (
                    df_filter.groupby("predicted_class")["confidence"]
                    .mean().reindex(CLASS_NAMES).fillna(0).round(2)
                )
                st.bar_chart(avg_by_class)

            st.divider()

            # ── Trend (รายวัน/สัปดาห์ ถ้าเลือกรายปี) ────
            if period == "รายปี":
                st.markdown("#### 📆 แนวโน้มรายเดือน")
                df_filter = df_filter.copy()
                df_filter["month"] = df_filter["timestamp"].dt.to_period("M").astype(str)
                trend = df_filter.groupby(["month","predicted_class"]).size().unstack(fill_value=0)
                st.line_chart(trend)

            elif period == "รายสัปดาห์":
                st.markdown("#### 📆 แนวโน้มรายวัน (ในสัปดาห์นี้)")
                trend = df_filter.groupby(["date","predicted_class"]).size().unstack(fill_value=0)
                st.line_chart(trend)

            # ── Raw Table ─────────────────────────────────
            st.divider()
            st.markdown("#### 🗂️ ข้อมูลดิบ")
            show_cols = ["timestamp","filename","predicted_class","confidence"]
            st.dataframe(
                df_filter[show_cols].sort_values("timestamp", ascending=False),
                use_container_width=True
            )

            # ── Download ──────────────────────────────────
            csv_bytes = df_filter.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ ดาวน์โหลด CSV ช่วงนี้",
                data=csv_bytes,
                file_name=f"rubber_report_{period_label}.csv",
                mime="text/csv",
                use_container_width=True
            )