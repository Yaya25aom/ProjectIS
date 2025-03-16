import streamlit as st
import pandas as pd
import pickle  # สำหรับการโหลดโมเดล

# ฟังก์ชันโหลดโมเดล
@st.cache_data
def load_model():
    with open("xgboost_model.pkl", "rb") as file:  # โหลดโมเดลจากไฟล์
        model = pickle.load(file)
    return model

def show():
    st.title("การทำงานของโมเดล")
    st.write("### ป้อนข้อมูลเพื่อลองทำนายผล")

    # กำหนดฟีเจอร์ที่ต้องรับ input
    feature_names = [
        'Age', 'Number of sexual partners', 'First sexual intercourse',
        'Num of pregnancies', 'Smokes', 'Smokes (years)',
        'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs',
        'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vulvo-perineal condylomatosis',
        'STDs:syphilis', 'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:AIDS',
        'STDs:HIV', 'STDs:HPV', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy'
    ]

    # รับค่าจากผู้ใช้
    user_inputs = {}
    for feature in feature_names:
        if "years" in feature or "Age" in feature or "age" in feature:
            user_inputs[feature] = st.number_input(feature, min_value=0, max_value=100, value=0)
        elif "STDs" in feature or "Dx" in feature or feature in ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']:
            user_inputs[feature] = st.selectbox(feature, options=[0, 1], index=0)
        else:
            user_inputs[feature] = st.number_input(feature, min_value=0, max_value=100, value=0)

    # สร้าง DataFrame จาก user input
    input_df = pd.DataFrame([user_inputs])
    st.write("### 🔎 ข้อมูลที่นำเข้าทำนาย")
    st.dataframe(input_df)

    if st.button("ทำนายผล"):
        try:
            # โหลดโมเดล
            model = load_model()

            # ทำนายผลจากโมเดลที่โหลดมา
            prediction = model.predict(input_df)[0]
            result = "พบความเสี่ยงมะเร็งปากมดลูก" if prediction == 1 else "ไม่มีความเสี่ยง"
            st.success(f"🔍 ผลการทำนาย: {result}")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {e}")
