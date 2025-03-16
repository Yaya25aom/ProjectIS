import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from PIL import Image
import base64
from io import BytesIO

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/cococo/ProjectIS/risk_Cervical_random_risk_balanced.csv")  # แทนที่ด้วยชื่อไฟล์ของคุณ
    return df

# โหลดข้อมูล
df = load_data()

image = Image.open('heatmap.png')
buffer = BytesIO()
image.save(buffer, format="PNG")
img_str = base64.b64encode(buffer.getvalue()).decode()

report_dict = {
    "precision": [0.98, 0.99],
    "recall": [0.99, 0.98],
    "f1-score": [0.98, 0.98],
    "support": [261, 254],
}
index_labels = ["Class 0", "Class 1"]
df_report = pd.DataFrame(report_dict, index=index_labels)

report_dict = {
    "precision": [0.98, 1.00],
    "recall": [1.00, 0.98],
    "f1-score": [0.99, 0.99],
    "support": [261, 254],
}
index_labels = ["Class 0", "Class 1"]
df_reportXG = pd.DataFrame(report_dict, index=index_labels)

report_dict = {
    "precision": [0.96, 0.95],
    "recall": [0.95, 0.96],
    "f1-score": [0.95, 0.95],
    "support": [261, 254],
}
index_labels = ["Class 0", "Class 1"]
df_reportKNN = pd.DataFrame(report_dict, index=index_labels)

def show():
    st.title("Machine Learning💻📈")
    st.write("## แนวทางการพัฒนาโมเดล")
    st.write("### About Datasets 📁🔎")
    st.write(" Dataset Machine Learning 📄")
    st.write(" ชุดข้อมูล Cervical Cancer Risk Classification จาก Website https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification")
    st.write(" Example features of Cervical Cancer Risk Classification in dataset")
    st.dataframe(df.iloc[:5, :])
    st.write("### Features of dataset :")
    st.write("- Age: อายุของผู้หญิง (ปี)")
    st.write("- Number of sexual partners: จำนวนคู่ทางเพศที่เคยมี")
    st.write("- First sexual intercourse: อายุเมื่อมีเพศสัมพันธ์ครั้งแรก")
    st.write("- Num of pregnancies: จำนวนการตั้งครรภ์")
    st.write("- Smokes: การสูบบุหรี่ (1 = สูบ, 0 = ไม่สูบ)")
    st.write("- Smokes (years): จำนวนปีที่สูบบุหรี่")
    st.write("- Hormonal Contraceptives (years): จำนวนปีที่ใช้การคุมกำเนิดแบบฮอร์โมน")
    st.write("- IUD (years): จำนวนปีที่ใช้ IUD (อุปกรณ์คุมกำเนิด)")
    st.write("- STDs: การมีโรคติดต่อทางเพศสัมพันธ์ (1 = มี, 0 = ไม่มี)")
    st.write("- STDs:condylomatosis: การติดเชื้อคอนดิโลมา (โรคติดต่อทางเพศ)")
    st.write("- STDs:cervical condylomatosis: การติดเชื้อคอนดิโลมาในปากมดลูก")
    st.write("- STDs:vulvo-perineal condylomatosis: การติดเชื้อคอนดิโลมาในอวัยวะเพศหญิง")
    st.write("- STDs:syphilis: การติดเชื้อซิฟิลิส")
    st.write("- STDs:pelvic inflammatory disease: การติดเชื้อโรคอักเสบในช่องท้อง")
    st.write("- STDs:genital herpes: การติดเชื้อโรคเริมทางเพศ")
    st.write("- STDs:AIDS: การติดเชื้อ HIV หรือโรคเอดส์")
    st.write("- STDs:HIV: การติดเชื้อ HIV")
    st.write("- STDs:HPV: การติดเชื้อ HPV (ไวรัสหูด)")
    st.write("- Dx:CIN: การตรวจหาภาวะเนื้องอกในปากมดลูก (CIN)")
    st.write("- Dx:HPV: การตรวจหาการติดเชื้อ HPV")
    st.write("- Dx: การวินิจฉัยโรค (อาจเป็นโรคที่เกี่ยวกับปากมดลูกหรือ HPV)")
    st.write("- Hinselmann: การตรวจวินิจฉัยโรคด้วยการตรวจปากมดลูก (Hinselmann test)")
    st.write("- Schiller: การตรวจวินิจฉัยโรคด้วยการทดสอบ Schiller")
    st.write("- Citology: การตรวจวิเคราะห์เซลล์ (การตรวจเซลล์จากปากมดลูก)")
    st.write("- iopsy: การตรวจชิ้นเนื้อ (การตัดชิ้นเนื้อเพื่อตรวจหามะเร็งหรือโรคอื่นๆ)")
    st.write("### การเตรียมข้อมูล 💻")
    st.write("- โหลดข้อมูล Dataset จาก Kaggle และตรวจสอบค่าที่ขาดหายไป") 
    st.write("##### วิธีการเติมข้อมูลที่หายไป (Missing Value Handling)")
    st.write("- ใช้หลักการเติมค่า NaN ที่เหมาะสม เช่น Rule-based Imputation (การเติมค่าตามกฎเงื่อนไข) ถ้าอายุต่ำกว่า 15 ปี ➝ กำหนดให้เป็น 0 ถ้าอายุ 15 ปีขึ้นไป ➝ สุ่มค่าระหว่าง 1 ถึง min(10, อายุ - 14), Random Sampling (การสุ่มค่าจากข้อมูลจริง) First sexual intercourse → สุ่มจากค่าที่มีอยู่แล้วในข้อมูล, Smokes (years) → สุ่มค่าในช่วง 0.5 ถึง (อายุ - 14), Conditional Imputation (การเติมค่าตามความสัมพันธ์ของตัวแปร) ถ้า Smokes (years) == 0 → Smokes = 0 ถ้า Smokes (years) มีค่า → Smokes = 1, Mean Imputation: เติมค่า NaN ด้วยค่าเฉลี่ยของแต่ละกลุ่มอายุที่ได้")
    st.write("##### วิธีจัดการค่าที่ผิดปกติ")
    st.write("- ใช้วิธี Outlier แก้ไขข้อมูลที่ไม่สมเหตุสมผลของคอลัมน์ First sexual intercourse ที่มีค่ามากกว่าอายุ โดยทำการใช้การสุ่มค่าใหม่ในช่วงที่เหมาะสม (ระหว่าง 15 ถึงอายุจริง - 1), .ใช้วิธีการ Skewness เพื่อดูว่าข้อมูลสมมาตรไหมและใช้วิธีวิธีการในการจัดการค่าที่ไม่สมมาตร Yeo-Johnson ใช้ในข้อมูลมีค่าติดลบหรือศูนย์, Box-Cox Transformation ใช้ในข้อมูลที่มีค่ามากกว่า 0, Log Transformation ใช้ในการแก้ปัญหาข้อมูลที่มีการกระจายแบบเบ้ขวา")
    st.write("- ใช้เทคนิค Standardization (Z-score Normalization) โดยใช้ StandardScaler เพื่อทำให้ข้อมูลมีค่าเฉลี่ย (mean) เป็น 0 และส่วนเบี่ยงเบนมาตรฐาน (std) เป็น 1")
    st.write("### MODEL ที่ใช้ในพัฒนา Machine Learning 🛠️ ")
    st.write("#### ModelRandomForestClassifier:")
    st.write(" RandomForest เป็นโมเดล Machine Learning ประเภท Supervised Learning แบบ Classification โดยอาศัยหลักการสร้างต้นไม้การตัดสินใจหลายต้น (Decision Trees) แล้วรวมผลลัพธ์เพื่อลดโอกาสเกิด Overfitting และเพิ่มความแม่นยำของโมเดล:")
   
    st.markdown(
    """ 
    - การเตรียมข้อมูล
        - ตรวจสอบค่า NULL และจัดการค่าที่หายไป 
        - วิเคราะห์ค่า Skew ของข้อมูล 
        - แปลงค่าข้อมูลที่จำเป็นให้อยู่ในรูปแบบที่เหมาะสม 
        - แบ่งข้อมูลเป็นชุด Train และ Test 
            - การฝึกและทดสอบโมเดล
            - ใช้ RandomForestClassifier โดยกำหนด n_estimators=100 และ random_state=42
            - ทำนายผลลัพธ์บนชุดข้อมูลทดสอบ
            - ประเมินผลลัพธ์ผ่าน Accuracy Score และ Confusion Matrix
    """,
    unsafe_allow_html=True
)

    st.write(" 📊Confusion Matrix")
    st.dataframe(df_report)
    
    st.write("##### 📌accuracy  0.98")
    st.write("#### ModelXGBClassifier:")
    st.write(" โมเดล XGBoost (Extreme Gradient Boosting) เป็นโมเดล Machine Learning ประเภท Supervised Learning ในการทำ Classification ที่ใช้เทคนิค Gradient Boosting เพื่อสร้างโมเดลที่มีประสิทธิภาพสูง โดยใช้หลาย ๆ Decision Trees ซึ่งแต่ละต้นจะพยายามแก้ไขข้อผิดพลาดของต้นที่แล้ว และเมื่อรวมกันจะให้ผลลัพธ์ที่มีความแม่นยำสูงขึ้น")
    st.markdown(
    """ 
    - การเตรียมข้อมูล
        - ตรวจสอบค่า NULL และจัดการค่าที่หายไป 
        - วิเคราะห์ค่า Skew ของข้อมูล 
        - แปลงค่าข้อมูลที่จำเป็นให้อยู่ในรูปแบบที่เหมาะสม 
        - แบ่งข้อมูลเป็นชุด Train และ Test 
            - การฝึกและทดสอบโมเดล
            - ใช้ XGBoost Classifier โดยกำหนด use_label_encoder=False และ eval_metric='logloss'
            - ทำนายผลลัพธ์บนชุดข้อมูลทดสอบ
            - ประเมินผลลัพธ์ผ่าน Accuracy Score และ Confusion Matrix
    """,
    unsafe_allow_html=True
)
    st.write(" 📊Confusion Matrix")
    st.dataframe(df_reportXG)
    st.write("##### 📌accuracy 0.99")
    st.write("#### KNeighborsClassifier:")
    st.write(" K-Nearest Neighbors (KNN) เป็นโมเดล Machine Learning ประเภท Supervised Learning ที่ใช้สำหรับการจำแนกประเภท (classification) และการทำนายค่า (regression) โดยอิงจากความใกล้ชิดของข้อมูลในลักษณะของเพื่อนบ้าน (neighbors) ที่ใกล้เคียงกันในพื้นที่เชิงปริมาณ (feature space)")
    st.markdown(
    """ 
    - การเตรียมข้อมูล
        - ตรวจสอบค่า NULL และจัดการค่าที่หายไป 
        - วิเคราะห์ค่า Skew ของข้อมูล 
        - แปลงค่าข้อมูลที่จำเป็นให้อยู่ในรูปแบบที่เหมาะสม 
        - แบ่งข้อมูลเป็นชุด Train และ Test 
            - การฝึกและทดสอบโมเดล
            - ใซึ่งกำหนดให้ค่าเริ่มต้นของพารามิเตอร์ k เท่ากับ 5 โมเดลจะพิจารณา 5 จุดข้อมูลที่ใกล้ที่สุดในชุดข้อมูลฝึกเพื่อทำการทำนายโดยใช้ระยะห่างระหว่างข้อมูลเพื่อหากลุ่มเพื่อนบ้านที่ใกล้เคียงที่สุด ซึ่งจะนำมาใช้ในการตัดสินใจแยกประเภทข้อมูลอย่างเหมาะสมที่สุด
            - ทำนายผลลัพธ์บนชุดข้อมูลทดสอบ
            - ประเมินผลลัพธ์ผ่าน Accuracy Score และ Confusion Matrix
    """,
    unsafe_allow_html=True
)
    st.write(" 📊Confusion Matrix")
    st.dataframe(df_reportKNN)
    st.write("##### 📌accuracy 0.95")
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_str}" style="width: 60%; border-radius: 10px;">
        <p style="font-size:18px; color:gray;">Data Distribution by Age and Cancer Diagnosis, Grouped by Age Group</p>
    </div>
    """,
    unsafe_allow_html=True
)
