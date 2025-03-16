import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from PIL import Image
import base64
from io import BytesIO


image = Image.open('Nue.png')
buffer = BytesIO()
image.save(buffer, format="PNG")
img_str = base64.b64encode(buffer.getvalue()).decode()

def show():
    st.title("Neural Network")
    st.write("## แนวทางการพัฒนาโมเดล")
    st.write("### About Datasets 📁🔎")
    st.write(" Dataset Neural Network 📄")
    st.write(" ชุกข้อมูลจำแนกอารมณ์จากใบหน้า FER-2013 จาก Website https://www.kaggle.com/datasets/msambare/fer2013")
    st.write("### Features of dataset :")
    st.write("- Image: รูปภาพใบหน้าตามอารมณ์")
    st.write("- Label: ข้อความกำหนดประเภทอารมณ์")
    st.write("### การเตรียมข้อมูล 💻")
    st.markdown(
    """ 
    - การเตรียมข้อมูล
        - ทำการอ่านไฟล์ csv แล้วจำแนกรูปภาพตาม Folder อารมณ์ 
        - ทำการใส่ label โดยการวนลูปรูปภาพว่าแตละรูปภาพอยู่ในหมวดไหนแล้วทำการบันทึกชื่อไฟล์กับประเภทอารมณ์ 
        - แปลงsize รูปภาพให้เป็น scale 48x48 และทำการเปลี่ยนสีรูปให้เป็นสีเทา
        - แปลงเป็น NumPy Array และทำการ Reshape เป็น 4D Tensor (จำนวนภาพ, 48, 48, 1) → เพิ่มมิติช่องสี (Grayscale) เพื่อให้โมเดลอ่านข้อมูลได้ถูกต้อง
        - Normalize ค่า Pixel (หาร 255.0) → ปรับค่าพิกเซลให้อยู่ในช่วง [0,1] ช่วยให้โมเดลเรียนรู้ได้เร็วขึ้นและลดปัญหา gradient vanishing
        - ทำการแปลงประเภทอารมณ์เป็นตัวเลขและทำการ One hot Encoding
    """,
    unsafe_allow_html=True
)
    st.write("### MODEL ที่ใช้ในพัฒนา Neural Network 🛠️ ")
    st.write("#### Model Convolutional Neural Network (CNN):")
    st.markdown(
    """ 
    - Convolutional Neural Network (CNN) เป็นประเภทหนึ่งของ Neural Network ที่ออกแบบมาเพื่อการประมวลผลข้อมูลที่มีลักษณะเป็น กริด เช่น ภาพ (Images) โดยมีโครงสร้างและขั้นตอนการทำงานที่เหมาะสมกับงานประมวลผลภาพโดยเฉพาะ 🖼️
    - ขั้นตอนการพัฒนา
        - แบ่งข้อมูล 80% Train, 20% Test
        - โครงสร้างประสาทเทียม(Convolutional Neural Network) โดยโครงสร้างนี้ประกอบไปด้วย หลายเลเยอร์ ที่ทำงานร่วมกัน:
        - ใช้ Conv2D(32, (3,3), activation='relu')ใช้ฟิลเตอร์ 32 ตัว ขนาด 3x3 เพื่อตรวจจับลักษณะของภาพ
        - ใช้ MaxPooling2D(2,2)ลดขนาดภาพเพื่อให้โมเดลทำงานเร็วขึ้น
        - Flatten ใช้แปลงข้อมูล 2D เป็น 1D ก่อนเข้าสู่ fully connected layers
        - Dense (Fully connected layers) ใช้ในการประมวลผลข้อมูลเพื่อจำแนกอารมณ์
        - Dropout ใช้ลดการ Overfitting โดยการปิดการทำงานของบางนิวรอนในระหว่างการฝึก
        - กำหนดให้มี 20 epochs และใช้ batch size = 64 ใช้ชุดข้อมูลฝึกสอน (X_train, y_train) และตรวจสอบผลลัพธ์ด้วยชุดข้อมูลทดสอบ (X_test, y_test)
        - ทำการฝึกและแสดงผลทำนายการประเมินผลเป็น แสดงกราฟ Accuracy และ Loss
    """,
    unsafe_allow_html=True
)
    st.write(" กราฟแสดงผล Accuracy และ Loss")
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_str}" style="width: 60%; border-radius: 10px;">
    </div>
    """,
    unsafe_allow_html=True
)
    st.write(" 📊 Accuracy: 51.80%")

    
