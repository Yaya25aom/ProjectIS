import streamlit as st
import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tempfile
import pyttsx3  # ใช้ pyttsx3 แทน pygame
from gtts import gTTS
import time
import os

# คลาส Custom Layer
class CastLayer(Layer):
    def call(self, inputs):
        return inputs

# โหลดโมเดล
@st.cache_resource
def load_emotion_model():
    model_path = "model_emotion.h5"
    return load_model(model_path, custom_objects={"Cast": CastLayer}, compile=False)

# ฟังก์ชันทำนายอารมณ์
def predict_emotion(face, model):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(face)  
    face = cv2.resize(face, (48, 48)) / 255.0
    face = np.expand_dims(face, axis=-1)  
    face = np.expand_dims(face, axis=0)  

    predictions = model.predict(face)
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return labels[np.argmax(predictions)]

# ตั้งค่า pyttsx3 สำหรับการพูด
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # ปรับความเร็วของเสียง
engine.setProperty('volume', 1)  # ปรับระดับเสียง (0.0 ถึง 1.0)

is_speaking = False  

# ฟังก์ชันพูด
def speak(text):
    global is_speaking

    if is_speaking:
        return
    
    is_speaking = True  

    def play_audio():
        global is_speaking
        engine.say(text)
        engine.runAndWait()
        is_speaking = False

    threading.Thread(target=play_audio, daemon=True).start()


# โหลดโมเดล
model = load_emotion_model()

# ✅ **เพิ่มฟังก์ชัน show() ตรงนี้**
def show():
    st.title("🎭 Real-time Emotion Detection")
    run = st.checkbox("📸 เปิดกล้อง")

    emotion_speech = {
        "Angry": "คุณดูอารมณ์ไม่ดีนะ ใจเย็นๆ นะครับ",
        "Disgust": "ดูเหมือนว่าคุณจะรู้สึกขยะแขยง",
        "Fear": "คุณกำลังรู้สึกกลัวหรือกังวลอยู่หรือเปล่า",
        "Happy": "วันนี้คุณดูมีความสุข",
        "Sad": "คุณดูเศร้า ขอให้ทุกอย่างดีขึ้นนะ",
        "Surprise": "คุณดูตกใจ มีอะไรเกิดขึ้นหรอ",
        "Neutral": "คุณดูปกติ ไม่มีอะไรเป็นพิเศษ"
    }

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_button = st.button("🔴 ปิดกล้อง")

        last_emotion = None  
        face_detected = False  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                st.warning("📌 กล้องถูกปิดแล้ว")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                face_detected = True  
            else:
                face_detected = False  
                last_emotion = None  

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                emotion = predict_emotion(face, model)

                if emotion != last_emotion:
                    if not is_speaking:
                        speak(emotion_speech.get(emotion, ""))
                        last_emotion = emotion  

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            stframe.image(frame, channels="BGR")
        
        cap.release()
    else:
        st.warning("📌 กดเปิดกล้องเพื่อเริ่มต้น")
