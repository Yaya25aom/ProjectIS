import streamlit as st

# ต้องมาก่อน import อะไรทั้งนั้น
st.set_page_config(page_title="โมเดลตรวจจับอารมณ์", layout="wide")

from pages import page_1, page_2, page_3, page_4

st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Machine Learning", 
    "🧠 Neural Network", 
    "💻 Machine Learning Demo", 
    "🛠️ Neural Network"
])

with tab1:
    page_1.show()

with tab2:
    page_2.show()

with tab3:
    page_3.show()

with tab4:
    page_4.show()
