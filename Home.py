import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")
st.title("Streamlit")
st.subheader(today)

p = PromptTemplate.from_template("XXXX")

# data flow : 데이터가 변경되면 위에서 부터 아래로 모든 스크립트가 재실행됨

model = st.selectbox(label="select box", options=("GPT3.5", "GPT4.0"))

if model == "GPT3.5":
    st.write("cheap")
else:
    name = st.text_input(label="What is your Name?")
    st.write(name)

    slider = st.slider("test", min_value=0.1, max_value=1.0)
    st.write(slider)
