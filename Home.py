import streamlit as st
from langchain.prompts import PromptTemplate

st.title("Streamlit")
st.subheader("Welcome Streamlit")

p = PromptTemplate.from_template("XXXX")


st.selectbox(label="select box", options=("GPT3.5", "GPT4.0"))
