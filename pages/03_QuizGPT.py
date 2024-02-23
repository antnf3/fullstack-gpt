import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader

st.set_page_config(page_title="QuizGPT", page_icon="❓")
st.title("QuizGPT")

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")


def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)

    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            retriever = WikipediaRetriever(top_k_results=1, lang="ko")
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)
                st.write(docs)

if not docs:
    st.write(
        """
Welcome to QuizGPT

I will make a quiz from Wikipedia articles or files you upload to test your knowledg and help you study.

Get started by uploading a file or searching on wikipedia in the sidebar
"""
    )
else:
    st.write(docs)
