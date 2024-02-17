import streamlit as st
import time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

st.set_page_config(page_title="DocumentGPT", page_icon="📑")

st.title("DocumentGPT")

st.markdown(
    """
Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader(
    label="Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
)


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/{file.name}"
    st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/embedings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings()
        cached_embeddir = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddir)
        retriever = vectorstore.as_retriever()
    return retriever


if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    st.write(s)
