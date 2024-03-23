from typing import Dict, List
from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk, LLMResult
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOllama
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📑",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

llm_memory = ChatOllama(model="mistral:latest", temperature=0.1, streaming=True)


@st.cache_resource
def init_memory(_llm):
    return ConversationSummaryBufferMemory(
        llm=_llm, max_token_limit=120, return_messages=True, memory_key="chat_history"
    )


memory = init_memory(llm_memory)


def load_memory(_):
    print(memory.load_memory_variables({})["chat_history"])
    return memory.load_memory_variables({})["chat_history"]


if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    file = st.file_uploader(
        label="Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OllamaEmbeddings(
            model="mistral:latest",
        )
        cached_embeddir = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddir)
        retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
                                       
Context:{context}
Question:{question}
"""
)

st.title("PrivateGPT")
st.markdown(
    """
Use this chatbot to ask questions to an AI about your files!

Upload Document file on SideBar
"""
)

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        if True:
            # chain 으로 후출
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                    "chat_history": load_memory,
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                result = chain.invoke(message)
                memory.save_context({"input": message}, {"output": result.content})
        else:
            # 수동으로 호출
            docs = retriever.invoke(message)
            docs = "\n\n".join(document.page_content for document in docs)
            prompt = prompt.format_messages(context=docs, question=message)
            result = llm.predict_messages(prompt)
            st.write(result.content)
else:
    st.session_state["messages"] = []
