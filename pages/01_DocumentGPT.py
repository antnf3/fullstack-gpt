import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

st.set_page_config(
    page_title="Document-GPT",
    page_icon="💾",
)

llm = ChatOpenAI(temperature=0.1)


@st.cache_resource(show_spinner="Embeddings...")
def embedd_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/document_gpt/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        # cache 디렉토리 설정
        cache_dir = LocalFileStore(f"./.cache/embeddings/document_gpt/{file.name}")
        # splitter 선언
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        # load file & split
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)

        # Cache & embedding
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            cache_dir,
        )

        # Vector Store
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        # Retriever
        retriever = vectorstore.as_retriever()
        return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_message():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
    
    Context: {context}
    """,
        ),
        ("human", "{question}"),
    ]
)
st.title("Document-GPT")

st.markdown(
    """
Welcome!


Use this chatbot to ask questions to an AI about your files

Upload your file.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload your file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embedd_file(file)
    send_message("I'm ready! Ask away", "ai", False)
    paint_message()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []
