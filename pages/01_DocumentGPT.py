import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="Document-GPT",
    page_icon="💾",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        with st.sidebar:
            st.write("llm started")

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        with st.sidebar:
            st.write("llm ended.")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

llm_memory = ChatOpenAI(temperature=0.1)


@st.cache_resource
def init_memory(_llm):
    return ConversationSummaryBufferMemory(
        llm=_llm,
        max_token_limit=160,
        memory_key="chat_history",
        return_messages=True,
    )


memory = init_memory(llm_memory)


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


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


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


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
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
st.title("Document-GPT")

st.markdown(
    """
Welcome!


Use this chatbot to ask questions to an AI about your files

Upload your file on the sidebar.
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
                "chat_history": load_memory,
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)
            memory.save_context(
                {"input": message},
                {"output": response.content},
            )

else:
    st.session_state["messages"] = []


print(memory.load_memory_variables({})["chat_history"])
