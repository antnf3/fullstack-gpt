import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
    """
)


# 다음 chain으로 answers와 question을 넘기기위해 dict타입으로 return 한다.
def get_answer(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {
    #             "question": question,
    #             "context": doc.page_content,
    #         }
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            
            Return the sources of the answers as they are, do not change them
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "answers": condensed,
            "question": question,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    target_div = soup.find("div", class_="contents_style")
    if target_div:
        return str(target_div.get_text()).replace("\n", " ").replace("\xa0", " ")
    else:
        return ""


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[r"^(?!.*(?:\/category|\/tag|\/guestbook))(?!.*\/[0-3]?\d{2}$).*$"],
        parsing_function=parse_page,
    )
    # 크롤러의 속도를 늦춤으로써 웹사이트의 차단정책(block)이나 속도 제한(rate limit)위반을 방지 할 수 있다.
    loader.requests_per_second = 2  # default=1, 요청 속도 설정
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()


st.set_page_config(page_title="Site-GPT", page_icon="💻")

st.markdown(
    """
# Site GPT

Ask question about the content of a website.

Start by writing the URL of the website on the sidebar.
"""
)

# https://waystation.tistory.com/sitemap.xml
with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriever = load_website(url)
        # docs = retriever.invoke("샤오미 아카라 미니 스위치가 무엇인가요?")
        # st.write(docs)

        # * Map Re-Rank
        # retriever로부터 받은 Documents를 보고 LLM에게 전달하여 물어본다. 그 Document만을 사용하여 사용자의 question에 답변해달라고한다.
        # 답변이 생성되면, LLM에게 답변의 유용도를 평가해달라고한다.(0점부터 5점까지 유용함을 평가해줘)
        # 그렇게 만들어진 모든 답변과 점수는 또 다른 Prompt에게 전달한다.
        # 그 Prompt는 다시 LLM에게 전달되어 요청한다.(주어진 답변들을 살펴보고, 가장 높은 점수를 가지고 있고, 가장 최근에 작성된 것을 선택해줘)

        # * 총 2개의 Chain을 만든다.
        # 1. 모든 개별 Document에 대한 답변 및 점수생성
        # 2. 모든 답변 및 점수가 생성된후에 실행되어 점수가 가장 높고 최근에 작성된 답변을 선택하여 반환한다.
        question = st.text_input("Ask a question to the website.")
        if question:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answer)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(question)
            st.write(result.content.replace("$", "\$"))
