import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Site-GPT", page_icon="💻")

st.markdown(
    """
# Site GPT

Ask question about the content of a website.

Start by writing the URL of the website on the sidebar.
"""
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


@st.cache_data(show_spinner="Loading website...")
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
    return docs


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
        docs = load_website(url)
        st.write(docs)
