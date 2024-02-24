import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config(page_title="SiteGPT", page_icon="a")

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
Ask questions about the content of a website.

Start by writing the URL of the website on the sidebars.  
"""
)


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()
    if footer:
        footer.decompose()

    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/blog\/).*",  # 포함
            # r"^(?!.*\/blog\/).*"  # 제외 ?! => 제외
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    # AsyncChromiumLoader
    # loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # transformed = html2text_transformer.transform_documents(docs)
    # st.write(transformed)
    if ".xml" not in url:
        with st.sidebar:
            st.error("❌ Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        st.write(docs)
