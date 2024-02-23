from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

st.set_page_config(page_title="SiteGPT", page_icon="a")

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
Ask questions about the content of a website.

Start by writing the URL of the website on the sidebars.  
"""
)


@st.cache_data(show_spinner="loading website...")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 1
    docs = loader.load()
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
