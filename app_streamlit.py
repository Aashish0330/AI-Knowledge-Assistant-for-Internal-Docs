import os, streamlit as st
from rag.hybrid_retriever import HybridRetriever
from rag.re_rank import CrossEncoderReranker
from rag.chain import make_chain

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("📚 AI Knowledge Assistant — Internal Docs")

if not (os.path.exists("storage/bm25_corpus.pkl") and os.path.isdir("storage/faiss_index")):
    st.warning("No index found. Add docs to ./data and run: python scripts/build_index.py")
    st.stop()
