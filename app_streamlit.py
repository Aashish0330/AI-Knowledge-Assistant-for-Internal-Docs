# app_streamlit.py
from __future__ import annotations
import os, re, pickle, subprocess
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from rag.universal_qa import universal_answer, QAResult
from rag.llm import generate_llm_answer, LLMResult

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("📚 AI Knowledge Assistant (Local RAG)")

IDX_CHUNKS = Path("storage/nn_chunks.pkl")

records = None
tfidf: TfidfVectorizer | None = None
svd = None
nn: NearestNeighbors | None = None
use_svd = False


def ollama_chat(prompt, context=None, model="llama3.1"):
    """Query Ollama locally with retrieved context"""
    url = "http://127.0.0.1:11434/api/generate"
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer clearly and factually based on context."
    data = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(url, json=data, timeout=60)
        r.raise_for_status()
        result = r.json()
        return result.get("response", "").strip()
    except Exception as e:
        print("Ollama error:", e)
        return None


def _load_index():
    global records, tfidf, svd, nn, use_svd
    if IDX_CHUNKS.exists():
        payload = pickle.load(open(IDX_CHUNKS, "rb"))
        records = payload["records"]
        tfidf = payload["tfidf"]
        use_svd = payload.get("use_svd", False)
        svd = payload.get("svd")
        nn = payload["nn"]
    else:
        records, tfidf, svd, nn, use_svd = None, None, None, None, False

def _num_items() -> int:
    return len(records) if records else 0

_load_index()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Diagnostics")
    st.write("Index:", "chunks" if records else "—")
    st.write("Use SVD:", bool(use_svd))
    st.write("Items:", _num_items())

    st.divider()
    st.subheader("Generation")
    use_llm = st.toggle("Use LLM (grounded)", value=True)
    st.caption("If disabled, uses extractive fallback.")

    st.divider()
    st.subheader("Manage documents")
    uploaded = st.file_uploader(
        "📄 Add files (TXT/MD/PDF) to ./data", type=["txt","md","markdown","pdf"], accept_multiple_files=True
    )
    if uploaded:
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (data_dir / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to ./data")

    if st.button("🔁 Rebuild (chunked) index"):
        try:
            out = subprocess.check_output(
                [os.environ.get("PYTHON", "python"), "scripts/build_index_sklearn_chunks.py"],
                stderr=subprocess.STDOUT, text=True
            )
            st.code(out)
            _load_index()
            st.success("Rebuilt and reloaded index.")
        except subprocess.CalledProcessError as e:
            st.error("Failed to rebuild index.")
            st.code(e.output or str(e))

# ------------- retrieval -------------
def encode(texts: List[str]) -> np.ndarray:
    V = tfidf.transform(texts)
    Z = (svd.transform(V).astype("float32") if (use_svd and svd is not None) else V.toarray().astype("float32"))
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return Z

def retrieve(query: str, k: int = 6) -> list[dict]:
    if not records or nn is None:
        return []
    Zq = encode([query])
    n = max(1, min(k, len(records)))
    D, I = nn.kneighbors(Zq, n_neighbors=n, return_distance=True)
    hits = []
    for d, i in zip(D[0], I[0]):
        sim = float(1.0 - d)
        r = records[i]
        hits.append({"score": sim, "source": r["source"], "text": r["text"], "chunk_id": r["chunk_id"]})
    return hits

# ------------- chat history -------------
if "history" not in st.session_state:
    st.session_state["history"] = []

for role, msg in st.session_state["history"]:
    with st.chat_message(role):
        st.markdown(msg)

# ------------- empty guard -------------
if _num_items() == 0:
    st.warning(
        "Index is empty.\n\n"
        "1) Put TXT/MD/PDF files in `./data`\n"
        "2) Click **Rebuild (chunked) index** in the sidebar\n"
        "3) Ask questions below."
    )

# ------------- chat input -------------
prompt = st.chat_input("Ask about policies…")

if prompt:
    st.session_state["history"].append(("user", prompt))
    top_hits = retrieve(prompt, k=8)

    if use_llm and top_hits:
        try:
            llm_res: LLMResult = generate_llm_answer(prompt, top_hits[:6])
            md = f"**Answer:** {llm_res.answer}\n\n"
            if llm_res.citations:
                # show just filenames if long
                short = [Path(c).name for c in llm_res.citations]
                md += f"**Sources:** {', '.join(f'`{s}`' for s in short)}"
            st.session_state["history"].append(("assistant", md))
        except Exception as e:
            # graceful fallback to extractive
            qa: QAResult = universal_answer(prompt, top_hits)
            md = f"**Answer (fallback):** {qa.answer}\n\n"
            if qa.sources:
                md += f"**Source:** `{qa.sources[0]}`"
            st.session_state["history"].append(("assistant", md))
    else:
        qa: QAResult = universal_answer(prompt, top_hits)
        md = f"**Answer:** {qa.answer}\n\n"
        if qa.sources:
            md += f"**Source:** `{qa.sources[0]}`"
        st.session_state["history"].append(("assistant", md))

    st.rerun()