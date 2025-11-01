import os, pickle, numpy as np, streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="AI Knowledge Assistant — sklearn index", layout="wide")
st.title("📚 AI Knowledge Assistant (TF-IDF / SVD + NearestNeighbors)")

if not os.path.exists("storage/nn_index.pkl"):
    st.warning("No index found. Run: `python scripts/build_index_sklearn.py`"); st.stop()

with open("storage/nn_index.pkl","rb") as f:
    payload=pickle.load(f)

docs = payload["docs"]
tfidf: TfidfVectorizer = payload["tfidf"]
use_svd: bool = payload.get("use_svd", False)
svd: TruncatedSVD | None = payload.get("svd")
nn: NearestNeighbors = payload["nn"]

with st.sidebar:
    st.header("Diagnostics")
    st.write("Docs:", len(docs))
    st.write("Use SVD:", use_svd)
    if use_svd and svd is not None: st.write("Dim:", svd.n_components)
    st.info("Tip: add more files into ./data for better results, then rebuild the index.")

def encode_query(q: str) -> np.ndarray:
    v = tfidf.transform([q])
    if use_svd and svd is not None:
        z = svd.transform(v).astype("float32")
    else:
        z = v.toarray().astype("float32")
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    return z

def search(q: str, k:int=5):
    z = encode_query(q)
    k = min(k, len(docs))
    distances, indices = nn.kneighbors(z, n_neighbors=k, return_distance=True)
    hits=[]
    for rank,(d, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        sim = float(1.0 - d)  # cosine similarity
        doc = docs[idx]
        snippet = doc["text"][:700].replace("\n"," ")
        hits.append({"rank":rank,"score":sim,"source":doc["source"],"snippet":snippet})
    return hits

if "history" not in st.session_state: st.session_state["history"]=[]

# render history
for role,msg in st.session_state["history"]:
    with st.chat_message(role): st.markdown(msg)

prompt = st.chat_input("Ask about policies, runbooks, architecture…")

if prompt:
    # save user turn
    st.session_state["history"].append(("user", prompt))

    # retrieve
    hits = search(prompt, k=5)
    if not hits:
        answer_md = "_No results found. Add more files under `./data` and rebuild the index._"
    else:
        top = hits[0]
        answer_md = (
            f"**Top result:** {top['snippet']}\n\n"
            f"_Score: {top['score']:.3f} • Source: {Path(top['source']).name}_\n"
            f"\n<details><summary>See all results</summary>\n\n" +
            "\n".join([f"- **{Path(h['source']).name}** · {h['score']:.3f}\n  \n  {h['snippet']}"
                       for h in hits]) +
            "\n</details>"
        )

    # save assistant turn so it persists after rerun
    st.session_state["history"].append(("assistant", answer_md))
    st.rerun()
