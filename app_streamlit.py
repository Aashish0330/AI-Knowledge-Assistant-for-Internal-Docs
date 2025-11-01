import os, re, pickle, numpy as np, streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("AI Knowledge Assistant (Local RAG)")

# ---------- prefer chunked index; fallback to doc-level ----------
idx_chunks = Path("storage/nn_chunks.pkl")
idx_docs   = Path("storage/nn_index.pkl")
mode = None

if idx_chunks.exists():
    with open(idx_chunks, "rb") as f: payload = pickle.load(f)
    records = payload["records"]                         # [{source, chunk_id, start, end, text}]
    tfidf: TfidfVectorizer = payload["tfidf"]
    use_svd: bool = payload.get("use_svd", False)
    svd: TruncatedSVD | None = payload.get("svd")
    nn: NearestNeighbors = payload["nn"]
    mode = "chunks"
elif idx_docs.exists():
    with open(idx_docs, "rb") as f: payload = pickle.load(f)
    docs = payload["docs"]                              # [{source, text}]
    tfidf: TfidfVectorizer = payload["tfidf"]
    use_svd: bool = payload.get("use_svd", False)
    svd: TruncatedSVD | None = payload.get("svd")
    nn: NearestNeighbors = payload["nn"]
    mode = "docs"
else:
    st.warning("No index found. Build one:\n- Chunked: `python scripts/build_index_sklearn_chunks.py`\n- Simple:  `python scripts/build_index_sklearn.py`")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("Manage documents")

    uploaded_files = st.file_uploader("📄 Add files to ./data", accept_multiple_files=True)
    if uploaded_files:
        from pathlib import Path
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True, parents=True)
        for f in uploaded_files:
            (data_dir / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to ./data")

    if st.button("🔁 Rebuild index"):
        import subprocess, sys
        cmd = [sys.executable, "scripts/build_index_sklearn_chunks.py"]
        with st.status("Rebuilding chunked index...", expanded=True):
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                st.write(out)
                st.success("✅ Index rebuilt. Reloading app...")
                st.experimental_rerun()
            except subprocess.CalledProcessError as e:
                st.error("❌ Failed to rebuild index")
                st.code(e.output or str(e))

def encode(texts):
    V = tfidf.transform(texts)
    Z = svd.transform(V).astype("float32") if (use_svd and svd is not None) else V.toarray().astype("float32")
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return Z

def retrieve(query, k=6):
    qz = encode([query])
    # pull a wider pool then filter by keywords if present (e.g., vpn)
    distances, indices = nn.kneighbors(qz, n_neighbors=max(k, 12), return_distance=True)
    prelim = list(zip(distances[0], indices[0]))
    ql = query.lower()
    require = []
    if "vpn" in ql: require.append("vpn")
    if require:
        def text_at(i): 
            return (records[i]["text"] if mode=="chunks" else docs[i]["text"]).lower()
        filt = [(d,i) for (d,i) in prelim if any(t in text_at(i) for t in require)]
        if filt: prelim = filt
    prelim = sorted(prelim, key=lambda x: x[0])[:k]
    hits=[]
    for d,i in prelim:
        sim = float(1.0 - d)
        if mode=="chunks":
            rec = records[i]
            hits.append({"score": sim, **rec})
        else:
            doc = docs[i]
            hits.append({"score": sim, "source": doc["source"], "chunk_id": 0, "start": 0, "end": len(doc["text"]), "text": doc["text"]})
    return hits

# --- tiny rule-based extractor for VPN timeout ---
def try_extract_password_rotation(query, hits):
    q = query.lower()
    if not any(k in q for k in ("password", "pwd")): return None
    if not any(k in q for k in ("rotate","rotation","expiry","change")): return None
    pat = re.compile(r'\b(?:every|each)\s*(\d{1,3})\s*day', re.I)
    for h in hits[:12]:
        if "password" not in h["text"].lower(): continue
        for sent in re.split(r'(?<=[.!?])\s+', h["text"]):
            if "password" not in sent.lower(): continue
            m = pat.search(sent.lower())
            if m: return int(m.group(1)), h["source"], sent.strip()
    return None

def try_extract_access_key_owner(query, hits):
    q = query.lower()
    if not any(k in q for k in ("access key","api key","keys")): return None
    who_pat = re.compile(r'\b(?:managed|owned|issued|administered)\s+by\s+([A-Za-z &/-]{2,40})', re.I)
    for h in hits[:12]:
        t = h["text"]; tl=t.lower()
        if "key" not in tl: continue
        for sent in re.split(r'(?<=[.!?])\s+', t):
            m = who_pat.search(sent)
            if m: return m.group(1).strip(), h["source"], sent.strip()
    return None

def try_extract_vpn_timeout(query: str, hits):
    q = query.lower()
    if not any(k in q for k in ("vpn","virtual private network")): return None
    if not any(k in q for k in ("timeout","duration","disconnect","idle")): return None
    pat = re.compile(r'\b(?:after|in)?\s*(\d{1,3})\s*(?:minutes?|mins?)\b', re.I)
    # scan top retrieved chunks that mention vpn
    for h in hits[:12]:
        if "vpn" not in h["text"].lower(): continue
        for sent in re.split(r'(?<=[.!?])\s+', h["text"]):
            if "vpn" not in sent.lower(): continue
            m = pat.search(sent)
            if m:
                return int(m.group(1)), h["source"], sent.strip()
    return None

def answer_composer(query, hits, max_sents=3):
    # simple sentence selector mixing lexical overlap + vector score
    qlow = re.sub(r'[^a-z0-9 ]+',' ', query.lower())
    q_terms = {t for t in qlow.split() if len(t)>=3}
    cand=[]
    for h in hits:
        for s in re.split(r'(?<=[.!?])\s+', h["text"])[:8]:
            low = re.sub(r'[^a-z0-9 ]+',' ', s.lower())
            s_terms = {t for t in low.split() if len(t)>=3}
            overlap = len(q_terms & s_terms) + (1 if any(t in low for t in q_terms) else 0)
            score = overlap*0.6 + h["score"]*0.4
            cand.append((score, s, h["source"]))
    cand.sort(reverse=True)
    picked=[]; seen=set()
    for sc, s, src in cand:
        norm=s.lower()
        if norm in seen: continue
        seen.add(norm); picked.append((sc,s,src))
        if len(picked)>=max_sents: break
    if not picked:
        return "_No clear answer found in current documents._", []
    ans = " ".join(s for _,s,_ in picked)
    srcs=[]; [srcs.append(src) for _,_,src in picked if src not in srcs]
    return ans, srcs

if "history" not in st.session_state: st.session_state["history"]=[]

for role,msg in st.session_state["history"]:
    with st.chat_message(role): st.markdown(msg)


prompt = st.chat_input("Ask about policies, runbooks, architecture…")

if prompt:
    st.session_state["history"].append(("user", prompt))
    top_hits = retrieve(prompt, k=6)

    # exact extractors first
    vpn = try_extract_vpn_timeout(prompt, top_hits) if "try_extract_vpn_timeout" in globals() else None
    pwd_res = try_extract_password_rotation(prompt, top_hits) if "try_extract_password_rotation" in globals() else None
    key_owner = try_extract_access_key_owner(prompt, top_hits) if "try_extract_access_key_owner" in globals() else None

    if vpn:
        minutes, src, sent = vpn
        answer = f"VPN session times out after **{minutes} minutes** of inactivity."
        srcs = [src]
    elif pwd_res:
        days, src, sent = pwd_res
        answer = f"Passwords must be rotated every **{days} days**."
        srcs = [src]
    elif key_owner:
        who, src, sent = key_owner
        answer = f"Access keys are managed by **{who}**."
        srcs = [src]
    else:
        answer, srcs = answer_composer(prompt, top_hits, max_sents=3)

    # Minimal render: just answer + single best source
    primary_src = srcs[0] if srcs else (top_hits[0]["source"] if top_hits else None)
    md = f"**Answer:** {answer}\n\n"
    if primary_src:
        from pathlib import Path as _P
        md += f"**Source:** `{_P(primary_src).name}`"

    st.session_state["history"].append(("assistant", md))
    st.rerun()
