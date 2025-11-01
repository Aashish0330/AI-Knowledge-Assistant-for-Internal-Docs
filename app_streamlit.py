# app_streamlit.py
import os
import re
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from rag.universal_qa import universal_answer, QAResult
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Dynamic extractors registry (numeric/boolean/entity)
from rag.extractors import EXTRACTORS, parse_claim_number

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("📚 AI Knowledge Assistant (Local RAG)")

# ============================================================
# Load index (prefer chunked index)
# ============================================================
IDX_CHUNKS = Path("storage/nn_chunks.pkl")
IDX_DOCS   = Path("storage/nn_index.pkl")

mode: Optional[str] = None
records: Optional[List[dict]] = None   # for chunked index: [{source, chunk_id, start, end, text}]
docs: Optional[List[dict]] = None      # for doc-level index: [{source, text}]
tfidf = None
svd = None
nn: Optional[NearestNeighbors] = None
use_svd: bool = False

def _load_index():
    """Load either chunked or doc-level index from storage/."""
    global mode, records, docs, tfidf, svd, nn, use_svd
    if IDX_CHUNKS.exists():
        payload = pickle.load(open(IDX_CHUNKS, "rb"))
        records = payload["records"]
        tfidf   = payload["tfidf"]
        use_svd = payload.get("use_svd", False)
        svd     = payload.get("svd")
        nn      = payload["nn"]
        mode    = "chunks"
    elif IDX_DOCS.exists():
        payload = pickle.load(open(IDX_DOCS, "rb"))
        docs    = payload["docs"]
        tfidf   = payload["tfidf"]
        use_svd = payload.get("use_svd", False)
        svd     = payload.get("svd")
        nn      = payload["nn"]
        mode    = "docs"
    else:
        mode = None

def _num_items() -> int:
    if mode == "chunks" and records is not None:
        return len(records)
    if mode == "docs" and docs is not None:
        return len(docs)
    return 0

_load_index()

# ============================================================
# Sidebar: diagnostics & quick actions
# ============================================================
with st.sidebar:
    st.header("Diagnostics")
    st.write("Index mode:", mode or "—")
    st.write("Use SVD:", bool(use_svd))
    st.write("Items:", _num_items())

    st.divider()
    st.subheader("Manage documents")
    uploaded = st.file_uploader("📄 Add files to ./data", accept_multiple_files=True)
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

# ============================================================
# Encode / Retrieve
# ============================================================
def encode(texts: List[str]) -> np.ndarray:
    V = tfidf.transform(texts)
    if use_svd and svd is not None:
        Z = svd.transform(V).astype("float32")
    else:
        Z = V.toarray().astype("float32")
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return Z

def retrieve(query: str, k: int = 6) -> List[dict]:
    n_items = _num_items()
    if n_items == 0 or nn is None:
        return []
    qz = encode([query])
    n = max(1, min(k, n_items))
    distances, indices = nn.kneighbors(qz, n_neighbors=n, return_distance=True)

    hits = []
    if mode == "chunks":
        for d, i in zip(distances[0], indices[0]):
            sim = float(1.0 - d)
            rec = records[i]
            hits.append({"score": sim, **rec})
    else:
        for d, i in zip(distances[0], indices[0]):
            sim = float(1.0 - d)
            doc = docs[i]
            hits.append({"score": sim, "source": doc["source"], "text": doc["text"]})
    return hits

# ============================================================
# Cleaning & summarizing
# ============================================================
def clean_policy_text(t: str) -> str:
    """Remove markdown headers/bullets/code markers; collapse whitespace."""
    lines = [ln.strip() for ln in t.splitlines()]
    out = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith("#"):
            continue
        if ln.startswith(("-", "*")) and len(ln.split()) <= 4:
            continue
        ln = re.sub(r"^#+\s*", "", ln)
        ln = re.sub(r"[*_`]", "", ln)
        out.append(ln)
    txt = " ".join(out)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def summarize_text(text: str, max_sentences: int = 2) -> str:
    """Extractive summary using TF-IDF + cosine centrality on cleaned sentences."""
    text = clean_policy_text(text)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sents = [s for s in sents if not (len(s.split()) <= 4 and s.istitle())]
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return " ".join(sents).strip()
    vec = TfidfVectorizer().fit_transform(sents)
    scores = cosine_similarity(vec, vec)
    centrality = scores.mean(axis=1)
    ranked = np.argsort(-centrality)
    top = sorted(ranked[:max_sentences])
    return " ".join([sents[i] for i in top]).strip()

def _strip_markdown_aggressive(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^[>\-\*\+]\s+", "", text, flags=re.M)
    text = re.sub(r"[`*_]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def conceptual_summary_from_hits(hits: List[dict], max_sentences: int = 2) -> str:
    if not hits:
        return ""
    raw = " ".join(h["text"] for h in hits if h.get("text"))
    if not raw.strip():
        return ""
    cleaned_gentle = clean_policy_text(raw)
    cleaned = cleaned_gentle or _strip_markdown_aggressive(raw)
    intro_match = re.search(r"\bIntroduction\b[:\-]?\s*(.+?)(?=(?:\s+[A-Z][^\s].{0,40}\b)|$)", cleaned, flags=re.S)
    paragraph = intro_match.group(1).strip() if intro_match else cleaned
    summary = summarize_text(paragraph, max_sentences=max_sentences).strip()
    if not summary:
        paras = [p.strip() for p in re.split(r"\n{2,}", paragraph) if len(p.strip()) > 30]
        if paras:
            summary = " ".join(paras[:2])[:400].strip()
        else:
            summary = _strip_markdown_aggressive(raw)[:350].strip()
    summary = re.sub(r"^\s*GreenPack Solutions\s*[-–—:]\s*", "GreenPack Solutions ", summary)
    return summary

def answer_composer(query: str, hits: List[dict], max_sents: int = 3) -> Tuple[str, List[str]]:
    qlow = re.sub(r"[^a-z0-9 ]+", " ", query.lower())
    q_terms = {t for t in qlow.split() if len(t) >= 3}
    cand = []
    for h in hits:
        text_clean = clean_policy_text(h["text"])
        for s in re.split(r"(?<=[.!?])\s+", text_clean)[:8]:
            low = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
            s_terms = {t for t in low.split() if len(t) >= 3}
            overlap = len(q_terms & s_terms) + (1 if any(t in low for t in q_terms) else 0)
            score = overlap * 0.6 + h["score"] * 0.4
            cand.append((score, s, h["source"]))
    cand.sort(reverse=True)
    picked, seen = [], set()
    for sc, s, src in cand:
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        picked.append((sc, s, src))
        if len(picked) >= max_sents * 2:
            break
    if not picked:
        return "_No clear answer found in current documents._", []
    text_block = " ".join(s for _, s, _ in picked)
    srcs = list({src for _, _, src in picked})
    # Conceptual questions -> summarize
    if re.search(r"\b(what|who|describe|overview|about|explain|introduction)\b", query.lower()):
        summary = summarize_text(text_block, max_sentences=2)
        return summary, srcs
    # Otherwise return top sentences
    ans = " ".join(s for _, s, _ in picked[:max_sents])
    return ans, srcs

# ============================================================
# Chat history
# ============================================================
if "history" not in st.session_state:
    st.session_state["history"] = []

for role, msg in st.session_state["history"]:
    with st.chat_message(role):
        st.markdown(msg)

# ============================================================
# Empty index guard
# ============================================================
if _num_items() == 0:
    st.warning(
        "Index is empty.\n\n"
        "1) Put PDFs/MD/TXT files in ./data\n"
        "2) Build index: python scripts/build_index_sklearn_chunks.py\n"
        "3) Then ask questions below."
    )

# ============================================================
# Chat input & response
# ============================================================
prompt = st.chat_input("Ask about policies (passwords, VPN, access keys, incidents, retention, etc.)…")

if prompt:
    st.session_state["history"].append(("user", prompt))

    # Retrieve top hits
    top_hits = retrieve(prompt, k=6)

    ua: QAResult = universal_answer(prompt, top_hits)
    if ua and ua.answer and ua.answer.strip() and ua.answer != "_No relevant information found in current documents._":
        md = f"**Answer:** {ua.answer}\n\n"
        if ua.sources:
            md += f"**Source:** `{ua.sources[0]}`"
        st.session_state["history"].append(("assistant", md))
        st.rerun()



    # Run all extractors dynamically
    facts: Dict[str, Tuple[str, str, str, str]] = {}  # key -> (value, unit, source, evidence)
    for key, fn in EXTRACTORS.items():
        try:
            res = fn(top_hits)
            if res:
                facts[key] = res
        except Exception:
            pass  # keep UI resilient

    ql = prompt.lower()
    numeric_intent = any(w in ql for w in [
        "how long", "how often", "valid", "expire", "expiration",
        "rotation", "timeout", "retention", "respond within",
        "minutes", "hours", "days", "months", "years", "is it", "is the"
    ])

    # Map triggers -> fact key + unit hint
    claim_map = [
        ("password", "password_days", "days"),
        (("access key", "api key", "keys"), "access_key_days", "days"),
        ("vpn timeout", "vpn_timeout_minutes", "minutes"),
        (("vpn credential", "vpn password"), "vpn_cred_days", "days"),
        (("incident", "respond", "triage", "soc", "irt"), "incident_response_time", None),
        (("retention", "retain", "archive"), "data_retention", None),
    ]

    answered = False

    # Numeric facts and Yes/No claims
    if numeric_intent:
        for trigger, fact_key, unit_hint in claim_map:
            trig_hit = False
            if isinstance(trigger, tuple):
                trig_hit = any(t in ql for t in trigger)
            else:
                trig_hit = trigger in ql
            if not trig_hit:
                continue
            if fact_key in facts:
                val, unit, src, ev = facts[fact_key]
                # If the user stated a number, do Yes/No; else return the fact
                claim_val = parse_claim_number(prompt, unit_hint or unit or "")
                if claim_val is not None and val.isdigit():
                    verdict = "Yes" if int(claim_val) == int(val) else "No"
                    md = f"**Answer:** {verdict}. Policy specifies **{val} {unit or unit_hint or ''}**.\n\n**Source:** `{Path(src).name}`"
                else:
                    pretty_unit = f" {unit}" if unit else ""
                    md = f"**Answer:** {val}{pretty_unit}.\n\n**Source:** `{Path(src).name}`"
                st.session_state["history"].append(("assistant", md))
                answered = True
                break

    # Boolean/entity questions
    if not answered and "mfa" in ql and "vpn" in ql and "vpn_mfa_required" in facts:
        _, _, src, _ = facts["vpn_mfa_required"]
        st.session_state["history"].append(("assistant", f"**Answer:** Yes — VPN requires MFA.\n\n**Source:** `{Path(src).name}`"))
        answered = True

    if not answered and any(w in ql for w in ["aes-256", "aes256", "aes 256", "encryption standard"]) and "aes_256" in facts:
        _, _, src, _ = facts["aes_256"]
        st.session_state["history"].append(("assistant", f"**Answer:** Yes — AES-256 is referenced.\n\n**Source:** `{Path(src).name}`"))
        answered = True

    if not answered and any(w in ql for w in ["who manages access key", "who manages access keys", "who owns keys", "key owner", "who issues keys"]) and "access_key_owner" in facts:
        owner, _, src, _ = facts["access_key_owner"]
        st.session_state["history"].append(("assistant", f"**Answer:** Access keys are managed by **{owner}**.\n\n**Source:** `{Path(src).name}`"))
        answered = True

    # Conceptual summary (what/describe/about/overview/explain/introduction)
    if not answered and re.search(r"\b(what|who|describe|overview|about|explain|introduction)\b", prompt.lower()):
        summary = conceptual_summary_from_hits(top_hits, max_sentences=2)
        if summary:
            primary_src = top_hits[0]["source"] if top_hits else None
            md = f"**Answer:** {summary}\n\n"
            if primary_src:
                md += f"**Source:** `{Path(primary_src).name}`"
            st.session_state["history"].append(("assistant", md))
            answered = True

    # Fallback: compose from snippets
    if not answered:
        ans, srcs = answer_composer(prompt, top_hits, max_sents=3)
        primary_src = srcs[0] if srcs else (top_hits[0]["source"] if top_hits else None)
        md = f"**Answer:** {ans}\n\n"
        if primary_src:
            md += f"**Source:** `{Path(primary_src).name}`"
        st.session_state["history"].append(("assistant", md))

    st.rerun()