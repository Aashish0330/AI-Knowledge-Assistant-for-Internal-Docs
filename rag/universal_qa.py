# rag/universal_qa.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class QAResult:
    answer: str
    sources: List[str]
    evidence: List[str]  # short supporting snippets


# ---------- Sentence utilities ----------

# add near existing imports
STOPWORDS = {
    "the","a","an","and","or","of","to","for","on","in","with","our","your","their",
    "is","are","be","by","from","as","at","that","this","these","those","it","we",
    "policy","policies","company","organization","guide","guidelines"
}

def key_terms(q: str) -> set[str]:
    q = re.sub(r"[^a-z0-9\s\-]", " ", q.lower())
    terms = {t for t in q.split() if len(t) >= 3 and t not in STOPWORDS}
    # simple synonym expansion for common phrasing
    if "clean" in terms and "desk" in terms:
        terms |= {"clean-desk", "clear", "clear-desk"}
    if "vpn" in terms:
        terms |= {"virtual", "private", "network"}
    return terms


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")

def split_sents(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    # drop ultra-short headings
    return [s for s in sents if len(s.split()) >= 3]


def clean_text(text: str) -> str:
    # remove markdown noise and collapse whitespace
    t = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    t = re.sub(r"^[>\-\*\+]\s+", "", t, flags=re.M)
    t = re.sub(r"[`*_]{1,3}", "", t)
    return _WS.sub(" ", t).strip()


# ---------- Intent detection (generic) ----------

YN_TRIGGERS = (" is ", " are ", " does ", " do ", " should ", " must ", " required ", " allowed ", " prohibited ")
NUM_TRIGGERS = (" how long ", " how often ", " valid ", " expire", " expiration", " retention", " rotate", " timeout", " minutes", " hours", " days", " months", " years")
DEF_TRIGGERS = (" what is ", " describe ", " overview ", " explain ", " introduction ", " about ")
LIST_TRIGGERS = (" steps ", " procedure ", " process ", " checklist ", " how to ", " guidelines ")

def detect_intent(q: str) -> str:
    """Heuristically classify the user question into intent type."""
    ql = " " + q.lower().strip() + " "

    # Definition / descriptive queries take precedence
    if any(t in ql for t in DEF_TRIGGERS) or ql.strip().startswith((" what ", "who ", " describe ", " explain ", " overview ", " about ")):
        return "define"

    # List-oriented
    if any(t in ql for t in LIST_TRIGGERS):
        return "list"

    # Numeric / duration
    if any(t in ql for t in NUM_TRIGGERS):
        return "numeric"

    # Generic yes/no (after others)
    if any(t in ql for t in YN_TRIGGERS):
        return "yesno"

    return "generic"


# ---------- Candidate sentence scoring ----------

def top_candidate_sents(query: str, hits: List[dict], max_sents: int = 12) -> List[Tuple[float, str, str]]:
    """
    Returns (score, sentence, source) ranked by:
      1) TF-IDF similarity to the query
      2) lexical overlap with query terms (big boost)
      3) original hit score
      4) +context window for sentences adjacent to a strong match
    """
    qterms = key_terms(query)

    # collect sentences with their source and base weights
    sents, srcs, weights = [], [], []
    sent_by_src: dict[str, List[str]] = {}
    for h in hits:
        txt = clean_text(h.get("text", ""))
        if not txt:
            continue
        ss = split_sents(txt)
        for s in ss:
            sents.append(s)
            srcs.append(h["source"])
            weights.append(h.get("score", 0.0))
        if ss:
            sent_by_src[h["source"]] = ss

    if not sents:
        return []

    # TF-IDF similarity
    vec = TfidfVectorizer().fit(sents + [query])
    S = vec.transform(sents)
    qv = vec.transform([query])
    tfidf_sim = (S @ qv.T).toarray().ravel()

    # lexical overlap
    def terms_in(s: str) -> set[str]:
        t = re.sub(r"[^a-z0-9\s\-]", " ", s.lower()).split()
        return {w for w in t if len(w) >= 3 and w not in STOPWORDS}

    overlaps = np.zeros(len(sents), dtype="float32")
    hard_match_mask = np.zeros(len(sents), dtype=bool)
    for i, s in enumerate(sents):
        st = terms_in(s)
        jacc = 0.0
        if st and qterms:
            inter = len(st & qterms)
            union = len(st | qterms)
            jacc = inter / union if union else 0.0
            if inter >= 1:
                hard_match_mask[i] = True
        overlaps[i] = jacc

    # base hit weight
    base_w = np.asarray(weights, dtype="float32")

    # combine scores
    combined = 0.55 * tfidf_sim + 0.35 * (overlaps * 2.0) + 0.10 * base_w

    # context window boost: if a sentence in the same doc hard-matches, boost its immediate neighbors
    idx_by_src: dict[str, List[int]] = {}
    for i, src in enumerate(srcs):
        idx_by_src.setdefault(src, []).append(i)

    for src, idxs in idx_by_src.items():
        strong = [i for i in idxs if hard_match_mask[i] and combined[i] > 0]
        for i in strong:
            for j in (i-1, i+1):
                if j in idxs and 0 <= j < len(combined):
                    combined[j] += 0.15  # gentle neighbor boost

    order = np.argsort(-combined)
    return [(float(combined[i]), sents[i], srcs[i]) for i in order[:max_sents]]

# ---------- Generic numeric extraction ----------

_NUM_PATTERNS = [
    (re.compile(r"\b(\d{1,3})\s*(minutes?|mins?)\b", re.I), "minutes"),
    (re.compile(r"\b(\d{1,3})\s*(hours?|hrs?)\b", re.I), "hours"),
    (re.compile(r"\b(\d{1,3})\s*(days?)\b", re.I), "days"),
    (re.compile(r"\b(\d{1,3})\s*(months?)\b", re.I), "months"),
    (re.compile(r"\b(\d{1,3})\s*(years?|yrs?)\b", re.I), "years"),
]

def extract_best_number(sentences: List[str]) -> Optional[Tuple[str, str, str]]:
    """
    From candidate sentences, pick first that carries a numeric duration/period.
    Returns (value, unit, evidence_sentence)
    """
    for s in sentences:
        sl = s.lower()
        for pat, unit in _NUM_PATTERNS:
            m = pat.search(sl)
            if m:
                return m.group(1), unit, s.strip()
    return None


# ---------- Yes/No claim parsing (numbers + simple negation) ----------

def parse_numeric_from_query(q: str) -> Optional[Tuple[int, str]]:
    ql = q.lower()
    for pat, unit in _NUM_PATTERNS:
        m = pat.search(ql)
        if m:
            try:
                return int(m.group(1)), unit
            except:
                pass
    return None

def has_negation(s: str) -> bool:
    return any(w in s.lower() for w in (" not ", " no ", " never ", " prohibited ", " forbidden ", " disallow", " do not "))


# ---------- List extraction ----------

def extract_bullets(paragraphs: List[str], max_items: int = 6) -> List[str]:
    items = []
    bullet_pat = re.compile(r"^\s*[-*•]\s+(.*)$")
    for p in paragraphs:
        for line in p.splitlines():
            m = bullet_pat.match(line)
            if m:
                items.append(m.group(1).strip())
    # fallback: split long sentence with semicolons
    if not items:
        for p in paragraphs:
            if ";" in p and len(p) > 60:
                parts = [x.strip() for x in p.split(";") if len(x.strip()) > 3]
                items.extend(parts)
                break
    return items[:max_items]


# ---------- Main universal QA ----------

def universal_answer(query: str, hits: List[dict]) -> QAResult:
    """
    Domain-agnostic answerer:
    - ranks sentences by similarity,
    - decides intent,
    - composes answer with evidence & sources.
    """
    cands = top_candidate_sents(query, hits, max_sents=18)
    sources_ordered = []
    sent_list = []
    for _, s, src in cands:
        sent_list.append(s)
        if src not in sources_ordered:
            sources_ordered.append(src)

    intent = detect_intent(query)

    # 1) Numeric intent → extract first numeric with unit from top sentences
    if intent == "numeric":
        num = extract_best_number([s for _, s, _ in cands] if cands else sent_list)
        if num:
            val, unit, ev = num
            return QAResult(
                answer=f"{val} {unit}",
                sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                evidence=[ev],
            )

    # 2) Yes/No → if query has a number, compare vs top numeric; else simple boolean presence with negation
    if intent == "yesno":
        claim = parse_numeric_from_query(query)
        if claim:
            cval, cunit = claim
            found = extract_best_number([s for _, s, _ in cands] if cands else sent_list)
            if found:
                val, unit, ev = found
                verdict = "Yes" if (unit.startswith(cunit[:3]) and int(val) == int(cval)) else "No"
                return QAResult(
                    answer=f"{verdict}. Policy indicates **{val} {unit}**.",
                    sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                    evidence=[ev],
                )
        else:
            # Boolean-style question → check for negation cues in top sentence
            if cands:
                best = cands[0][1]
                verdict = "No" if has_negation(best) else "Yes"
                return QAResult(
                    answer=verdict,
                    sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                    evidence=[best],
                )

    # 3) List intent → pull bullets or pseudo-bullets
    if intent == "list":
        paras = [clean_text(h.get("text", "")) for h in hits]
        bullets = extract_bullets(paras)
        if bullets:
            ans = "\n".join(f"- {b}" for b in bullets[:6])
            return QAResult(
                answer=ans,
                sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                evidence=bullets[:2],
            )

    # 4) Define/overview → summarize top sentences
        # 4) Define/overview → summarize top sentences, but prefer lexical matches
    if intent == "define":
        # Prefer sentences that share terms with the query
        qterms = key_terms(query)
        matched = [s for _, s, _ in cands if (set(re.sub(r"[^a-z0-9\s\-]", " ", s.lower()).split()) & qterms)]
        pool = matched if matched else [s for _, s, _ in cands]
        joined = " ".join(pool)
        if joined:
            sents = split_sents(joined)
            if sents:
                vec = TfidfVectorizer().fit_transform(sents)
                sims = cosine_similarity(vec, vec).mean(axis=1)
                idx = np.argsort(-sims)[:2]
                summ = " ".join([sents[i] for i in sorted(idx)]).strip()
                summ = re.sub(r"^[#\s]*", "", summ)
                return QAResult(
                    answer=(f"{summ[0].upper()}{summ[1:]}" if summ else "_No summary found._"),
                    sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                    evidence=[sents[i] for i in sorted(idx)],
                )

    # 5) Generic fallback → stitch 2–3 best sentences
    if cands:
        best = [s for _, s, _ in cands[:3]]
        ans = " ".join(best)
        return QAResult(
            answer=ans,
            sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
            evidence=best[:2],
        )

    return QAResult(answer="_No relevant information found in current documents._", sources=[], evidence=[])