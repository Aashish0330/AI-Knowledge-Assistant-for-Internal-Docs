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
    evidence: List[str]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")

STOPWORDS = {
    "the","a","an","and","or","of","to","for","on","in","with","our","your","their",
    "is","are","be","by","from","as","at","that","this","these","those","it","we",
    "policy","policies","company","organization","guide","guidelines"
}

YN_TRIGGERS  = (" is ", " are ", " does ", " do ", " should ", " must ", " required ", " allowed ", " prohibited ")
NUM_TRIGGERS = (" how long ", " how often ", " valid ", " expire", " expiration", " retention", " rotate", " timeout", " minutes", " hours", " days", " months", " years")
DEF_TRIGGERS = (" what is ", " describe ", " overview ", " explain ", " introduction ", " about ")
LIST_TRIGGERS= (" steps ", " procedure ", " process ", " checklist ", " how to ", " guidelines ")

def split_sents(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return [s for s in sents if len(s.split()) >= 3]

def clean_text(text: str) -> str:
    t = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)     # headers
    t = re.sub(r"^[>\-\*\+]\s+", "", t, flags=re.M)     # bullets/quotes
    t = re.sub(r"[`*_]{1,3}", "", t)                    # inline md
    return _WS.sub(" ", t).strip()

def key_terms(q: str) -> set[str]:
    q = re.sub(r"[^a-z0-9\s\-]", " ", q.lower())
    terms = {t for t in q.split() if len(t) >= 3 and t not in STOPWORDS}
    if "clean" in terms and "desk" in terms:
        terms |= {"clean-desk", "clear", "clear-desk", "workspace", "workstation", "work-area", "work area"}
    if "vpn" in terms:
        terms |= {"virtual", "private", "network"}
    return terms

def detect_intent(q: str) -> str:
    """Definition/overview takes precedence to avoid yes/no misfires."""
    ql = " " + q.lower().strip() + " "
    if any(t in ql for t in DEF_TRIGGERS) or ql.strip().startswith(("what ", "who ", " describe ", " explain ", " overview ", " about ")):
        return "define"
    if any(t in ql for t in LIST_TRIGGERS):
        return "list"
    if any(t in ql for t in NUM_TRIGGERS):
        return "numeric"
    if any(t in ql for t in YN_TRIGGERS):
        return "yesno"
    return "generic"

def top_candidate_sents(query: str, hits: List[dict], max_sents: int = 12) -> List[Tuple[float, str, str]]:
    qterms = key_terms(query)
    sents, srcs, weights = [], [], []
    idx_by_src = {}

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
            idx_by_src[h["source"]] = list(range(len(sents)-len(ss), len(sents)))

    if not sents:
        return []

    vec = TfidfVectorizer().fit(sents + [query])
    S = vec.transform(sents)
    qv = vec.transform([query])
    tfidf_sim = (S @ qv.T).toarray().ravel()

    def terms_in(s: str) -> set[str]:
        t = re.sub(r"[^a-z0-9\s\-]", " ", s.lower()).split()
        return {w for w in t if len(w) >= 3 and w not in STOPWORDS}

    overlaps = np.zeros(len(sents), dtype="float32")
    hard_match = np.zeros(len(sents), dtype=bool)
    for i, s in enumerate(sents):
        st = terms_in(s)
        inter = len(st & qterms)
        union = len(st | qterms) if (st or qterms) else 1
        jacc = inter / union if union else 0.0
        overlaps[i] = jacc
        if inter >= 1:
            hard_match[i] = True

    base_w = np.asarray(weights, dtype="float32")
    combined = 0.55 * tfidf_sim + 0.35 * (overlaps * 2.0) + 0.10 * base_w

    # neighbor boost
    for src, idxs in idx_by_src.items():
        strong = [i for i in idxs if hard_match[i] and combined[i] > 0]
        for i in strong:
            for j in (i-1, i+1):
                if j in idxs and 0 <= j < len(combined):
                    combined[j] += 0.15

    order = np.argsort(-combined)
    return [(float(combined[i]), sents[i], srcs[i]) for i in order[:max_sents]]

_NUM_PATTERNS = [
    (re.compile(r"\b(\d{1,3})\s*(minutes?|mins?)\b", re.I), "minutes"),
    (re.compile(r"\b(\d{1,3})\s*(hours?|hrs?)\b",  re.I), "hours"),
    (re.compile(r"\b(\d{1,3})\s*(days?)\b",        re.I), "days"),
    (re.compile(r"\b(\d{1,3})\s*(months?)\b",      re.I), "months"),
    (re.compile(r"\b(\d{1,3})\s*(years?|yrs?)\b",  re.I), "years"),
]

def extract_best_number(sentences: List[str]) -> Optional[Tuple[str, str, str]]:
    for s in sentences:
        sl = s.lower()
        for pat, unit in _NUM_PATTERNS:
            m = pat.search(sl)
            if m:
                return m.group(1), unit, s.strip()
    return None

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
    return any(w in s.lower() for w in (" not ", " no ", " never ", " prohibited ", " forbidden ", " disallow ", " do not "))

def extract_bullets(paragraphs: List[str], max_items: int = 6) -> List[str]:
    items = []
    bullet_pat = re.compile(r"^\s*[-*•]\s+(.*)$")
    for p in paragraphs:
        for line in p.splitlines():
            m = bullet_pat.match(line)
            if m:
                items.append(m.group(1).strip())
    if not items:
        for p in paragraphs:
            if ";" in p and len(p) > 60:
                parts = [x.strip() for x in p.split(";") if len(x.strip()) > 3]
                items.extend(parts)
                break
    return items[:max_items]

def universal_answer(query: str, hits: List[dict]) -> QAResult:
    cands = top_candidate_sents(query, hits, max_sents=18)
    sources_ordered, sent_list = [], []
    for _, s, src in cands:
        sent_list.append(s)
        if src not in sources_ordered:
            sources_ordered.append(src)

    intent = detect_intent(query)

    # numeric
    if intent == "numeric":
        num = extract_best_number([s for _, s, _ in cands] if cands else sent_list)
        if num:
            val, unit, ev = num
            return QAResult(
                answer=f"{val} {unit}",
                sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                evidence=[ev],
            )

    # yes/no
    if intent == "yesno":
        if re.search(r"\b(what|describe|overview|explain|introduction|about)\b", query.lower()):
            intent = "define"
        else:
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
            if cands:
                best = cands[0][1]
                verdict = "No" if has_negation(best) else "Yes"
                return QAResult(
                    answer=verdict,
                    sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                    evidence=[best],
                )

    # list / steps
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

    # define / overview
    if intent == "define":
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
                summ = (f"{summ[0].upper()}{summ[1:]}" if summ else "_No summary found._")
                return QAResult(
                    answer=summ,
                    sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
                    evidence=[sents[i] for i in sorted(idx)],
                )

    # generic fallback
    if cands:
        best = [s for _, s, _ in cands[:3]]
        ans = " ".join(best)
        return QAResult(
            answer=ans,
            sources=[Path(sources_ordered[0]).name] if sources_ordered else [],
            evidence=best[:2],
        )

    return QAResult(answer="_No relevant information found in current documents._", sources=[], evidence=[])