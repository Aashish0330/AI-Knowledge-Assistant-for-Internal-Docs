# rag/extractors.py
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# ---------- Utilities

_NUM_WORDS = {
    "ten": 10, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40,
    "sixty": 60, "ninety": 90, "hundred": 100,
}

def _word_to_num(s: str) -> Optional[int]:
    s = s.lower().strip()
    return _NUM_WORDS.get(s)

def _clean_sents(text: str) -> List[str]:
    # sentence split (simple)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _has_any(s: str, keys: List[str]) -> bool:
    s = s.lower()
    return any(k in s for k in keys)

def _num_days(s: str) -> Optional[int]:
    s = s.lower()
    m = re.search(r"(?:every|each|within|for)\s*(\d{1,3})\s*day", s)
    if m: return int(m.group(1))
    m2 = re.search(r"\b(twenty|thirty|forty|sixty|ninety|ten|twelve|hundred)\b\s*day", s)
    if m2:
        n = _word_to_num(m2.group(1))
        if n is not None: return n
    return None

def _num_minutes(s: str) -> Optional[int]:
    s = s.lower()
    m = re.search(r"(\d{1,3})\s*(?:minute|minutes|min|mins)\b", s)
    if m: return int(m.group(1))
    return None

def _num_hours(s: str) -> Optional[int]:
    s = s.lower()
    m = re.search(r"(\d{1,3})\s*(?:hour|hours|hr|hrs)\b", s)
    if m: return int(m.group(1))
    return None

def _num_months(s: str) -> Optional[int]:
    s = s.lower()
    m = re.search(r"(\d{1,3})\s*(?:month|months)\b", s)
    if m: return int(m.group(1))
    return None

def _num_years(s: str) -> Optional[int]:
    s = s.lower()
    m = re.search(r"(\d{1,3})\s*(?:year|years|yr|yrs)\b", s)
    if m: return int(m.group(1))
    return None

# ---------- Generic extractor interface

# Each extractor returns: (normalized_value, unit, source_path, evidence_sentence) OR None
ExtractorReturn = Optional[Tuple[str, str, str, str]]

def _scan_hits(hits: List[dict], sent_filter, value_parser) -> ExtractorReturn:
    for h in hits:
        text = h.get("text", "")
        for sent in _clean_sents(text):
            if sent_filter(sent):
                val = value_parser(sent)
                if val is not None:
                    return (str(val), "", h["source"], sent.strip())
    return None

# ---------- Concrete extractors

def ex_password_rotation_days(hits: List[dict]) -> ExtractorReturn:
    def sent_filter(s: str) -> bool:
        s2 = s.lower()
        return "password" in s2 and _has_any(s2, ["rotate", "rotation", "change", "changed"])
    def parser(s: str): return _num_days(s)
    r = _scan_hits(hits, sent_filter, parser)
    if r: return (r[0], "days", r[2], r[3])
    return None

def ex_access_key_rotation_days(hits: List[dict]) -> ExtractorReturn:
    def sent_filter(s: str) -> bool:
        s2 = s.lower()
        return _has_any(s2, ["access key", "access keys", "api key"]) and _has_any(s2, ["rotate", "rotation", "change"])
    def parser(s: str): return _num_days(s)
    r = _scan_hits(hits, sent_filter, parser)
    if r: return (r[0], "days", r[2], r[3])
    return None

def ex_vpn_timeout_minutes(hits: List[dict]) -> ExtractorReturn:
    def sent_filter(s: str) -> bool:
        s2 = s.lower()
        return "vpn" in s2 and _has_any(s2, ["timeout", "disconnect", "inactivity", "idle"])
    def parser(s: str): return _num_minutes(s)
    r = _scan_hits(hits, sent_filter, parser)
    if r: return (r[0], "minutes", r[2], r[3])
    return None

def ex_vpn_credentials_days(hits: List[dict]) -> ExtractorReturn:
    def sent_filter(s: str) -> bool:
        s2 = s.lower()
        return "vpn" in s2 and _has_any(s2, ["credential", "password"]) and _has_any(s2, ["rotate", "change", "rotation"])
    def parser(s: str): return _num_days(s)
    r = _scan_hits(hits, sent_filter, parser)
    if r: return (r[0], "days", r[2], r[3])
    return None

def ex_incident_response_time(hits: List[dict]) -> ExtractorReturn:
    def sent_filter(s: str) -> bool:
        return _has_any(s.lower(), ["incident", "response", "triage", "soc", "irt"])
    def parser(s: str):
        return _num_minutes(s) or _num_hours(s)
    r = _scan_hits(hits, sent_filter, parser)
    if r:
        # best-effort unit detection
        unit = "minutes" if "min" in r[3].lower() or "minute" in r[3].lower() else "hours"
        return (r[0], unit, r[2], r[3])
    return None

def ex_data_retention(hits: List[dict]) -> ExtractorReturn:
    def sent_filter(s: str) -> bool:
        return _has_any(s.lower(), ["retention", "retain", "archive", "kept for"])
    def parser(s: str):
        return _num_months(s) or _num_years(s)
    r = _scan_hits(hits, sent_filter, parser)
    if r:
        unit = "months" if "month" in r[3].lower() else ("years" if "year" in r[3].lower() else "")
        return (r[0], unit, r[2], r[3])
    return None

def ex_vpn_mfa_required(hits: List[dict]) -> ExtractorReturn:
    # boolean detector
    for h in hits:
        for sent in _clean_sents(h.get("text","")):
            s = sent.lower()
            if "vpn" in s and ("mfa" in s or "multi-factor" in s or "multi factor" in s):
                return ("yes", "bool", h["source"], sent.strip())
    return None

def ex_aes256(hits: List[dict]) -> ExtractorReturn:
    for h in hits:
        for sent in _clean_sents(h.get("text","")):
            s = sent.lower()
            if "aes-256" in s or "aes256" in s or "aes 256" in s:
                return ("yes", "bool", h["source"], sent.strip())
    return None

def ex_access_key_owner(hits: List[dict]) -> ExtractorReturn:
    who_pat = re.compile(r"\b(?:managed|owned|issued|administered)\s+by\s+([A-Za-z &/-]{2,60})", re.I)
    for h in hits:
        for sent in _clean_sents(h.get("text","")):
            m = who_pat.search(sent)
            if m:
                return (m.group(1).strip(), "org", h["source"], sent.strip())
    return None

# ---------- Registry

EXTRACTORS = {
    # numeric
    "password_days": ex_password_rotation_days,
    "access_key_days": ex_access_key_rotation_days,
    "vpn_timeout_minutes": ex_vpn_timeout_minutes,
    "vpn_cred_days": ex_vpn_credentials_days,
    "incident_response_time": ex_incident_response_time,
    "data_retention": ex_data_retention,
    # booleans / entities
    "vpn_mfa_required": ex_vpn_mfa_required,
    "aes_256": ex_aes256,
    "access_key_owner": ex_access_key_owner,
}

# ---------- Claim parsing for Yes/No

def parse_claim_number(query: str, unit_hint: str) -> Optional[int]:
    q = query.lower()
    if unit_hint == "days":
        m = re.search(r"(\d{1,3})\s*day", q)
        if m: return int(m.group(1))
        for w, n in _NUM_WORDS.items():
            if re.search(rf"\b{w}\b\s*day", q): return n
        return None
    if unit_hint == "minutes":
        m = re.search(r"(\d{1,3})\s*(?:minute|min)\b", q)
        return int(m.group(1)) if m else None
    if unit_hint == "hours":
        m = re.search(r"(\d{1,3})\s*(?:hour|hr)\b", q)
        return int(m.group(1)) if m else None
    if unit_hint in ("months","years"):
        m = re.search(r"(\d{1,3})\s*"+unit_hint[:-1]+r"\w*\b", q)
        return int(m.group(1)) if m else None
    return None