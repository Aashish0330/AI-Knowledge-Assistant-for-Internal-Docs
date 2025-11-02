# rag/llm.py
from __future__ import annotations
import json, os, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# --- Provider selection ---
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" | "ollama"

# OpenAI (official SDK)
_openai = None
if PROVIDER == "openai":
    from openai import OpenAI
    _openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ollama (local)
_ollama_client = None
if PROVIDER == "ollama":
    try:
        import ollama
        _ollama_client = ollama
    except Exception:
        _ollama_client = None

DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")  # change if you like
MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "600"))
TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------- Prompt ----------
SYS_PROMPT = """You are a careful, citation-first assistant answering only from the provided context.
If the answer is not in the context, say you don't find it in the documents.
Answer briefly and professionally. Always include citations with filenames (and page markers if present).
"""

USER_PROMPT_TEMPLATE = """Answer the user question using ONLY the context chunks below.
Return a short answer (2–5 sentences) and citations to the exact chunks you used.

# QUESTION
{question}

# CONTEXT CHUNKS (each begins with ===)
{contexts}

# REQUIRED OUTPUT (JSON)
Return valid JSON with the shape:
{{
  "answer": "<brief grounded answer>",
  "citations": ["<filename_or_path>#optional_page_or_chunk>", ...]
}}

Rules:
- Only use facts present in the context.
- Cite at least one chunk you used. Use the chunk header to form citation.
- If insufficient info, set "answer" to "I couldn't find this in the documents." and leave an empty list for "citations".
"""

def _format_context(chunks: List[Dict], max_chars: int = 8000) -> str:
    """
    Format retrieved chunks with headers; trim to budget.
    """
    out = []
    total = 0
    for i, ch in enumerate(chunks, 1):
        src = ch["source"]
        # keep page marker if present in text
        header = f"=== [{i}] {src} (chunk {ch.get('chunk_id','?')})"
        body = ch["text"].strip()
        block = f"{header}\n{body}\n"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n".join(out)

@dataclass
class LLMResult:
    answer: str
    citations: List[str]
    raw: str

def _safe_json_parse(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        # Attempt to extract a JSON object via regex
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def generate_llm_answer(question: str, top_chunks: List[Dict]) -> LLMResult:
    contexts = _format_context(top_chunks)
    user = USER_PROMPT_TEMPLATE.format(question=question, contexts=contexts)

    if PROVIDER == "openai":
        if _openai is None:
            raise RuntimeError("OpenAI client unavailable. Set OPENAI_API_KEY.")
        resp = _openai.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content.strip()
    elif PROVIDER == "ollama":
        if _ollama_client is None:
            raise RuntimeError("Ollama client unavailable. pip install ollama and run `ollama run <model>`")
        model = os.getenv("LLM_MODEL_NAME", "llama3.1")
        r = _ollama_client.chat(model=model, messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ])
        text = (r.get("message") or {}).get("content", "").strip()
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {PROVIDER}")

    data = _safe_json_parse(text) or {}
    answer = (data.get("answer") or "").strip()
    cites  = data.get("citations") or []

    # Minimal guardrails
    if not answer:
        answer = "I couldn't parse a valid answer."
    if not isinstance(cites, list):
        cites = []

    return LLMResult(answer=answer, citations=cites, raw=text)