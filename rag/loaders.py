# rag/loaders.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, List, Dict, Tuple
from pypdf import PdfReader

TEXT_EXTS = {".txt", ".md", ".markdown"}
PDF_EXTS  = {".pdf"}

def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf_file(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception:
        reader = PdfReader(str(path))
        try:
            reader.decrypt("")
        except Exception:
            pass

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        # light cleanup
        raw = re.sub(r"-\n(?=[a-z])", "", raw)  # de-hyphenation
        raw = re.sub(r"\n{2,}", "\n\n", raw)
        raw = raw.strip()
        if raw:
            pages.append(f"[PAGE {i+1}]\n{raw}")
    return "\n\n".join(pages)

def iter_documents(data_dir: Path) -> Iterable[Tuple[str, str]]:
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in TEXT_EXTS:
            yield (str(p), _read_text_file(p))
        elif ext in PDF_EXTS:
            yield (str(p), _read_pdf_file(p))

def load_corpus_with_metadata(data_dir: Path) -> List[Dict]:
    out: List[Dict] = []
    for src, txt in iter_documents(data_dir):
        t = (txt or "").strip()
        if t:
            out.append({"source": src, "text": t})
    return out