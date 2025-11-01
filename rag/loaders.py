from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, TextLoader
)
from langchain.schema import Document

SUPPORTED = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}

def load_docs(data_dir: str = "data") -> List[Document]:
    """Load all supported files under data_dir into Document objects."""
    docs: List[Document] = []
    for p in Path(data_dir).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            continue
        if p.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(p)).load()
        elif p.suffix.lower() in {".html", ".htm"}:
            docs += UnstructuredHTMLLoader(str(p)).load()
        elif p.suffix.lower() in {".md", ".markdown"}:
            docs += UnstructuredMarkdownLoader(str(p)).load()
        elif p.suffix.lower() in {".txt"}:
            docs += TextLoader(str(p), encoding="utf-8").load()

    # normalize metadata
    for d in docs:
        d.metadata["source"] = d.metadata.get("source") or d.metadata.get("file_path") or d.metadata.get("source")
    return docs
