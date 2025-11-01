import tiktoken
from typing import List
from langchain.schema import Document

def budget_docs(docs: List[Document], model: str = "gpt-4o-mini", max_tokens: int = 6000, reserve: int = 800) -> List[Document]:
    enc = tiktoken.get_encoding("cl100k_base")
    limit = max_tokens - reserve
    kept, used = [], 0
    for d in docs:
        t = len(enc.encode(d.page_content))
        if used + t > limit:
            break
        kept.append(d)
        used += t
    return kept
