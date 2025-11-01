import json, os
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from .prompts import prompt
from .budget import budget_docs

def render_context(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "")
        lines.append(f"[{i}] Source: {src} Page: {page}\n{d.page_content}")
    return "\n\n".join(lines)

def make_chain():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    def run(query: str, retrieve_fn, reranker=None, k: int = 8) -> Dict:
        docs: List[Document] = retrieve_fn(query, k=k)
        if reranker:
            docs = reranker.rerank(query, docs, top_k=min(6, k))
        docs = budget_docs(docs)
        ctx = render_context(docs)
        res = llm.invoke(prompt.format(question=query, context=ctx))
        try:
            payload = json.loads(res.content)
        except Exception:
            payload = {"answer_markdown": res.content, "citations": []}
        # attach actual citations
        payload["citations"] = [
            {"source": d.metadata.get("source", "unknown"), "page": str(d.metadata.get("page", ""))}
            for d in docs
        ]
        return payload

    return run
