from sentence_transformers import CrossEncoder
from typing import List
from langchain.schema import Document

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        sorted_docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return sorted_docs[:top_k]
