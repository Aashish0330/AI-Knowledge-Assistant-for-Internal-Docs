import os, pickle
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

class HybridRetriever:
    """Hybrid = FAISS (dense) + BM25 (sparse) with simple score fusion."""
    def __init__(self, faiss_store: FAISS, bm25: BM25Okapi, corpus_docs: List[Document], alpha: float = 0.5):
        self.faiss_store = faiss_store
        self.bm25 = bm25
        self.corpus_docs = corpus_docs
        self.alpha = alpha  # weight for dense scores (0..1)

    @staticmethod
    def build(corpus_docs: List[Document], persist_dir="storage", index_dir="storage/faiss_index") -> "HybridRetriever":
        os.makedirs(index_dir, exist_ok=True); os.makedirs(persist_dir, exist_ok=True)
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        faiss_store = FAISS.from_documents(corpus_docs, embeddings)
        faiss_store.save_local(index_dir)
        tokenized = [d.page_content.split() for d in corpus_docs]
        bm25 = BM25Okapi(tokenized)
        with open(os.path.join(persist_dir, "bm25_corpus.pkl"), "wb") as f:
            pickle.dump({"bm25": bm25, "docs": corpus_docs}, f)
        return HybridRetriever(faiss_store, bm25, corpus_docs)

    @staticmethod
    def load(persist_dir="storage", index_dir="storage/faiss_index") -> "HybridRetriever":
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        faiss_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        with open(os.path.join(persist_dir, "bm25_corpus.pkl"), "rb") as f:
            obj: Dict = pickle.load(f)
        return HybridRetriever(faiss_store, obj["bm25"], obj["docs"])

    def _bm25_scores(self, query: str, k: int) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return ranked

    def get_relevant_documents(self, query: str, k: int = 8) -> List[Document]:
        dense = self.faiss_store.similarity_search_with_score(query, k=k)
        bm25_ranked = self._bm25_scores(query, k=k)

        max_d = max((s for _, s in dense), default=1.0) or 1.0
        max_b = max((s for _, s in bm25_ranked), default=1.0) or 1.0

        dense_map = {i: (doc, s/max_d) for i, (doc, s) in enumerate(dense)}
        combo: Dict[int, Dict] = {}

        for i, s in bm25_ranked:
            combo.setdefault(i, {"doc": self.corpus_docs[i], "bm25": 0.0, "dense": 0.0})
            combo[i]["bm25"] = s/max_b

        for i, (doc, nd) in dense_map.items():
            combo.setdefault(i, {"doc": doc, "bm25": 0.0, "dense": 0.0})
            combo[i]["doc"] = doc
            combo[i]["dense"] = nd

        fused = [(v["doc"], self.alpha * v["dense"] + (1 - self.alpha) * v["bm25"]) for v in combo.values()]
        fused_sorted = [d for d, _ in sorted(fused, key=lambda x: x[1], reverse=True)[:k]]
        return fused_sorted
