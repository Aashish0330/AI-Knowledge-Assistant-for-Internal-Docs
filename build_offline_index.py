import os, pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
os.makedirs("storage/faiss_index", exist_ok=True)

docs = []
for p in Path("data").rglob("*"):
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        if text.strip():
            docs.append({"source": str(p), "text": text})

if not docs:
    print("No docs in ./data — add a .txt/.md/.pdf (extracted text) and rerun.")
    raise SystemExit(0)

# 2) simple chunks (one per file, capped to 2k chars)
corpus = [d["text"][:2000] for d in docs]

# 3) local embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)

# 4) FAISS index (cosine via inner product on normalized vectors)
dim = X.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X)

# 5) persist
faiss.write_index(index, "storage/faiss_index/index.faiss")

with open("storage/bm25_corpus.pkl", "wb") as f:
    pickle.dump({"docs": docs, "bm25": None}, f)

print("✅ Offline index created:")
print(" - storage/faiss_index/index.faiss")
print(" - storage/bm25_corpus.pkl")
