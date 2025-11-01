import os, pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss

os.makedirs("storage/faiss_index", exist_ok=True)

# Collect docs
docs=[]
for p in Path("data").rglob("*"):
    if p.is_file():
        try:
            txt=p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt=""
        if txt.strip():
            docs.append({"source": str(p), "text": txt})

if not docs:
    print("No docs in ./data — add a .txt/.md and rerun."); raise SystemExit(0)

corpus=[d["text"] for d in docs]

# TF-IDF -> dense 384-d via SVD
tfidf=TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_sparse=tfidf.fit_transform(corpus)
svd=TruncatedSVD(n_components=384, random_state=42)
X=svd.fit_transform(X_sparse).astype("float32")
X/= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

# Save FAISS index (cosine via inner product on normalized vectors)
index=faiss.IndexFlatIP(X.shape[1]); index.add(X)
faiss.write_index(index, "storage/faiss_index/index.faiss")

# Save artifacts
with open("storage/bm25_corpus.pkl","wb") as f:
    pickle.dump({"docs": docs, "tfidf": tfidf, "svd": svd}, f)

print("✅ Index ready: storage/faiss_index/index.faiss, storage/bm25_corpus.pkl")
