# build_index_sklearn.py
import os, pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

os.makedirs("storage", exist_ok=True)

# 1) collect text docs
docs = []
for p in Path("data").rglob("*"):
    if p.is_file():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""
        if txt.strip():
            docs.append({"source": str(p), "text": txt})

if not docs:
    print("No docs in ./data — add a .txt/.md and rerun.")
    raise SystemExit(0)

corpus = [d["text"] for d in docs]

# 2) TF-IDF -> dense 384-d via SVD (LSA)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_sparse = tfidf.fit_transform(corpus)
svd = TruncatedSVD(n_components=384, random_state=42)
X = svd.fit_transform(X_sparse).astype("float32")

# 3) normalize to unit length (cosine sim)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

# 4) scikit-learn nearest neighbors (cosine distance)
nn = NearestNeighbors(n_neighbors=10, metric="cosine")
nn.fit(X)

# 5) persist artifacts
with open("storage/nn_index.pkl", "wb") as f:
    pickle.dump({"docs": docs, "tfidf": tfidf, "svd": svd, "nn": nn}, f)

print("✅ Saved storage/nn_index.pkl with docs, tfidf, svd, nn")