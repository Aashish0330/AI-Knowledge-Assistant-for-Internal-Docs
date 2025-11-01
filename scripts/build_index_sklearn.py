import os, pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

os.makedirs("storage", exist_ok=True)

# --- add near top (after imports) ---
ORG_PAT = re.compile(r"^\s*#\s*([A-Za-z0-9][A-Za-z0-9 &_-]{1,80})\b.*", re.M)
ORG_ALT = re.compile(r"^\s*([A-Za-z][A-Za-z &_-]{1,80})\s*[-–—]\s*", re.M)

def infer_org(text: str, fallback_path: str) -> str:
    m = ORG_PAT.search(text)
    if m:
        return m.group(1).strip()
    m = ORG_ALT.search(text)
    if m:
        return m.group(1).strip()
    # fallback: use filename stem (e.g., blueorbit_it_security -> Blueorbit)
    stem = Path(fallback_path).stem.replace("_", " ").strip()
    return stem[:1].upper() + stem[1:]

# collect docs
docs = []
for p in Path("data").rglob("*"):
    if p.is_file():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""
        if txt.strip():
            org = infer_org(txt, str(p))
            docs.append({
                "source": str(p),
                "org": org,
                "text": txt.strip()
            })

if not docs:
    print("No docs in ./data — add a .txt/.md and rerun.")
    raise SystemExit(0)

corpus=[d["text"] for d in docs]

# TF-IDF
tfidf=TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_sparse=tfidf.fit_transform(corpus)
n_features = X_sparse.shape[1]

payload = {"docs": docs, "tfidf": tfidf, "use_svd": False}

# Decide embedding space
if n_features >= 64:
    # adaptive SVD size: up to 384, but <= n_features-1
    target = min(384, max(64, n_features - 1))
    svd=TruncatedSVD(n_components=target, random_state=42)
    X=svd.fit_transform(X_sparse).astype("float32")
    payload.update({"svd": svd, "use_svd": True})
else:
    # small corpus: just use TF-IDF dense
    X=X_sparse.toarray().astype("float32")

# normalize (cosine)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

# NearestNeighbors (cosine distance)
nn=NearestNeighbors(n_neighbors=min(10, len(docs)), metric="cosine")
nn.fit(X)
payload["nn"] = nn

with open("storage/nn_index.pkl","wb") as f:
    pickle.dump(payload, f)

print(f"✅ Saved storage/nn_index.pkl | docs={len(docs)} | features={n_features} | dim={X.shape[1]} | use_svd={payload['use_svd']}")
