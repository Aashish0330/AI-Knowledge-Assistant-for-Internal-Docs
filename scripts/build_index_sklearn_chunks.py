import os, pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path("data")
STORAGE = Path("storage"); STORAGE.mkdir(exist_ok=True, parents=True)

# ---------- fast, safe readers ----------
TEXT_EXTS = {".txt", ".md", ".log", ".rst", ".csv", ".json", ".yml", ".yaml"}

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

# ---------- FAST chunker (no regex in the hot loop) ----------
def compact_spaces(s: str) -> str:
    # much faster than a regex for large strings
    return " ".join(s.split())

def chunk_text(text: str, size=1000, overlap=150):
    """
    Make ~size-char chunks with ~overlap chars. Prefer to end near a '.' or newline
    using rfind on the local window (no global regex).
    """
    text = compact_spaces(text)
    n = len(text)
    if n == 0:
        return []

    chunks = []
    i = 0
    while i < n:
        end = min(i + size, n)
        window = text[i:end]

        # try to end at a sentence boundary within the window
        cut = window.rfind(". ")
        if cut == -1:
            cut = window.rfind("\n")
        if cut != -1:
            # ensure we don't make a tiny chunk; require at least 60% of target
            if (cut + 1) >= int(size * 0.6):
                end = i + cut + 1

        chunk = text[i:end].strip()
        if chunk:
            chunks.append((i, end, chunk))

        # advance with overlap (but always move forward at least 1 char)
        i_next = end - overlap
        i = i_next if i_next > i else end + 1

    return chunks

# ---------- collect + chunk ----------
records = []
files = [p for p in DATA_DIR.rglob("*") if p.is_file() and (p.suffix.lower() in TEXT_EXTS or p.suffix == "")]
if not files:
    print("No text files in ./data. Add .txt/.md/etc and rerun.")
    raise SystemExit(0)

for idx, p in enumerate(files, 1):
    txt = read_text(p)
    if not txt.strip():
        continue
    chs = chunk_text(txt, size=1200, overlap=200)  # feel free to tune
    for j, (s, e, ch) in enumerate(chs):
        records.append({"source": str(p), "chunk_id": j, "start": s, "end": e, "text": ch})
    if idx % 10 == 0:
        print(f"Processed {idx}/{len(files)} files ... total chunks: {len(records)}")

if not records:
    print("No non-empty chunks produced.")
    raise SystemExit(0)

corpus = [r["text"] for r in records]

# ---------- TF-IDF (char n-grams = robust to plurals/typos) ----------
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=300_000,
    lowercase=True,
    min_df=1,
)
X_sparse = tfidf.fit_transform(corpus)
n_features = X_sparse.shape[1]

payload = {"records": records, "tfidf": tfidf, "use_svd": False}

# ---------- optional SVD ----------
if n_features >= 512:
    target = min(768, max(256, n_features - 1))
    svd = TruncatedSVD(n_components=target, random_state=42)
    X = svd.fit_transform(X_sparse).astype("float32")
    payload.update({"svd": svd, "use_svd": True})
else:
    X = X_sparse.toarray().astype("float32")

# ---------- normalize + NN ----------
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
nn = NearestNeighbors(n_neighbors=10, metric="cosine")
nn.fit(X)
payload["nn"] = nn

with open(STORAGE / "nn_chunks.pkl", "wb") as f:
    pickle.dump(payload, f)

print(f"✅ Saved storage/nn_chunks.pkl | files={len(files)} | chunks={len(records)} "
      f"| features={n_features} | dim={X.shape[1]} | use_svd={payload['use_svd']}")
