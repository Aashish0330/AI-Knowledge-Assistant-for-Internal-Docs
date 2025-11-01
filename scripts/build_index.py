# scripts/build_index.py (verbose)
import os, sys, traceback
print("▶ Starting build_index.py")
print("▪ CWD:", os.getcwd())

def fail(msg: str, code: int = 1):
    print(f"❌ {msg}")
    sys.exit(code)

try:
    print("▪ Checking data dir …")
    if not os.path.isdir("data"):
        os.makedirs("data", exist_ok=True)
        print("  created ./data")

    data_files = []
    for root, _, files in os.walk("data"):
        for f in files:
            data_files.append(os.path.join(root, f))
    print(f"▪ Data files found: {len(data_files)}")
    for p in data_files[:5]:
        print("  -", p)

    from rag.loaders import load_docs
    from rag.splitter import split_docs
    from rag.hybrid_retriever import HybridRetriever

    print("▪ Loading docs …")
    docs = load_docs("data")
    print(f"  Loaded {len(docs)} documents")

    if not docs:
        print("⚠️ No documents found in ./data. Add PDFs/MD/TXT and rerun.")
        sys.exit(0)

    print("▪ Splitting into chunks …")
    chunks = split_docs(docs)
    print(f"  Chunks: {len(chunks)}")

    for i, d in enumerate(chunks):
        d.metadata["id"] = i

    print("▪ Ensuring storage paths exist …")
    os.makedirs("storage/faiss_index", exist_ok=True)

    # embeddings require OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not set in environment. If this fails next, set it or add to .env.")

    print("▪ Building HybridRetriever (FAISS + BM25) … this may take a moment …")
    retriever = HybridRetriever.build(chunks)
    print("✅ Built FAISS + BM25 and saved to ./storage")
    print("   - storage/faiss_index/")
    print("   - storage/bm25_corpus.pkl")

except Exception as e:
    print("‼️ Exception during build:")
    traceback.print_exc()
    fail("Build failed.")
