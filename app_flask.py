from flask import Flask, request, jsonify
from rag.hybrid_retriever import HybridRetriever
from rag.re_rank import CrossEncoderReranker
from rag.chain import make_chain

app = Flask(__name__)
retriever = HybridRetriever.load()
chain = make_chain()
reranker = CrossEncoderReranker()

@app.post("/ask")
def ask():
    q = request.json.get("query", "")
    out = chain(q, retrieve_fn=retriever.get_relevant_documents, reranker=reranker, k=8)
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
