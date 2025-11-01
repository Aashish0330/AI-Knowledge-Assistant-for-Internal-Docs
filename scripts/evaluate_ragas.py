from datasets import load_dataset
from rag.hybrid_retriever import HybridRetriever
from rag.re_rank import CrossEncoderReranker
from rag.chain import make_chain
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate
import pandas as pd

# minimal eval set — replace with your own
# file: storage/eval_set.jsonl (question, ground_truth, contexts[])
# create a sample if missing
import os, json
os.makedirs("storage", exist_ok=True)
eval_path = "storage/eval_set.jsonl"
if not os.path.exists(eval_path):
    with open(eval_path, "w") as f:
        f.write(json.dumps({
            "question":"What is our password rotation policy?",
            "ground_truth":"Passwords must be rotated every 90 days for privileged accounts per Policy SEC-12.",
            "contexts":[]
        })+"\n")

retriever = HybridRetriever.load()
reranker = CrossEncoderReranker()
chain = make_chain()

ds = load_dataset("json", data_files=eval_path, split="train")

def predict(example):
    out = chain(example["question"], retrieve_fn=retriever.get_relevant_documents, reranker=reranker, k=8)
    example["answer"] = out["answer_markdown"]
    example["contexts"] = [c["source"] for c in out["citations"]][:8]
    return example

pred = ds.map(predict)
result = evaluate(pred, metrics=[answer_relevancy, faithfulness, context_precision, context_recall])
print(result)
df = result.to_pandas()
df.to_csv("storage/ragas_scores.csv", index=False)
print("✅ Saved RAGAS scores to storage/ragas_scores.csv")
