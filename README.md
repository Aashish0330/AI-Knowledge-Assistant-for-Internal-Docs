# AI Knowledge Assistant (Local RAG)

A **local Retrieval-Augmented Generation (RAG)** system that lets you query your organization’s internal policies, procedures, and documents **entirely offline** — without any OpenAI or cloud dependency.

Built using **Python, scikit-learn, FAISS/NearestNeighbors, and Streamlit**, it indexes your Markdown, TXT, and PDF files into searchable semantic vectors and retrieves the most relevant text snippets when you ask a question.

---


## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AI-Knowledge-Assistant-for-Internal-Docs.git
cd AI-Knowledge-Assistant-for-Internal-Docs

2. Create and activate a conda or virtual environment

conda create -n rag-local python=3.11 -y
conda activate rag-local

3. Install dependencies

pip install -r requirements.txt

If you face NumPy/FAISS compatibility issues on Apple Silicon, use:

conda install -c conda-forge numpy==1.26.4 faiss-cpu scikit-learn==1.4.2 pandas==2.2.2


⸻

🗂️ Add Your Documents

Place all your .txt, .md, or .pdf files in the data/ folder.
Example:

data/
├── password_policy.txt
├── security_guidelines.md
└── blueorbit_security.pdf

Each file should contain human-readable text (no binary or encrypted PDFs).

⸻

🔧 Build the Index

Run one of the following to create your search index:

Full document-level index:

python scripts/build_offline_index_tfidf.py

Chunked index (recommended for large files):

python scripts/build_index_sklearn_chunks.py

This generates:

storage/
├── nn_index.pkl         # Full-document index
└── nn_chunks.pkl        # Chunked index (preferred)


⸻

💬 Run the Streamlit App

Launch the local UI:

streamlit run app_streamlit.py

Then open the app in your browser (usually http://localhost:8501￼).

⸻

🕵️ Query Examples

Once the index is loaded, you can ask natural-language questions like:
	•	“What is our password policy?”
	•	“How often should VPN credentials be changed?”
	•	“Who manages access keys?”
	•	“What is the clean desk policy?”
	•	“Summarize our data protection policy.”

The assistant will retrieve the most relevant snippets from your documents and show the answer with source file names.

⸻

🧠 How It Works
	1.	Text Extraction: All files in data/ are read (PDFs parsed with PyMuPDF).
	2.	Chunking: Large documents are split into overlapping text chunks for granular retrieval.
	3.	Vectorization: TF-IDF transforms each chunk into a sparse vector.
	4.	(Optional) SVD compresses the TF-IDF matrix for faster search.
	5.	NearestNeighbors Search: Queries are embedded using the same TF-IDF model and compared against the stored vectors to find semantically similar chunks.
	6.	Context Display: The top results and their source files are displayed in Streamlit.

⸻

🧩 Troubleshooting

Issue	Fix
Streamlit app hangs at “Solving environment”	Use conda-forge channel for installs
faiss import fails	Try conda install -c conda-forge faiss-cpu
App shows “Index is empty”	Run python scripts/build_index_sklearn_chunks.py again
Mac M3 / ARM issues	Ensure numpy<2.0 and rebuild index


⸻

🧭 Roadmap
	•	Add LLM-based summarization via local Ollama or OpenAI API
	•	Implement question-answer reasoning beyond keyword retrieval
	•	Add document re-ranking using embeddings (SentenceTransformers)
	•	Support for Docx / HTML / CSV ingestion

⸻

📄 License

MIT License © 2025

⸻

🌟 Acknowledgements

This project uses open-source components from:
	•	Streamlit￼
	•	scikit-learn￼
	•	FAISS￼
	•	PyMuPDF￼
	•	LangChain community examples￼

---
