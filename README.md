# AI Knowledge Assistant (Local RAG)

A **local Retrieval-Augmented Generation (RAG)** system that lets you query your organization’s internal policies, procedures, and documents **entirely offline** — without any OpenAI or cloud dependency.

Built using **Python, scikit-learn, FAISS/NearestNeighbors, and Streamlit**, it indexes your Markdown, TXT, and PDF files into searchable semantic vectors and retrieves the most relevant text snippets when you ask a question.

---


## Installation

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

Add Your Documents

Place all your .txt, .md, or .pdf files in the data/ folder.
Example:

data/
├── password_policy.txt
├── security_guidelines.md
└── blueorbit_security.pdf

Each file should contain human-readable text (no binary or encrypted PDFs).

⸻


⸻

Run the Streamlit App

Launch the local UI:

streamlit run app_streamlit.py

Then open the app in your browser (usually http://localhost:8501￼).
⸻

License

MIT License © 2025

⸻

Acknowledgements

This project uses open-source components from:
	•	Streamlit￼
	•	scikit-learn￼
	•	FAISS￼
	•	PyMuPDF￼
	•	LangChain community examples￼

---
