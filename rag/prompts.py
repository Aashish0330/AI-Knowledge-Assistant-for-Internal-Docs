from langchain.prompts import ChatPromptTemplate

SYSTEM = """You are an internal knowledge assistant.
- Answer ONLY with information grounded in the provided context.
- If the answer is not in the context, say you don't know.
- Include a short 'Sources' list with source & page numbers when possible."""

USER = """Question: {question}

Context:
{context}

Return JSON exactly in this shape:
{{
  "answer_markdown": "...",
  "citations": [{{"source": "file_or_url", "page": "num"}}]
}}"""

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", USER)])
