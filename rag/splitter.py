from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = make_splitter()
    return splitter.split_documents(docs)
