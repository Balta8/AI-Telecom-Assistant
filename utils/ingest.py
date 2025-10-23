# utils/ingest.py

from typing import List 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from config import OPENAI_API_KEY, EMBEDDING_MODEL

class ChromaIngestor:

    def __init__(self, chroma_dir: str ="./chroma_store", embedding_model: str = EMBEDDING_MODEL):
        self.chroma_dir = chroma_dir
        self.embedding_model = embedding_model
        self.vectorstore = None

    # Clean text by removing "nan" and trimming whitespace

    @staticmethod
    def clean_text(text: str) -> str:
        if not text or text.lower() == "nan":
            return ""
        return str(text).replace("nan", "").strip()

    def ingest(self, docs: List[Document]):

        # Clean docs قبل ingestion

        cleaned_docs = [
            Document(
                page_content=self.clean_text(doc.page_content),
                metadata=doc.metadata
            )
            for doc in docs
            if doc.page_content and self.clean_text(doc.page_content) != ""
        ]
       
       # Initialize Chroma vectorstore مع cleaned documents

        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=self.embedding_model)
        self.vectorstore = Chroma.from_documents(
            documents=cleaned_docs,
            embedding=embeddings,
            persist_directory=self.chroma_dir
        )
        # Chroma automatically persists, no need for manual persist()
    
        # Getter for vectorstore

    def get_vectorstore(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized. Please run ingest() first.") # ensure ingest called
        return self.vectorstore
