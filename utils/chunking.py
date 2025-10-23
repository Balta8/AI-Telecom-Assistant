# utils/chunking.py

from typing import List, Dict, Any
from langchain.schema import Document

# Chunker that groups records by a specified number of rows

class NRowsChunker:

    # Initialize with number of rows per chunk

    def __init__(self, n: int = 5):
        self.n = n

    def chunk(self, records: List[Dict[str, Any]]) -> List[Document]:
        docs = []

        for i in range(0, len(records), self.n):
            group = records[i:i + self.n]

            # Combine content of the group into a single string

            page_content = "\n".join(
                f"{r.get('title','')} — {r.get('content','')} — {r.get('price', '')}" 
                for r in group
            )

            # Metadata 

            metadata = {
                "id": str(list(range(i, i + len(group)))),  
                "type": group[0].get("type", ""),  
                "category": group[0].get("category", ""),
                "tags": group[0].get("tags", ""),
                "title": group[0].get("title", ""),
                "price": group[0].get("price", ""),
                "chunk_index": i // self.n,
            }

            docs.append(Document(page_content=page_content, metadata=metadata))

        return docs
