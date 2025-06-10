from pathlib import Path
from langchain_community.retrievers import TFIDFRetriever
from src.types import RagDocument

class TFIDFRag:
    def __init__(self, vectorizer_path: Path = Path("/home/sracha/proper_kg_project/src/components/tfidf_rag/tfidf_vector_store")):
        self.vectorizer = TFIDFRetriever.load_local(vectorizer_path, allow_dangerous_deserialization=True)
    
    def run(self, query: str, top_k: int = 5) -> list[RagDocument]:
        self.vectorizer.k = top_k
        results = self.vectorizer.invoke(query)
        return [RagDocument(content=doc.page_content, score=doc.metadata.get("score", 0.0), source="TFIDFRag", type="text") for doc in results]
        