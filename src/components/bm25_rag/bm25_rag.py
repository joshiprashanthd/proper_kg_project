from pathlib import Path
from llama_index.retrievers.bm25 import BM25Retriever
from src.types import RagDocument
from src.utils import logger

class BM25Rag:
    def __init__(self, top_k: int, vectorizer_path: Path = Path("/home/sracha/proper_kg_project/src/components/bm25_rag/bm25_vector_store")):
        logger.info(f"Loading BM25 vectorizer from {vectorizer_path}")
        self.vectorizer = BM25Retriever.from_persist_dir(vectorizer_path)
        self.top_k = top_k
    
    def run(self, query: str) -> list[RagDocument]:
        nodes_with_scores = self.vectorizer.retrieve(query)
        docs = [RagDocument(content=node.get_content(), score=node.get_score(), source="BM25Rag", type="text") for node in nodes_with_scores]
        docs = sorted(docs, key=lambda x: x.score, reverse=True)[:self.top_k]
        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        return docs
        