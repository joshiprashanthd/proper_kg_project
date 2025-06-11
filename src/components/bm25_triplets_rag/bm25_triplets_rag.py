from pathlib import Path
from llama_index.retrievers.bm25 import BM25Retriever
from src.types import RagDocument
from src.utils import logger

class BM25TripletsRag:
    def __init__(self, top_k: int, vector_store_path: Path = Path("/home/sracha/proper_kg_project/src/components/bm25_triplets_rag/vector_store")):
        logger.info(f"Loading BM25 vectorizer from {vector_store_path}")
        self.retriever = BM25Retriever.from_persist_dir(str(vector_store_path))
        self.top_k = top_k
    
    def run(self, entities: list[str]) -> list[RagDocument]:
        nodes_with_scores = self.retriever.retrieve(entities)
        docs = [RagDocument(content=node.get_content(), score=node.get_score(), source="BM25TripletsRag", type="triplet") for node in nodes_with_scores]
        docs = sorted(docs, key=lambda x: x.score, reverse=True)[:self.top_k]
        logger.info(f"Retrieved {len(docs)} documents for query: {entities}")
        return docs
    

