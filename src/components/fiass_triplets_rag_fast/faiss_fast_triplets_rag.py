from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.types import RagDocument
from pathlib import Path
from src.utils import logger

class FaissFastTripletsRag:
    def __init__(self, top_k: int, vector_store_path: Path = Path("/home/sracha/proper_kg_project/src/components/fiass_triplets_rag_fast/vector_store")):
        logger.info(f"Loading FAISS vector store from {vector_store_path}")
        self.embeddings_func = HuggingFaceEmbeddings(model_name="FremyCompany/BioLORD-2023", model_kwargs={"device": "cuda"})
        self.vector_store = FAISS.load_local(str(vector_store_path), embeddings=self.embeddings_func, allow_dangerous_deserialization=True)
        self.top_k = top_k
    
    def run(self, query: str) -> list[RagDocument]:
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        docs = [RagDocument(content=doc.page_content, score=-float(score), source="FaissTripletsRag", type="triplet") for (doc, score) in results]
        docs = sorted(docs, key=lambda x: x.score, reverse=True)[:self.top_k]
        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        return docs
        