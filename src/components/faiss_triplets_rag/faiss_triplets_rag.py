from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.types import RagDocument
from pathlib import Path
from src.utils import logger

class FaissTripletsRag:
    def __init__(self, vector_store_path: Path = Path("/home/sracha/proper_kg_project/src/components/faiss_triplets_rag/vector_store")):
        logger.info(f"Loading FAISS vector store from {vector_store_path}")
        self.embeddings_func = HuggingFaceEmbeddings(model_name="FremyCompany/BioLORD-2023", model_kwargs={"device": "cuda"})
        self.vector_store = FAISS.load_local(vector_store_path, embeddings=self.embeddings_func, allow_dangerous_deserialization=True)

    def run(self, query: str, top_k=40) -> list[RagDocument]:
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        docs = [RagDocument(content=doc.page_content, score=-float(score), source="FaissTripletsRag", type="triplet") for (doc, score) in results]
        docs = sorted(docs, key=lambda x: x.score, reverse=True)[:top_k]
        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        return docs
        