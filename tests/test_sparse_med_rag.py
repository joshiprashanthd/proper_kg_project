import sys
sys.path.append("/home/sracha/proper_kg_project")

from pathlib import Path

from src.components.med_rag.sparse_retrieval.sparse_med_rag import SparseMedRag

if __name__ == "__main__":
    sparse_med_rag = SparseMedRag(vectorizer_path=Path("/home/sracha/proper_kg_project/src/components/med_rag/sparse_retrieval/vector_store"))
    documents = sparse_med_rag.extract_documents("depression", top_k=20)
    