from pathlib import Path

import faiss
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(data_path: Path, vector_store_path: Path):
    print("Using model: FremyCompany/BioLORD-2023")
    embeddings_func = HuggingFaceEmbeddings(model_name="FremyCompany/BioLORD-2023")
    index = faiss.IndexFlatIP(768)
    vector_store = FAISS(embedding_function=embeddings_func, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    documents = SimpleDirectoryReader(data_path).load_data()
    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)
    nodes = splitter.get_nodes_from_documents(documents)
    documents = [Document(id=node.id_, page_content=node.get_content()) for node in nodes]
    ids = vector_store.add_documents(documents=documents)
    print("Number of documents indexed: ", len(ids))

    vector_store.save_local(vector_store_path)
    print("Vector store saved.")

    return vector_store

if __name__ == "__main__":
    vector_store_path = Path("./vector_store")
    data_path = Path("/home/sracha/proper_kg_project/data/books/text")

    build_vector_store(data_path, vector_store_path)

