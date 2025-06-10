from pathlib import Path

import faiss
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(data_path: Path, vector_store_path: Path):
    print("Using model: FremyCompany/BioLORD-2023")
    embeddings_func = HuggingFaceEmbeddings(model_name="FremyCompany/BioLORD-2023")

    index = faiss.IndexFlatIP(768)
    vector_store = FAISS(embedding_function=embeddings_func, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print("Number of documents loaded: ", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    all_splits = text_splitter.split_documents(documents)
    print("Number of splits: ", len(all_splits))

    ids = vector_store.add_documents(documents=all_splits)
    print("Number of documents indexed: ", len(ids))

    vector_store.save_local(vector_store_path)
    print("Vector store saved.")

    return vector_store

if __name__ == "__main__":
    vector_store_path = Path("./vector_store")
    data_path = Path("/home/sracha/proper_kg_project/data/books/text")

    build_vector_store(data_path, vector_store_path)

