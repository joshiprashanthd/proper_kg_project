from pathlib import Path
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_vectorizer(data_path: Path, pkl_path: Path):
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print("Number of documents loaded: ", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    print("Splitting documents...")
    all_splits = text_splitter.split_documents(documents)
    print("Number of splits: ", len(all_splits))

    print("Indexing documents...")
    retriever = TFIDFRetriever.from_documents(all_splits)

    print("Saving vectorizer...")
    retriever.save_local(pkl_path)
    print("Vectorizer saved.")

    return retriever


if __name__ == "__main__":
    data_path = Path("../data")
    pkl_path = Path("./vector_store")
    build_vectorizer(data_path, pkl_path)