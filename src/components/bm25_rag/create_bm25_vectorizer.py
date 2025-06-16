from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever

if __name__ == "__main__":
    documents = SimpleDirectoryReader("/home/sracha/proper_kg_project/data/books/text").load_data()
    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)
    nodes = splitter.get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=20,
    )

    bm25_retriever.persist("./vector_store")