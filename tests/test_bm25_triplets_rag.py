import sys
sys.path.append("/home/sracha/proper_kg_project/")

from pprint import pprint
import timeit
from src.components.bm25_triplets_rag import BM25TripletsRag

vector_store_path = "/home/sracha/proper_kg_project/src/components/retrieval/triplets/bm25_retriever/vector_store"
mock_queries = [
    "What are the symptoms of major depressive disorder in someone with a history of substance abuse and a family history of bipolar disorder?",
    "How does childhood trauma impact the development of borderline personality disorder in those with a family history of schizophrenia?",
    "What are the treatment options for obsessive compulsive disorder in someone with a history of anxiety attacks and a family history of depression?",
    "Can someone with a history of anorexia nervosa and a family history of depression develop bulimia nervosa?",
    "What are the signs and symptoms of post traumatic stress disorder in someone with a history of anxiety attacks and a family history of depression?",
]


def run_time_analysis():
    start = timeit.default_timer()
    medrag = BM25TripletsRag(top_k=5)
    end = timeit.default_timer()
    print(f"load_retriever took {end-start} seconds")

    times = []

    for query in mock_queries:
        start = timeit.default_timer()
        documents = medrag.run(query)
        end = timeit.default_timer()
        times.append(end-start)
        print(f"Retrieving {query} took {end-start} seconds")

    avg_time = sum(times) / len(times)
    print(f"Average time taken for retrieving: {avg_time}")


def inference():
    medrag = BM25TripletsRag(vector_store_path)
    result = medrag.run(mock_queries[0])
    pprint(result)

if __name__ == "__main__":
    run_time_analysis()
    # inference()

# Loading took: 155.07830915800878 seconds ~ 2 min 35 sec
# Average time taken for retrieving: 0.21529952459677587 seconds ~ 215 ms
