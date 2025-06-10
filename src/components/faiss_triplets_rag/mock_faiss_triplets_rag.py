from pathlib import Path
from src.utils import logger
from src.types import TripletLabel

class MockFaissTripletsRag:
    def __init__(self, vector_store_path: Path):
        logger.info("Loading MockFaissTripletsRag")
        pass

    def run(self, query: str, top_k=40) -> list[TripletLabel]:
        logger.info(f"Retrieving {top_k} triplets for query: {query}")
        results = []
        for _ in range(top_k):
            results.append(TripletLabel(triplet="(head: 'nothing')--[creates]->(tail: 'everything')", label="Useful", justification="Mock", score=0.5))
        return results
        