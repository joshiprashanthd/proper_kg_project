from .base_triplets_retriever import BaseTripletsRetriever
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

class BM25TripletsRetriever(BaseTripletsRetriever):
    def __init__(self, edges_df: pd.DataFrame):
        super().__init__(edges_df=edges_df)
        tokenized_triplets = self.edges_df[['source', 'target', 'type']].astype(str).agg(' '.join, axis=1).str.split()
        self.bm25 = BM25Okapi(tokenized_triplets)

    def run(self, entities: list[str], top_k=10) -> pd.DataFrame:
        query_tokens = [entity.lower() for entity in entities]
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(-scores)[:top_k]
        top_triplets = self.edges_df.iloc[top_indices].copy()
        top_triplets['score'] = scores[top_indices]
        df = top_triplets.sort_values('score', ascending=False)
        return df
    

