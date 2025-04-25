from more_itertools import flatten
import pandas as pd

class BaseTripletsRetriever:
    def __init__(self, edges_df: pd.DataFrame, source_col: str = 'source', target_col: str = 'target', type_col: str = 'type'):
        assert source_col in edges_df.columns, "edges_df must contain a 'source' column"
        assert target_col in edges_df.columns, "edges_df must contain a 'target' column"
        assert type_col in edges_df.columns, "edges_df must contain a 'type' column"

        self.edges_df = edges_df

    def run(self, entities: list[str], top_k=20) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")

    def run_mult(self, entities_list: list[list[str]], top_k=20) -> pd.DataFrame:
        results = []
        for entities in entities_list:
            result = self.run(entities, top_k=top_k)
            results.append(result)
        return flatten(list(set(results)))