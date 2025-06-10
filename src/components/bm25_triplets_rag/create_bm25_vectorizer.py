from pathlib import Path
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

# PAGE_CONTENT_TEMPLATE = """
# Head: {head}
# Tail: {tail}
# Relation: {relation}
# Head Definition: {head_definition}
# Tail Definition: {tail_definition}
# Triple: ({head}, {head_definition})-[{relation}]->({tail}, {tail_definition})
# """

PAGE_CONTENT_COMPACT = "(Head: '{head}', Definition: '{head_definition}')-[{relation}]->(Tail: '{tail}', Definition: '{tail_definition}')"

def get_definition(cui: str, df: pd.DataFrame) -> str:
    return df[df['CUI'] == cui]['DEF'].values.tolist() 

def batch_apply_func(row: pd.Series, definitions: pd.DataFrame) -> str:
    head_definition = get_definition(row['NAME1'], definitions)
    tail_definition = get_definition(row['NAME2'], definitions)

    if len(head_definition) > 0 and len(tail_definition) > 0:
        head_definition = "\n".join(f"{i+1}. {defn}" for i, defn in enumerate(head_definition))
        tail_definition = "\n".join(f"{i+1}. {defn}" for i, defn in enumerate(tail_definition))
    else:
        head_definition = "<|NO_DEFINITION|>"
        tail_definition = "<|NO_DEFINITION|>"
        
    return PAGE_CONTENT_COMPACT.format(head=row['NAME1'], head_definition=head_definition, relation=row['RELATION'], tail=row['NAME2'], tail_definition=tail_definition)

def process_batch(batch, cui_to_def_map, batch_idx, batch_size):
    # Pre-process definitions
    head_defs = batch['CUI1'].map(cui_to_def_map).fillna('<|NO_DEFINITION|>')
    tail_defs = batch['CUI2'].map(cui_to_def_map).fillna('<|NO_DEFINITION|>')
    
    # Format definitions using vectorized operations
    def format_defs(defs):
        return np.where(
            defs.apply(lambda x: isinstance(x, list)),
            defs.apply(lambda x: "\n".join(f"{j+1}. {d}" for j, d in enumerate(x)) if isinstance(x, list) else x),
            defs
        )
    
    formatted_head_defs = format_defs(head_defs)
    formatted_tail_defs = format_defs(tail_defs)

    documents = [
        TextNode(text=PAGE_CONTENT_COMPACT.format(
            head=row['NAME1'],
            head_definition=head_def,
            relation=row['RELATION'],
            tail=row['NAME2'],
            tail_definition=tail_def
        ))
        for (_, row), head_def, tail_def in zip(
            batch.iterrows(), 
            formatted_head_defs, 
            formatted_tail_defs
        )
    ]
    
    return documents

def build_vector_store(triplet_path: Path, definition_path: Path):
    batch_size = 10000
    triplets = pd.read_parquet(triplet_path, engine="pyarrow")
    definitions = pd.read_parquet(definition_path, engine="pyarrow")
    cui_to_def_map = definitions.groupby('CUI')['DEF'].apply(list).to_dict()
    
    all_documents = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(triplets), batch_size):
            batch = triplets.iloc[i:i+batch_size]
            futures.append(executor.submit(
                process_batch, 
                batch, 
                cui_to_def_map,
                i,
                batch_size
            ))
        
        for future in tqdm(futures, desc="Processing batches"):
            all_documents.extend(future.result())
    
    vector_store = BM25Retriever.from_defaults(nodes=all_documents, similarity_top_k=20)
    return vector_store

if __name__ == "__main__":
    triplets_path = Path("../data/triplets.parquet")
    definitions_path = Path("../data/definitions.parquet")
    retriever = build_vector_store(triplets_path, definitions_path)
    retriever.persist("./vector_store")