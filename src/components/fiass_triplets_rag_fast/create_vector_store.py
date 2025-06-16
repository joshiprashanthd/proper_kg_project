from pathlib import Path
import pandas as pd

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# PAGE_CONTENT_TEMPLATE = """
# Head: {head}
# Tail: {tail}
# Relation: {relation}
# Head Definition: {head_definition}
# Tail Definition: {tail_definition}
# Triple: ({head}, {head_definition})-[{relation}]->({tail}, {tail_definition})
# """

MENTAL_HEALTH_SABS = [
    "DSM-5",
    "ICD10",       # Includes mental & behavioral disorders (F00–F99)
    "ICD10CM",
    "ICD10AM",
    "ICD9CM",
    "SNOMEDCT_US",
    "SNOMEDCT_VET",  # US edition of SNOMED CT
    "ICPC2P",      # ICPC‑2 PLUS
    "ICPC2EENG",   # ICPC‑2 Electronic English
    "ICF",
    "ICF-CY",
    "PSY",
    "RXNORM",      # Drug terminology—psychotropics included
    "MDR",         # MedDRA for adverse event reporting
    "CPT",         # Procedures including psychiatric assessments
    "HCPT",        # HCPCS version of CPT
    "CCS",         # Clinical Classifications Software
    "ALT"          # Alternative billing, useful for mental health service codes
]

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

def build_vector_store(triplet_path: Path, definition_path: Path, vector_store_path: Path):
    embeddings_func = HuggingFaceEmbeddings(model_name="FremyCompany/BioLORD-2023", model_kwargs={"device": "cuda"})

    print("Building vector store...")

    index = faiss.IndexHNSWFlat(768, 32)
    vector_store = FAISS(embedding_function=embeddings_func, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

    triplets = pd.read_parquet(triplet_path, engine="pyarrow")
    definitions = pd.read_parquet(definition_path, engine="pyarrow")

    triplets = triplets[triplets['SAB'].isin(MENTAL_HEALTH_SABS)]

    cui_to_def_map = definitions.groupby('CUI')['DEF'].apply(list).to_dict()

    batch_size = 5000
    for i in tqdm(range(0, len(triplets), batch_size), desc="Processing triplets"):
        batch = triplets.iloc[i:i+batch_size]

        head_definitions = batch['CUI1'].map(cui_to_def_map).fillna('<|NO_DEFINITION|>')
        tail_definitions = batch['CUI2'].map(cui_to_def_map).fillna('<|NO_DEFINITION|>')

        formatted_head_defs = head_definitions.apply(lambda x: "\n".join(f"{j+1}. {defn}" for j, defn in enumerate(x)) if isinstance(x, list) else x)
        formatted_tail_defs = tail_definitions.apply(lambda x: "\n".join(f"{j+1}. {defn}" for j, defn in enumerate(x)) if isinstance(x, list) else x)

        page_content_list = [
            PAGE_CONTENT_COMPACT.format(
                head=row['NAME1'],
                head_definition=formatted_head_defs.iloc[j],
                relation=row['RELATION'],
                tail=row['NAME2'],
                tail_definition=formatted_tail_defs.iloc[j]
            )
            for j, (_, row) in enumerate(batch.iterrows())
        ]

        documents_to_add = [Document(page_content=content) for content in page_content_list]
        vector_store.add_documents(documents_to_add, ids=[str(i+j) for j in range(len(documents_to_add))])

    return vector_store

if __name__ == "__main__":
    triplets_path = Path("/home/sracha/proper_kg_project/data/triplets/triplets.parquet")
    definitions_path = Path("/home/sracha/proper_kg_project/data/triplets/definitions.parquet")
    vector_store_path = Path("./vector_store")

    vector_store = build_vector_store(triplets_path, definitions_path, vector_store_path)
    vector_store.save_local("vector_store")