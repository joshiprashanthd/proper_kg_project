import sys
sys.path.append("/home/sracha/proper_kg_project")

import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tqdm import tqdm
import argparse

from src.components.med_rag import MedRag
from src.components.retrieval.triplets.vector_retriever import VectorTripletsRetriever

TRIPLETS_SPLITTER = "<|triplet|>"
MED_DOC_SPLITTER = "<|med_doc|>"

def get_completed_batch_numbers(outputs_folder):
    completed_batches = set()
    for file in outputs_folder.rglob("*.csv"):
        if file.name.startswith("batch_") and file.name.endswith(".csv"):
            batch_number = int(file.stem.split("_")[1])
            completed_batches.add(batch_number)
    return completed_batches

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = {
        "triplets": [],
        "med_docs": []
    }
    
    for i, row in batch.iterrows():
        question = row['question']
        answer = row['answer']
        options = [row[f'option{i}'] for i in range(1, 5) if not pd.isna(row[f'option{i}'])] 

        query = f"{question}\n{answer}\n{options}"
        
        med_rag_documents = med_rag.extract_documents(query, 20)
        triplets_documents = triplets_retriever.extract_triplets(query, 20)

        outputs["triplets"].append(TRIPLETS_SPLITTER.join([doc.triplet for doc in triplets_documents]))
        outputs["med_docs"].append(MED_DOC_SPLITTER.join([doc.content for doc in med_rag_documents]))
    
    batch.loc[:, "retrieved_triplets"] = outputs["triplets"]
    batch.loc[:, "retrieved_documents"] = outputs["med_docs"]

    batch.to_csv(batch_save_path, index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explanations for a dataset.")
    parser.add_argument("csv_path", type=str, help="Path to the dataset")
    
    args = parser.parse_args()
    
    med_rag = MedRag(Path("/home/sracha/proper_kg_project/src/components/med_rag/vector_store"))
    triplets_retriever = VectorTripletsRetriever(Path("/home/sracha/proper_kg_project/src/components/retrieval/triplets/vector_retriever/vector_store"))
    
    df = pd.read_csv(args.csv_path)
    df = df.sample(30, random_state=42)

    outputs_folder = Path(f"./outputs/retrieve_paragraphs_triplets/{Path(args.csv_path).stem}")
    outputs_folder.mkdir(parents=True, exist_ok=True)

    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = outputs_folder / f"run_{timestring}"
    run_folder.mkdir(parents=True, exist_ok=True)

    batch_size = 10
    completed_batches = get_completed_batch_numbers(outputs_folder)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_number = i // batch_size
            if batch_number in completed_batches:
                print(f"Batch {batch_number} already completed. Skipping.")
                continue
            futures.append(executor.submit(thread_func, df[i:i+batch_size], run_folder / f"batch_{batch_number}.csv"))
        for future in tqdm(futures):
            future.result()
    
    
        