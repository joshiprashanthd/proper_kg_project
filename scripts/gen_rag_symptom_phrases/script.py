import sys
sys.path.append("/home/sracha/proper_kg_project")

import pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime


from src.components.usefulness import TripletsUsefulness
from src.components.faiss_rag import FaissRag
from src.components.bm25_rag import BM25Rag
from src.components.symptom_phrases import SymptomPhrasesExtractor
from src.utils import get_completed_batch_numbers, keep_first_definition, merge_jsonl_files
from src.types import TripletLabel

def build_naive_rag_query(docs: list[TripletLabel]):
    DOC_TEMPLATE = """{triplet}
{justification}
{label}
"""
    return "\n".join([DOC_TEMPLATE.format(triplet=doc.triplet, justification=doc.justification, label=doc.label) for doc in docs])

def pipeline(phrase: str, symptom: str, analysis: str):
    query = f"{phrase}\n{symptom}\n{analysis}"
    
    # faiss_triplets_rag_docs = faiss_triplets_rag.run(query)
    faiss_triplets_rag_docs = []
    # bm25_triplets_rag_docs = bm25_triplets_rag.run(query)
    bm25_triplets_rag_docs = []
    
    # usefulness_triplets_docs = triplet_usefulness.run(query, keep_first_definition([t.content for t in bm25_triplets_rag_docs]))
    # usefulness_triplets_docs = [t for t in sorted(usefulness_triplets_docs, key=lambda t: t.score, reverse=True) if t.label == "Useful"]
    usefulness_triplets_docs = []
    # naive_rag_query = f"{query}\n{build_naive_rag_query(usefulness_triplets_docs)}"
    
    naive_rag_query = query

    faiss_rag_docs = faiss_rag.run(naive_rag_query) 
    bm25_rag_docs = bm25_rag.run(naive_rag_query)
    
    return faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = []

    for _, row in batch.iterrows():
        text = row["text"]
        label = row["label"]

        symptom_phrases = symptom_phrase_extractor.run(text)
        sp_rag = []
        
        for sp in symptom_phrases:
            phrase = sp.phrase
            symptom = sp.symptom
            analysis = sp.analysis

            faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs = pipeline(phrase, symptom, analysis)

            sp_rag.append({
                "phrase": phrase,
                "symptom": symptom,
                "analysis": analysis,
                "faiss_rag_docs": [doc.model_dump()  for doc in faiss_rag_docs],
                "bm25_rag_docs": [doc.model_dump() for doc in bm25_rag_docs],
                "usefulness_triplets_docs": [t.model_dump() for t in usefulness_triplets_docs]
            })

        outputs.append({
            "text": text,
            "label": label,
            "symptom_phrases": sp_rag
        })
            
    save_df = pd.DataFrame(outputs)
    save_df.to_json(batch_save_path, orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--random_sample", type=int, default=None, help="Number of random samples to take")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    
    if "text" not in df.columns or 'label' not in df.columns:
        raise ValueError("The csv file does not have 'text' or 'label' column.")
    
    if args.random_sample is not None:
        df = df.sample(n=args.random_sample, random_state=42)
    
    symptom_phrase_extractor = SymptomPhrasesExtractor()
    # faiss_triplets_rag = FaissTripletsRag(top_k=5)
    # bm25_triplets_rag = BM25TripletsRag(top_k=10)
    triplet_usefulness = TripletsUsefulness()
    print('Loading fiass_rag')
    faiss_rag = FaissRag(top_k=3)
    print("Loading bm25_rag")
    bm25_rag = BM25Rag(top_k=3)

    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = Path(f"./outputs/{csv_path.stem}/{timestring}")
    run_folder.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        completed_batches = get_completed_batch_numbers(run_folder.parent)
        for i in range(0, len(df), batch_size):
            if i//batch_size in completed_batches:
                print(f"Skipping batch {i//batch_size}")
                continue
            futures.append(executor.submit(thread_func, df[i:i+batch_size], run_folder / f"batch_{i//batch_size}.jsonl"))
        for future in tqdm(futures):
            future.result()
    
    merge_jsonl_files(run_folder)