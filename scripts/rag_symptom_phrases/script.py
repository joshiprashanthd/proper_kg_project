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
from src.components.faiss_triplets_rag import FaissTripletsRag
from src.components.bm25_triplets_rag import BM25TripletsRag
from src.components.symptom_phrases import SymptomPhrasesExtractor
from src.types import TripletLabel, SymptomPhrase

from src.utils import get_completed_batch_numbers, keep_first_definition, merge_jsonl_files

def build_naive_rag_query(docs: list[TripletLabel]):
    DOC_TEMPLATE = """{triplet}
{justification}
{label}
"""
    return "\n".join([DOC_TEMPLATE.format(triplet=doc.triplet, justification=doc.justification, label=doc.label) for doc in docs])

def pipeline(phrase: str, symptom: str, analysis: str, useful_triplets_top_k=10, faiss_triplets_top_k = 10, bm25_triplets_top_k = 10, faiss_rag_top_k = 5, bm25_rag_top_k = 5):
    query = f"{phrase}\n{symptom}\n{analysis}"
    
    faiss_triplets_rag_docs = faiss_triplets_rag.run(query, top_k=faiss_triplets_top_k)
    bm25_triplets_rag_docs = bm25_triplets_rag.run(query, top_k=bm25_triplets_top_k)
    
    usefulness_triplets_docs = triplet_usefulness.run(query, keep_first_definition([t.content for t in faiss_triplets_rag_docs + bm25_triplets_rag_docs]))
    topk_useful_triplets_docs = [t for t in sorted(usefulness_triplets_docs, key=lambda t: t.score, reverse=True) if t.label == "Useful"][:useful_triplets_top_k]
    
    naive_rag_query = f"{query}\n{build_naive_rag_query(topk_useful_triplets_docs)}"
    
    faiss_rag_docs = faiss_rag.run(naive_rag_query, top_k=faiss_rag_top_k) 
    bm25_rag_docs = bm25_rag.run(naive_rag_query, top_k=bm25_rag_top_k)
    
    return faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs, topk_useful_triplets_docs

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = {
        "text": [],
        "label": [],
        "phrase": [],
        "symptom": [],
        "analysis": [],
        "faiss_rag_results": [],
        "bm25_rag_results": [],
        "usefulness_triplets_docs": [],
        "topk_useful_triplets_docs": []
    }

    for _, row in batch.iterrows():
        text = row["text"]
        label = row["label"]

        for sp in row["symptom_phrases"]:
            phrase = sp['phrase']
            symptom = sp['symptom']
            analysis = sp['analysis']

            faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs, topk_useful_triplets_docs = pipeline(phrase, symptom, analysis)

            outputs['text'].append(text)
            outputs['label'].append(label)
            outputs['phrase'].append(phrase)
            outputs['symptom'].append(symptom)
            outputs['analysis'].append(analysis)
            outputs['faiss_rag_results'].append([doc.model_dump()  for doc in faiss_rag_docs])
            outputs['bm25_rag_results'].append([doc.model_dump() for doc in bm25_rag_docs])
            outputs['usefulness_triplets_docs'].append([t.model_dump() for t in usefulness_triplets_docs])
            outputs['topk_useful_triplets_docs'].append([t.model_dump() for t in topk_useful_triplets_docs])
    
    save_df = pd.DataFrame(outputs)
    save_df.to_json(batch_save_path, orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("jsonl_path", type=Path, help="Path to the jsonl file")
    parser.add_argument("--random_sample", type=int, default=None, help="Number of random samples to take")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    df = pd.read_json(jsonl_path, lines=True)
    
    if "symptom_phrases" not in df.columns:
        raise ValueError("The jsonl file does not have 'symptom_phrases' column.")
    
    if args.random_sample is not None:
        df = df.sample(n=args.random_sample, random_state=42)
    
    symptom_phrase_extractor = SymptomPhrasesExtractor()
    faiss_triplets_rag = FaissTripletsRag()
    bm25_triplets_rag = BM25TripletsRag()
    triplet_usefulness = TripletsUsefulness()
    faiss_rag = FaissRag()
    bm25_rag = BM25Rag()

    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = Path(f"./outputs/{jsonl_path.stem}/{timestring}")
    run_folder.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        completed_batches = get_completed_batch_numbers(run_folder)
        for i in range(0, len(df), batch_size):
            if i//batch_size in completed_batches:
                print(f"Skipping batch {i//batch_size}")
                continue
            futures.append(executor.submit(thread_func, df[i:i+batch_size], run_folder / f"batch_{i//batch_size}.jsonl"))
        for future in tqdm(futures):
            future.result()
    
    merge_jsonl_files(run_folder)