import sys
sys.path.append("/home/sracha/proper_kg_project")

import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime
from src.components.symptom_phrases import SymptomPhraseLabeller
import argparse
import shutil

from src.utils import get_completed_batch_numbers, merge_jsonl_files, keep_first_definition
from src.types import RagDocument, TripletLabel

CONTEXT_TEMPLATE = """
<analysis>
{analysis}
</analysis>

<documents>
{faiss_rag_docs}
{bm25_rag_docs}
</documents>

<triplets>
{usefulness_triplets_docs}
</triplets>
"""

def build_context(analysis: str, faiss_rag_docs: list[RagDocument], bm25_rag_docs: list[RagDocument], usefulness_triplets_docs: list[TripletLabel]):
    return CONTEXT_TEMPLATE.format(
        analysis=analysis,
        faiss_rag_docs="\n".join([doc['content'] for doc in faiss_rag_docs]),
        bm25_rag_docs="\n".join([doc['content'] for doc in bm25_rag_docs]),
        usefulness_triplets_docs="\n".join(keep_first_definition([doc['triplet'] for doc in usefulness_triplets_docs]))
    )

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = {
        "symptom_phrase_label": [],
    }

    for _, row in batch.iterrows():
        label = row['label']
        phrase = row['phrase']
        analysis = row['analysis']
        faiss_rag_docs = row['faiss_rag_docs']
        bm25_rag_docs = row['bm25_rag_docs']
        usefulness_triplets_docs = row['usefulness_triplets_docs']

        query = f"Phrase: '{phrase}'\nReddit Post Label: '{label}'"

        label = symptom_phrase_labeller.run(query, build_context(analysis, faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs))
        outputs["symptom_phrase_label"].append(label.model_dump())
    
    batch.loc[:, 'symptom_phrase_label'] = outputs["symptom_phrase_label"]
    batch.to_json(batch_save_path, orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", type=str)
    parser.add_argument("--random_sample", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)

    args = parser.parse_args()
    jsonl_path = Path(args.jsonl_path)

    df = pd.read_json(jsonl_path, lines=True)
    if args.random_sample:
        df = df.sample(args.random_sample, random_state=42)
    
    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = Path(f"./outputs/{jsonl_path.stem}") / timestring
    run_folder.mkdir(parents=True, exist_ok=True)
    symptom_phrase_labeller = SymptomPhraseLabeller()

    batch_size = args.batch_size
    completed_batches = get_completed_batch_numbers(run_folder)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            if i//batch_size in completed_batches:
                continue
            futures.append(executor.submit(thread_func, df[i:i+batch_size], run_folder / f"batch_{i//batch_size}.jsonl"))
        for future in tqdm(futures):
            future.result()
    
    merge_jsonl_files(run_folder)
    shutil.copyfile(jsonl_path, run_folder / f"{jsonl_path.stem}_input.jsonl")