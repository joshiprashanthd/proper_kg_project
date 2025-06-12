import sys
sys.path.append("/home/sracha/proper_kg_project")

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.components.conversations import AngstSymptomDisorderId
from src.utils import convert_df_to_nested_structure

PHRASE_TEMPLATE = """
<phrase> 
Phrase: '{phrase}' 

<symptom>
{symptom}
</symptom>

<analysis>
{analysis}
</analysis>

<phrase_label>
{phrase_label}
</phrase_label>

<justification>
{justification}
</justification>

<retrieved_documents>
{faiss_rag_docs}
{bm25_rag_docs}
</retrieved_documents>

<triplets>
{usefulness_triplets_docs}
</triplets>
</phrase>
"""

def create_inputs(row: pd.Series):
    query = f"Text: {row['text']}\nLabel: {row['label']}"
    phrase_contexts = []
    for phrase in row['symptom_phrases']:
        phrase_context = PHRASE_TEMPLATE.format(
            phrase=phrase['phrase'],
            symptom=phrase['symptom'],
            analysis=phrase['analysis'],
            phrase_label=phrase['symptom_phrase_label']['label'],
            justification=phrase['symptom_phrase_label']['justification'],
            faiss_rag_docs="\n\n".join([data['content'] for data in phrase['faiss_rag_docs']]),
            bm25_rag_docs="\n\n".join([data['content'] for data in phrase['bm25_rag_docs']]),
            usefulness_triplets_docs="\n\n".join([data['triplet'] for data in phrase['usefulness_triplets_docs']])
        )
        phrase_contexts.append(phrase_context)
    
    context = "\n\n".join(phrase_contexts)
    return query, context


def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = {
        "conversation": []
    }
    for i, row in batch.iterrows():
        query, context = create_inputs(row)
        response = conv_angst.run(query, context)
        outputs["conversation"].append(response.model_dump())
    batch.loc[:, 'conversation'] = outputs["conversation"]
    batch.to_json(batch_save_path, orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate supporting questions for the given dataset.")
    parser.add_argument("jsonl_path", type=str, help="Path to the jsonl file")
    parser.add_argument("--random_sample", type=int, help="Number of random samples to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--workers", type=int, default=10, help="Number of workers")
    args = parser.parse_args()
    jsonl_path = Path(args.jsonl_path)

    conv_angst = AngstSymptomDisorderId()
    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_path = Path(f"./outputs/{jsonl_path.stem}/{timestring}")
    run_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(jsonl_path, lines=True)
    if args.random_sample:
        df = df.sample(args.random_sample, random_state=42)
    
    df = convert_df_to_nested_structure(df)

    batch_size = args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df[i:i+batch_size]
            futures.append(executor.submit(thread_func, batch, run_path / f"batch_{i}.jsonl"))

        for future in tqdm(as_completed(futures)):
            future.result()




