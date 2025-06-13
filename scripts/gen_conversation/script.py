import sys
sys.path.append("/home/sracha/proper_kg_project")

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.components.conversations import AngstSymptomDisorderId
from src.utils import convert_df_to_nested_structure, merge_jsonl_files

PHRASE_TEMPLATE = """
<phrase> 
Phrase: '{phrase}' 
Symptom: '{symptom}'
Analysis: '{analysis}'
Label: '{label}'
Justification: '{justification}'
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
            label=phrase['label'],
            justification=phrase['justification']
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
    
    batch_size = args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df[i:i+batch_size]
            futures.append(executor.submit(thread_func, batch, run_path / f"batch_{i}.jsonl"))

        for future in tqdm(as_completed(futures)):
            future.result()

    merge_jsonl_files(run_path)



