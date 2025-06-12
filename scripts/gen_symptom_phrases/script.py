import sys
sys.path.append("/home/sracha/proper_kg_project")

import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.utils import OpenAIModel
from datetime import datetime
from src.components.symptom_phrases import SymptomPhrasesExtractor
import argparse
import shutil

from src.utils import get_completed_batch_numbers, merge_jsonl_files

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = {
        "symptom_phrases": [],
    }

    for _, row in batch.iterrows():
        text = row["text"]
        response = extractor.run(text)
        outputs["symptom_phrases"].append([phrase.model_dump() for phrase in response])
    batch.loc[:, 'symptom_phrases'] = outputs["symptom_phrases"]
    batch.to_json(batch_save_path, orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--random_sample", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    df = pd.read_csv(csv_path)
    if args.random_sample:
        df = df.sample(args.random_sample, random_state=42)
    
    model = OpenAIModel(model_name="gpt-4o-mini", device="cpu")
    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = Path(f"./outputs/{csv_path.stem}") / timestring
    run_folder.mkdir(parents=True, exist_ok=True)
    extractor = SymptomPhrasesExtractor()

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
    shutil.copyfile(csv_path, run_folder / f"{csv_path.stem}_input.csv")