import sys
sys.path.append("/home/sracha/proper_kg_project")

import pandas as pd
import argparse
from pathlib import Path
import re
import shutil
from datetime import datetime

from src.components.usefulness import TripletsUsefulness

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def split_triplets(triplets_str: str):
    splitter = r"\)\n\(Head:"
    triplets = re.split(splitter, triplets_str)
    cleaned_triplets = []
    for i, t in enumerate(triplets):
        if t.strip():  # Only add non-empty strings
            if i > 0: # For all except the very first one, prepend '(Head:' back
                cleaned_triplets.append("(Head:" + t)
            else:
                cleaned_triplets.append(t)
    return cleaned_triplets

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    triplets_usefulness = TripletsUsefulness()
    outputs = {
        "id": [],
        "question": [],
        "answer": [],
        "triplets": [],
        "label": [],
        "justification": [],
    }

    for _, row in batch.iterrows():
        triplets = split_triplets(row["triplets"])
        triplet_labels = triplets_usefulness.run(row["question"], row["answer"], triplets)
        for tl in triplet_labels:
            outputs["id"].append(row["id"])
            outputs["question"].append(row["question"])
            outputs["answer"].append(row["answer"])
            outputs["triplets"].append(tl.triplet)
            outputs["label"].append(tl.label)
            outputs["justification"].append(tl.justification)

    batch = pd.DataFrame(outputs)
    batch.to_csv(batch_save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the usefulness of the triplets in the given run folder.")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    outputs_folder = Path(f"./outputs/find_usefulness/{csv_path.stem}")
    outputs_folder.mkdir(parents=True, exist_ok=True)

    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = outputs_folder / f"run_{timestring}"
    run_folder.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise ValueError("The csv file does not exist.")
    
    df = pd.read_csv(csv_path)
    
    if "triplets" not in df.columns:
        raise ValueError("The csv file does not have a 'triplets' column.")
    
    triplets_usefulness = TripletsUsefulness()

    batch_size = 10
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            futures.append(executor.submit(thread_func, df[i:i+batch_size], run_folder / f"batch_{i//batch_size}.csv"))
        for future in tqdm(futures):
            future.result()
    
    shutil.copyfile(csv_path, run_folder / f"{csv_path.parent.stem}_input.csv")
    