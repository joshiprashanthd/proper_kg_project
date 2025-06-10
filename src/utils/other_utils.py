import pandas as pd
from pathlib import Path

def merge_jsonl_files(run_path: Path):
    dfs = []
    for file in run_path.glob("*.jsonl"):
        if file.name.endswith(".jsonl"):
            df = pd.read_json(file, lines=True)
            dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_json(run_path / "merged.jsonl", orient='records', lines=True)

def merge_csv_files(run_path: Path, dropna=True, remove_duplicates=True):
    dfs = [pd.read_csv(file) for file in run_path.rglob("*.csv")]
    df = pd.concat(dfs, ignore_index=True)
    if dropna:
        df.dropna(inplace=True)
    if remove_duplicates:
        df.drop_duplicates(subset="question", keep="first", inplace=True)
    df.to_csv(run_path / "merged.csv", index=False)

def get_completed_batch_numbers(outputs_folder: Path) -> set[int]:
    completed_batches = set()
    for file in outputs_folder.glob("*.csv"):
        if file.name.startswith("batch_") and file.name.endswith(".csv"):
            batch_number = int(file.stem.split("_")[1])
            completed_batches.add(batch_number)
    return completed_batches