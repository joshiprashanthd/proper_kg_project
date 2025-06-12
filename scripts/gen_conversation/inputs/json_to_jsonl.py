import argparse
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert json to jsonl")
    parser.add_argument("json_path", type=str, help="Path to the json file")
    parser.add_argument("jsonl_path", type=str, help="Path to the jsonl file")
    args = parser.parse_args()
    json_path = Path(args.json_path)
    jsonl_path = Path(args.jsonl_path)
    df = pd.read_json(json_path)
    df.to_json(jsonl_path, orient='records', lines=True)