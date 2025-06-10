import pandas as pd
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge the output files from the gen_triplet_rag_explanations script.")
    parser.add_argument("run_folder", type=str, help="Path to the run folder")
    parser.add_argument("--remove_duplicates", action="store_true", help="Remove duplicate questions")
    args = parser.parse_args()
    run_folder = Path(args.run_folder)
    
    dfs = [pd.read_csv(file) for file in run_folder.rglob("*.csv")]
    
    df = pd.concat(dfs, ignore_index=True)

    df.dropna(inplace=True)
    if args.remove_duplicates:
        df.drop_duplicates(subset="question", keep="first", inplace=True)
    
    df.to_csv(run_folder / f"{run_folder.parent.stem}_merged.csv", index=False)