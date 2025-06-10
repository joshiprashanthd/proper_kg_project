import pandas as pd
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge the output files from the gen_triplet_rag_explanations script.")
    parser.add_argument("run_folder", type=str, help="Path to the run folder")
    parser.add_argument("--remove_duplicates", action="store_true", help="Remove duplicate questions")
    args = parser.parse_args()
    run_folder = Path(args.run_folder)
    
    