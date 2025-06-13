import pandas as pd
from pathlib import Path
import ast
import json

def merge_jsonl_files(run_path: Path):
    dfs = []
    for file in run_path.glob("batch_*.jsonl"):
        df = pd.read_json(file, lines=True)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_json(run_path / "merged.jsonl", orient='records', lines=True)
    merged_df.to_json(run_path / "merged.json", orient='records')

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

def convert_df_to_nested_structure(df):
    """
    Convert DataFrame to the desired nested JSON structure
    """
    result = []
    
    # Group by 'text' column
    grouped = df.groupby('text')
    
    for text, group in grouped:
        # Get the label (assuming it's the same for all rows with same text)
        label = group['label'].iloc[0]
        
        # Create symptom_phrases list
        symptom_phrases = []
        
        for _, row in group.iterrows():
            def safe_parse_json(value):
                if value is None:
                    return None
                
                # Check if it's a pandas NA/NaN (but not an array)
                if not isinstance(value, (list, dict)) and pd.isna(value):
                    return None
                
                # If it's already a list or dict, return as is
                if isinstance(value, (list, dict)):
                    return value
                
                # Handle empty string
                if isinstance(value, str) and value == '':
                    return None
                
                # Try to parse string as JSON
                if isinstance(value, str):
                    try:
                        return ast.literal_eval(value)
                    except:
                        try:
                            return json.loads(value)
                        except:
                            return value
                
                return value
            
            symptom_phrase = {
                "phrase": row['phrase'] if pd.notna(row['phrase']) else "",
                "symptom": row['symptom'] if pd.notna(row['symptom']) else "",
                "analysis": row['analysis'] if pd.notna(row['analysis']) else "",
                "symptom_phrase_label": safe_parse_json(row['symptom_phrase_label']),
                "faiss_rag_docs": safe_parse_json(row['faiss_rag_docs']),
                "bm25_rag_docs": safe_parse_json(row['bm25_rag_docs']),
                "usefulness_triplets_docs": safe_parse_json(row['usefulness_triplets_docs'])
            }
            
            symptom_phrases.append(symptom_phrase)
        
        # Create the final structure
        text_entry = {
            "text": text,
            "label": label,
            "symptom_phrases": symptom_phrases
        }
        
        result.append(text_entry)
    
    return pd.DataFrame(result)