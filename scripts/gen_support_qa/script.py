import sys
sys.path.append("/home/sracha/proper_kg_project")

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.components.supporting_questions import MHQASupportQuestions, MedMCQASupportQuestions

def create_inputs(batch: pd.DataFrame):
    Q_TEMPLATE = """
Question ID: {question_id}
Question: {question}
Options: {options}
Answer: {answer}
"""
    inputs = []
    for i, row in batch.iterrows():
        inputs.append(Q_TEMPLATE.format(
            question_id=row['question_id'],
            question=row['question'],
            options=str(row['option1']) + ", " + str(row['option2']) + ", " + str(row['option3']) + ", " + str(row['option4']),
            answer=str(row['answer'])
        ))
    return "\n\n".join(inputs)

def process_batch(batch: pd.DataFrame, batch_save_path: Path, sqa):
    response = sqa.generate_text(inputs=create_inputs(batch))
    batch.loc[:, 'supporting_questions'] = list(map(lambda x: x.supporting_questions, response.question_outputs))
    batch.to_json(batch_save_path, orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate supporting questions for the given dataset.")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--random_sample", type=int, help="Number of random samples to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--prompt_template", type=str, required=True, help="Prompt template", choices=["mhqa", "medmcqa"])
    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if args.prompt_template == "mhqa" or "mhqa" in csv_path.stem:
        sqa = MHQASupportQuestions()
    elif args.prompt_template == "medmcqa" or "medmcqa" in csv_path.stem:
        sqa = MedMCQASupportQuestions()
    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_path = Path(f"./outputs/{csv_path.stem}/{timestring}")
    run_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.random_sample:
        df = df.sample(args.random_sample, random_state=42)
    
    df['question_id'] = df.index
    batch_size = args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df[i:i+batch_size]
            futures.append(executor.submit(process_batch, batch, run_path / f"batch_{i}.jsonl", sqa))

        for future in tqdm(as_completed(futures)):
            future.result()





