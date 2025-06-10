import os
import requests
from dotenv import load_dotenv
import pandas as pd
import argparse
from prompts_cot import (
    ask_llama_mcqa_cot,  
    ask_llama_yes_no_cot,
    ask_llama_long_form_cot,
    ask_llama_classification_single_cot,
    ask_llama_classification_multiple_cot
)

# Load your HF token from the .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

hf_model_path = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_ENDPOINT = f"https://api-inference.huggingface.co/models/{hf_model_path}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


results = []

# Set up argument parser
parser = argparse.ArgumentParser(description="Load dataset for chain-of-thought evaluation.")
parser.add_argument('--type', type=str, required=True,
                    choices=['mcqa', 'yes_no', 'long_form', 'classification_single', 'classification_multiple'],
                    help="Type of dataset to use.")
parser.add_argument('--name', type=str, required=True,
                    help="Name of the dataset to load.")

args = parser.parse_args()

dataset_type = args.type
dataset_name = args.name

df = pd.read_csv(f'/home/sracha/proper_kg_project/data/qna/combined/test/{dataset_name}_test.csv')
total = len(df)
counter = 1

if dataset_type == 'mcqa':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        opa, opb, opc, opd = row['option1'], row['option2'], row['option3'], row['option4']
        try:
            # Chain-of-thought MCQA
            answer, reasoning = ask_llama_mcqa_cot(question, opa, opb, opc, opd)
        except Exception as e:
            answer, reasoning = 'error', f'Error: {e}'
            print(f"Error processing question ID {id}: {e}")
        print(f"Progress: {counter}/{total}")
        print("Answer:", answer)
        print("Reasoning:", reasoning)
        results.append({
            'id': id,
            'source': f'{dataset_name}',
            'question': question,
            'answer': answer,
            'reasoning': reasoning
        })
        counter += 1

if dataset_type == 'yes_no':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        try:
            answer, reasoning = ask_llama_yes_no_cot(question)
        except Exception as e:
            answer, reasoning = 'error', f'Error: {e}'
            print(f"Error processing question ID {id}: {e}")
        print(f"Progress: {counter}/{total}")
        print("Answer:", answer)
        print("Reasoning:", reasoning)
        results.append({
            'id': id,
            'source': f'{dataset_name}',
            'question': question,
            'answer': answer,
            'reasoning': reasoning
        })
        counter += 1

if dataset_type == 'long_form':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        try:
            answer, reasoning = ask_llama_long_form_cot(question)
        except Exception as e:
            answer, reasoning = 'error', f'Error: {e}'
            print(f"Error processing question ID {id}: {e}")
        print(f"Progress: {counter}/{total}")
        print("Answer:", answer)
        print("Reasoning:", reasoning)
        results.append({
            'id': id,
            'source': f'{dataset_name}',
            'question': question,
            'answer': answer,
            'reasoning': reasoning
        })
        counter += 1

if dataset_type == 'classification_single':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        labels = row['labels']
        try:
            answer, reasoning = ask_llama_classification_single_cot(question, labels)
        except Exception as e:
            answer, reasoning = 'error', f'Error: {e}'
            print(f"Error processing question ID {id}: {e}")
        print(f"Progress: {counter}/{total}")
        print("Answer:", answer)
        print("Reasoning:", reasoning)
        results.append({
            'id': id,
            'source': f'{dataset_name}',
            'question': question,
            'answer': answer,
            'reasoning': reasoning
        })
        counter += 1

if dataset_type == 'classification_multiple':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        labels = row['labels']
        try:
            answer, reasoning = ask_llama_classification_multiple_cot(question, labels)
        except Exception as e:
            answer, reasoning = 'error', f'Error: {e}'
            print(f"Error processing question ID {id}: {e}")
        print(f"Progress: {counter}/{total}")
        print("Answer:", answer)
        print("Reasoning:", reasoning)
        results.append({
            'id': id,
            'source': f'{dataset_name}',
            'question': question,
            'answer': answer,
            'reasoning': reasoning
        })
        counter += 1

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(f'/home/sracha/proper_kg_project/scripts/qa_evaluation/results/{dataset_name}_llama_cot_results.csv', index=False)
'''
 # Simple hardcoded test example

if __name__ == "__main__":
   
    question = "Which of the following is a common warning sign of depression in adolescents?"
    opa = "Persistent sadness or low mood"
    opb = "Sudden improvement in academic performance"
    opc = "Increased interest in social activities"
    opd = " Consistent high energy levels"
    answer, reasoning = ask_llama_mcqa_cot(question, opa, opb, opc, opd)
    print("Answer:", answer)
    print("Reasoning:", reasoning)
  '''  