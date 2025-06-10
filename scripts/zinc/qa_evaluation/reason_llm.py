import os
import requests
from dotenv import load_dotenv
import pandas as pd
import argparse
import re
from prompts_reason import (
    ask_llama_mcqa,
    ask_llama_yes_no,
    ask_llama_long_form,
    ask_llama_classification_single,
    ask_llama_classification_multiple
)


# Load your HF token from the .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

hf_model_path ="meta-llama/Llama-3.1-8B-Instruct"
# The endpoint for LLaMA-3.1 8B
LLAMA_ENDPOINT = f"https://api-inference.huggingface.co/models/{hf_model_path}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# load data

results=[]

# Set up argument parser
parser = argparse.ArgumentParser(description="Load dataset for evaluation.")
parser.add_argument('--type', type=str, required=True,
                    choices=['mcqa', 'yes_no', 'long_form', 'classification_single', 'classification_multiple'],
                    help="Type of dataset to use.")
parser.add_argument('--name', type=str, required=True,
                    help="Name of the dataset to load.")

args = parser.parse_args()

# Access arguments
dataset_type = args.type
dataset_name = args.name


df = pd.read_csv(f'/home/sracha/proper_kg_project/data/qna/combined/test/{dataset_name}_test.csv')
len=len(df)
counter=1
if dataset_type == 'mcqa':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        opa, opb, opc, opd = row['option1'], row['option2'], row['option3'], row['option4']
        try:
            answer, reason, store = ask_llama_mcqa(question, opa, opb, opc, opd)
        except Exception as e:
            answer, reason, store = 'error', 'error', 'error'
            print(f"Error processing question ID {id}: {e}")
        print(f"Progress: {counter}/{len}")
        print('answer:', answer)
        print('reason:', reason)
        #print('store', store)
        results.append({'id': id, 'source': f'{dataset_name}', 'question': question, 'answer': answer, 'reason': reason, 'store': store})
        counter+=1

if dataset_type == 'yes_no':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        try:
            answer, store = ask_llama_yes_no(question)
        except Exception as e:
            answer, store = 'error', 'error', 'error'
            print(f"Error processing question ID {id}: {e}")
        print(f"Progress: {counter}/{len}")
        counter+=1
        print('answer', answer)
        #print('store', store)
        results.append({'id': id, 'source': f'{dataset_name}', 'question': question, 'answer': answer})

if dataset_type == 'long_form':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        answer = ask_llama_long_form(question)
        print(f"Progress: {counter}/{len}")
        counter+=1
        print(answer)
        results.append({'id': id, 'source': f'{dataset_name}', 'question': question, 'answer': answer})

if dataset_type == 'classification_single':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        labels = row['labels']
        answer = ask_llama_classification_single(question, labels)
        print(answer)
        print(f"Progress: {counter}/{len}")
        counter+=1
        results.append({'id': id, 'source': f'{dataset_name}', 'question': question, 'answer': answer})

if dataset_type == 'classification_multiple':
    for index, row in df.iterrows():
        id = row['id']
        question = row['question']
        labels = row['labels']
        answer = ask_llama_classification_multiple(question, labels)
        print(answer)
        print(f"Progress: {counter}/{len}")
        counter+=1
        results.append({'id': id, 'source': f'{dataset_name}', 'question': question, 'answer': answer})

# save results
results_df = pd.DataFrame(results)
results_df.to_csv(f'/home/sracha/proper_kg_project/scripts/qa_evaluation/results/reasoning_llm/{dataset_name}_llama_results.csv', index=False)

