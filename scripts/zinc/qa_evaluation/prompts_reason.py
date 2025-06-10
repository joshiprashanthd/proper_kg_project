import os
import requests
from dotenv import load_dotenv
import pandas as pd
import argparse
import re

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


def ask_llama_mcqa(question, opa, opb, opc, opd):
    # Create a simple instruction-style prompt
    prompt = f"""
    You are a mental health expert AI. You will be given a multiple-choice question with four options.  
Your task is to select the *single best answer* and provide a *brief reasoning (1–3 sentences)* for your choice.  

Important Instructions:  
- Provide only one final answer and one reasoning.  
- Do NOT evaluate or comment on other options.  
- Do NOT repeat or generate multiple answers.  
- Follow the exact format below and do NOT include anything else in your output.

Response Format:  
\"\"\"json
"answer": "<selected_option_letter>"  
"reason": "<your concise reasoning>"  
\"\"\"

Example Output:  
\"\"\"json
"answer": "B"  
"reason": "The symptoms described align with B, indicating a depressive disorder. The patient shows hallmark signs such as low mood and loss of interest."  
\"\"\"

Question:  
{question}

Options:  
(A) {opa}  
(B) {opb}  
(C) {opc}  
(D) {opd}

Output:
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            
            "seed": 42,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)

    if response.status_code == 200:
        result = response.json()
        store = result[0]["generated_text"]
    else:
        raise Exception(f"Failed to get response: {response.status_code}, {response.text}")
    
    # Extract the answer from the response
    # find "answer": " in the response using regex
    
    answer_match = re.search(r'"answer": "(.*?)"', store)
    reason_match = re.search(r'"reason":\s*"(.*?)"', store)
    if answer_match and reason_match:
        answer = answer_match.group(1).strip()
        reason = reason_match.group(1).strip()
        return answer, reason, store
    else:
        raise Exception(f"Failed to extract answer or reason from response: {store}")
    

def ask_llama_yes_no(question):
    # Create a simple instruction-style prompt
    prompt = f"""
    You are a mental health expert. You are given a question whose answer can be either yes or no.
    Please answer the question by outputting 'yes' or 'no' that best answers the question.
    Output 'yes' or 'no' only, and nothing else and respond ONLY in the structure shown below.
    Respond in this structure:
    "answer": "<yes or no>"

    Example output 1:
    "answer": "yes"

    Example output 2:
    "answer": "no"

    Do NOT include any explanation, commentary, or additional text.

    Question: 
    {question}

    Output: 
    
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            
            "seed": 42,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)

    if response.status_code == 200:
        result = response.json()
        store = result[0]["generated_text"]
    else:
        raise Exception(f"Failed to get response: {response.status_code}, {response.text}")

    match = re.search(r'"answer": "(.*?)"', store)
    if match:
        answer = match.group(1)
        # Remove any leading or trailing whitespace
        answer = answer.strip()
        return answer, store
    else:
        raise Exception(f"Failed to extract answer from response: {store}")
    

def ask_llama_long_form(question):
    # Create a simple instruction-style prompt
    prompt = f"""
    You are a mental health expert. You are given a question to be answered in long-form fashion.
    Please answer the question by outputting a long-form answer in relation to the question. The answer should not be more than 5-6 sentences.
    Output the complete answer only, and nothing else.

    Question: 
    {question}

    Output: 
    <answer>
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"]
    else:
        raise Exception(f"Failed to get response: {response.status_code}, {response.text}")

def ask_llama_classification_single(question, labels):
    # label string
    # options = ""
    # for i in labels:
    #     options += f"{i}, "
    # options = options[:-2]

    # Create a simple instruction-style prompt
    prompt = f"""
    You are a mental health expert. You are given a query description and options to which the description can belong to.
    Classify the query into one of the options where it best belongs to based on symptoms. Select one of the options.
    Output the complete option name only, and nothing else.

    Question: 
    {question}

    Options: 
    {labels}

    Output: 
    <answer>
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"]
    else:
        raise Exception(f"Failed to get response: {response.status_code}, {response.text}")

def ask_llama_classification_multiple(question, labels):
    # label string
    # options = ""
    # for i in labels:
    #     options += f"{i}, "
    # options = options[:-2]

    # Create a simple instruction-style prompt
    prompt = f"""
    You are a mental health expert. You are given a query description and options to which the description can belong to.
    Classify the query description into relevant options where it best belongs to based on symptoms. More than one option can be selected.
    Output the complete option name. If there are multiple options, separate them with a comma. Do not add 'and' or 'or' between the options.
    Output the complete option name only, and nothing else.

    Question: 
    {question}

    Options: 
    {labels}

    Output: 
    <answer>
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"]
    else:
        raise Exception(f"Failed to get response: {response.status_code}, {response.text}")
