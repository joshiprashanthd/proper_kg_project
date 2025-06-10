import os
import requests
from dotenv import load_dotenv
import re
from huggingface_hub import InferenceClient

client = InferenceClient()

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

hf_model_path = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_ENDPOINT = f"https://api-inference.huggingface.co/models/{hf_model_path}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# -------------------------------
# Chain-of-Thought Prompt Functions
# -------------------------------

def ask_llama_mcqa_cot(question, opa, opb, opc, opd):
    """MCQA with step-by-step reasoning."""
    prompt = f"""
    You are a mental health expert. Analyze each option below step by step, then select the best answer.
    Follow this structure:
    
    "reasoning": "<evaluate each option (A/B/C/D) with 1-2 sentences>"
    "answer": "<correct_option_letter>"

    Example:
    "reasoning": "Option A: Incorrect because... Option B: Correct because..." 
    "answer": "B"

    Question: {question}
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
            "max_new_tokens": 400,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    
    response_text = response.json()[0]["generated_text"]
    
    # Extract reasoning and answer
    reasoning_match = re.search(r'"reasoning":\s*"(.*?)"\s*"answer":', response_text, re.DOTALL)
    answer_match = re.search(r'"answer":\s*"(.*?)"', response_text)
    
    if not (reasoning_match and answer_match):
        raise ValueError(f"Failed to parse response: {response_text}")
    
    return answer_match.group(1).strip(), reasoning_match.group(1).strip()


def ask_llama_yes_no_cot(question):
    """Yes/No with reasoning."""
    prompt = f"""
    You are a mental health expert. Analyze the question below step by step, then answer yes/no.
    Structure:
    
    "reasoning": "<explain factors supporting yes/no>"
    "answer": "<yes/no>"

    Example:
    "reasoning": "The DSM-5 criteria state... Thus, the answer is yes."
    "answer": "yes"

    Question: {question}
    Output:
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    
    response_text = response.json()[0]["generated_text"]
    
    reasoning_match = re.search(r'"reasoning":\s*"(.*?)"\s*"answer":', response_text, re.DOTALL)
    answer_match = re.search(r'"answer":\s*"(.*?)"', response_text)
    
    if not (reasoning_match and answer_match):
        raise ValueError(f"Failed to parse response: {response_text}")
    
    return answer_match.group(1).strip(), reasoning_match.group(1).strip()


def ask_llama_long_form_cot(question):
    """Long-form answer with structured reasoning."""
    prompt = f"""
    You are a mental health expert. Break down your answer into logical steps, then provide a conclusion.
    Structure:
    
    "reasoning": "<step-by-step analysis>"
    "answer": "<final detailed answer>"

    Example:
    "reasoning": "1. Define depression. 2. List symptoms. 3. Explain causes..."
    "answer": "Depression is characterized by..."

    Question: {question}
    Output:
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    
    response_text = response.json()[0]["generated_text"]
    
    reasoning_match = re.search(r'"reasoning":\s*"(.*?)"\s*"answer":', response_text, re.DOTALL)
    answer_match = re.search(r'"answer":\s*"(.*?)"', response_text)
    
    if not (reasoning_match and answer_match):
        raise ValueError(f"Failed to parse response: {response_text}")
    
    return answer_match.group(1).strip(), reasoning_match.group(1).strip()


def ask_llama_classification_single_cot(question, labels):
    """Single classification with reasoning."""
    prompt = f"""
    You are a mental health expert. Evaluate each label below, then select the best match.
    Structure:
    
    "reasoning": "<analyze each label against the question>"
    "answer": "<selected_label>"

    Example:
    "reasoning": "Label A: Relevant because... Label B: Irrelevant because..."
    "answer": "Label A"

    Question: {question}
    Labels: {labels}
    Output:
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    
    response_text = response.json()[0]["generated_text"]
    
    reasoning_match = re.search(r'"reasoning":\s*"(.*?)"\s*"answer":', response_text, re.DOTALL)
    answer_match = re.search(r'"answer":\s*"(.*?)"', response_text)
    
    if not (reasoning_match and answer_match):
        raise ValueError(f"Failed to parse response: {response_text}")
    
    return answer_match.group(1).strip(), reasoning_match.group(1).strip()


def ask_llama_classification_multiple_cot(question, labels):
    """Multi-label classification with reasoning."""
    prompt = f"""
    You are a mental health expert. Evaluate all labels below, then select all relevant ones.
    Structure:
    
    "reasoning": "<explain relevance of each label>"
    "answer": "<comma-separated_labels>"

    Example:
    "reasoning": "Label A: Relevant because... Label C: Relevant because..."
    "answer": "Label A, Label C"

    Question: {question}
    Labels: {labels}
    Output:
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(LLAMA_ENDPOINT, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    
    response_text = response.json()[0]["generated_text"]
    
    reasoning_match = re.search(r'"reasoning":\s*"(.*?)"\s*"answer":', response_text, re.DOTALL)
    answer_match = re.search(r'"answer":\s*"(.*?)"', response_text)
    
    if not (reasoning_match and answer_match):
        raise ValueError(f"Failed to parse response: {response_text}")
    
    return answer_match.group(1).strip(), reasoning_match.group(1).strip()
