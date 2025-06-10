import os
import requests
from dotenv import load_dotenv
import pandas as pd
import argparse

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
    You are a mental health expert. You are given a question and four option candidates.
    Please answer the question by selecting one of the options that best answers the question.
    Output the complete option name only, and nothing else.

    Question: 
    {question}
    Options: 
    (A) {opa}
    (B) {opb} 
    (C) {opc}
    (D) {opd}

    Output: 
    <answer candidate>
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

def ask_llama_yes_no(question):
    # Create a simple instruction-style prompt
    prompt = f"""
    You are a mental health expert. You are given a question whose answer can be either yes or no.
    Please answer the question by outputting 'yes' or 'no' that best answers the question.
    Output 'yes' or 'no' only, and nothing else.

    Question: 
    {question}

    Output: 
    <yes or no>
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
