from concurrent.futures import ThreadPoolExecutor
import datetime
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from datetime import datetime

class InferenceResponse(BaseModel):
    raw_response: str
    correct_option: Optional[str] = None
    justification: Optional[str] = None


def get_completed_batch_numbers(outputs_folder):
    completed_batches = set()
    for file in outputs_folder.rglob("*.csv"):
        if file.name.startswith("batch_") and file.name.endswith(".csv"):
            batch_number = int(file.stem.split("_")[1])
            completed_batches.add(batch_number)
    return completed_batches

def perform_inference(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[prompt_length:].strip()
    
    structured_response = InferenceResponse(raw_response=response)
    try:
        option_text = response.split("Correct Option:")[-1].strip()
        if option_text and option_text[0] in "1234":
            structured_response.correct_option = option_text[0]
            if "Justification:" in response:
                justification = response.split("Justification:")[-1].strip()
                structured_response.justification = justification
    except:
        pass
    print('output', structured_response)
    return structured_response

def evaluate(batch, batch_save_path):
    responses = []
    for index, row in batch.iterrows():
        question = row['question']
        op1 = row['option1']
        op2 = row['option2']
        op3 = row['option3']
        op4 = row['option4']

        formatted_prompt = prompt.format(
            question=question,
            op1=op1,
            op2=op2,
            op3=op3,
            op4=op4
        )

        response = perform_inference(formatted_prompt)
        responses.append(response)
    
    batch['output_raw_response'] = [resp.raw_response for resp in responses]
    batch['output_predictions'] = [resp.correct_option for resp in responses]
    batch['output_justification'] = [resp.justification for resp in responses]

    batch.to_csv(batch_save_path, index=False)
    
prompt = """
You are a highly skilled medical expert tasked with evaluating multiple-choice questions. 
Your role is to select the single most accurate and contextually correct option from the given choices. 
Do not favor any option based on its order. Evaluate all options carefully and justify your choice implicitly.

Output Format:
Correct Option: <1 or 2 or 3 or 4>
Justification: <justification for the answer>

Question: {question}
Options: 
Option1: {op1}
Option2: {op2}
Option3: {op3}
Option4: {op4}

Respond with the letter corresponding to your choice (1 or 2 or 3 or 4) followed by a justification.
Correct Option: """


if __name__ == "__main__":
    model_path = Path("/home/sracha/proper_kg_project/finetuning/mhqa-lora-40-new")
    adapter_path = model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    df = pd.read_csv("/home/sracha/proper_code_base_pubmed_dataset/datasets/mhqa-gold-all-labels.csv")
    df = df[["id", "topic", "type", "question", "option1", "option2", "option3", "option4", "correct_option", "correct_option_number"]]

    outputs_folder = Path(f"./outputs/{model_path.stem}")
    outputs_folder.mkdir(parents=True, exist_ok=True)

    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = outputs_folder / f"{model_path.stem}_{timestring}"
    run_folder.mkdir(parents=True, exist_ok=True)

    batch_size = 20
    completed_batches = get_completed_batch_numbers(outputs_folder)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_number = i // batch_size
            if batch_number in completed_batches:
                print(f"Batch {batch_number} already completed. Skipping.")
                continue

            futures.append(executor.submit(evaluate, df[i:i+batch_size], outputs_folder / run_folder / f"batch_{batch_number}.csv"))
        for future in tqdm(futures):
            future.result()