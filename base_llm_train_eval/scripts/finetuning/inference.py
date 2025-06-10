import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your saved model
model_path = "/home/sracha/proper_kg_project/finetuning/merged-mhqa-lora-20"  # Change if needed

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",  # Automatically load on available GPU
)

# Function to prepare prompt and generate response
def generate_response(query, system_prompt="You are a mental health expert. Answer the following question:"):
    # Prepare prompt manually because you trained with "chatml" format
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>human\n{query}<|im_end|>\n"
        f"<|im_start|>gpt\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the output
    generated_response = generated_text[len(prompt):].strip()
    
    return generated_response

# Example usage
if __name__ == "__main__":
    question = "How are anxiety and depression related"
    response = generate_response(question)
    print("Response:", response)
