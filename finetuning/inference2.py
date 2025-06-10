import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Path to your saved model (folder containing adapter + base model)
model_path = "/home/sracha/proper_kg_project/finetuning/mhqa-lora-20"  # <<<< change this
adapter_path = model_path  # Same folder, because adapter_model.safetensors is here

# Load base model first
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Now inject adapter weights
model = PeftModel.from_pretrained(model, adapter_path)

# Merge LoRA into the model if you want
model = model.merge_and_unload()

# Inference
model.eval()

# Example prompt
prompt = "You are a mental health expert. Answer the following question: How are anxiety and depression related?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
