import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Path to your current model folder (where adapter_model.safetensors and base model shards are)
model_path = "/home/sracha/proper_kg_project/finetuning/mhqa-lora-20"  # <-- change this

# Path where you want to save the fully merged model
merged_model_path = "/home/sracha/proper_kg_project/finetuning/merged-mhqa-lora-20"  # <-- change this
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load adapter
model = PeftModel.from_pretrained(model, model_path)

# Merge LoRA weights into base model
model = model.merge_and_unload()

# ⭐⭐ Important step ⭐⭐
# Now model is merged but still wrapped inside a PeftModel class
# Unwrap it back to base model
try:
    model = model.base_model
except AttributeError:
    pass  # In case it is already unwrapped

# Save merged model
model.save_pretrained(merged_model_path, safe_serialization=True)
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model saved at: {merged_model_path}")