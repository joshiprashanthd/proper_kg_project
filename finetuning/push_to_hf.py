from huggingface_hub import create_repo, upload_folder

# 1. Set your model name
model_name = "llama-mhqa-60-lora"  # Example: "my-finetuned-llama-adapter"

# 2. Create a new repo (private=False if you want it public)
create_repo(model_name, private=True)

# 3. Push your local folder
upload_folder(
    folder_path="/home/sracha/proper_kg_project/finetuning/mhqa-lora-60",  # <-- your local folder path
    repo_id=f"surajracha/{model_name}",
    commit_message="Initial adapter upload"
)
