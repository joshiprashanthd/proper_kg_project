import os
import wandb
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from unsloth import FastLanguageModel
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from unsloth import unsloth_train
from sklearn.model_selection import train_test_split
from datasets import concatenate_datasets
# Load environment variables
load_dotenv()

def setup_wandb(project_name: str, run_name: str):
    try:
        api_key = os.getenv("WANDB_API_KEY")
        if api_key is None:
            raise EnvironmentError("WANDB_API_KEY is not set in the environment variables.")
        wandb.login(key=api_key)
        print("Successfully logged into WandB.")
    except Exception as e:
        print(f"Error logging into WandB: {e}")
    
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "true"
    
    try:
        wandb.init(project=project_name, name=run_name)
        print(f"WandB run initialized: Project - {project_name}, Run - {run_name}")
    except Exception as e:
        print(f"Error initializing WandB run: {e}")

# Set up WandB
setup_wandb(project_name="DiagMent-LLM", run_name="no_explanation-1")

# Authenticate with Hugging Face
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise EnvironmentError("HUGGINGFACE_TOKEN is not set in the environment variables.")
login(token=hf_token)

# Model setup
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load your CSV dataset
csv_angst = "/home/sracha/proper_kg_project/llm_training/dataset/train/angst-silver_train.csv"  # Replace with your actual file path
csv_medmcqa = "/home/sracha/proper_kg_project/llm_training/dataset/train/medmcqa_train.csv"
csv_mhqa = "/home/sracha/proper_kg_project/llm_training/dataset/train/mhqa-b-subset_train.csv"
df_angst = pd.read_csv(csv_angst)
df_medmcqa = pd.read_csv(csv_medmcqa)
df_mhqa = pd.read_csv(csv_mhqa)
# combine df_part1 and df_part2
# Combine the DataFrames
#df = pd.concat([df_part1, df_part2, df_part3], ignore_index=True)
#print(len(df))
#print(df['correct_answer'][10])
# Ensure column names are correct and create the new DataFrame with proper line breaks
'''
df1 = pd.DataFrame({
    "query": df.apply(lambda row: f"""question: {row['question']}\nOptions:\noption1: {row['option1']}\noption2: {row['option2']}\noption3: {row['option3']}\noption4: {row['option4']}""", axis=1),
    "answers": df.apply(lambda row: f"""explanation: {row['explanation']}\nanswer: {row['correct_answer']}""", axis=1)
})
'''

df1_angst = pd.DataFrame({
    "query": df_angst.apply(lambda row: f"""post: {row['text']}""", axis=1),
    "answers": df_angst.apply(lambda row: f"""depression: {row['depression_label']}\nanxiety: {row['anxiety_label']}""", axis=1)
})

df1_medmcqa = pd.DataFrame({
    "query": df_medmcqa.apply(lambda row: f"""question: {row['question']}\nOptions:\noption1: {row['opa']}\noption2: {row['opb']}\noption3: {row['opc']}\noption4: {row['opd']}""", axis=1),
    "answers": df_medmcqa.apply(lambda row: f"answer: { [row['opa'], row['opb'], row['opc'], row['opd']][int(row['cop'])] }", axis=1)
})

df1_mhqa = pd.DataFrame({
    "query": df_mhqa.apply(lambda row: f"""question: {row['question']}\nOptions:\noption1: {row['option1']}\noption2: {row['option2']}\noption3: {row['option3']}\noption4: {row['option4']}""", axis=1),
    "answers": df_mhqa.apply(lambda row: f"""answer: {row['correct_answer']}""", axis=1)
})

#Optional: Save the new DataFrame
# Save individual dataframes
#df1_angst.to_csv('/home/sracha/formatted_angst.csv', index=False)
#df1_medmcqa.to_csv('/home/sracha/formatted_medmcqa.csv', index=False)
#df1_mhqa.to_csv('/home/sracha/formatted_mhqa.csv', index=False)

train_df_angst, valid_df_angst = train_test_split(df1_angst, test_size=0.15, random_state=42)
train_df_medmcqa, valid_df_medmcqa = train_test_split(df1_medmcqa, test_size=0.15, random_state=42)
train_df_mhqa, valid_df_mhqa = train_test_split(df1_mhqa, test_size=0.15, random_state=42)

dataset_angst = Dataset.from_pandas(train_df_angst)
valid_dataset_angst = Dataset.from_pandas(valid_df_angst)
dataset_medmcqa = Dataset.from_pandas(train_df_medmcqa)
valid_dataset_medmcqa = Dataset.from_pandas(valid_df_medmcqa)
dataset_mhqa = Dataset.from_pandas(train_df_mhqa)
valid_dataset_mhqa = Dataset.from_pandas(valid_df_mhqa)

# Print a sample row to verify
#print(df1.head(1).to_string(index=False))

#df = df.rename(columns={"question": "query", "answer": "answers"})  # Replace with actual column names

# Combine all formatted data
df1 = pd.concat([df1_angst, df1_medmcqa, df1_mhqa], ignore_index=True)
df1.to_csv('/home/sracha/proper_kg_project/llm_training/dataset/finetuning_mixtures/combined_formatted-1.csv', index=False)

# Split for training/validation
#train_df, valid_df = train_test_split(df1, test_size=0.15, random_state=42)
#dataset = Dataset.from_pandas(train_df)
#valid_dataset = Dataset.from_pandas(valid_df)
#--------------------

# Initialize the tokenizer with the chat template and mapping
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3", 
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True,        # Maps <|im_end|> to <|eot_id|> instead
)

# Apply individual formatting
def format_angst(examples):
    convos = []
    for query, answers in zip(examples['query'], examples['answers']):
        convos.append([
            {"role": "system", "content": "You are a medical expert. You are given a user post. Your task is to decide whether the user has depression and anxiety or not based on the signs and symptoms. Output 1 for yes and 0 for no.\nOutput in the format:\ndepression: <0 or 1>\nanxiety: <0 or 1>"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answers}
        ])
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convos]
    return {"text": texts}

def format_medmcqa(examples):
    convos = []
    for query, answers in zip(examples['query'], examples['answers']):
        convos.append([
            {"role": "system", "content": "You are a medical expert. You are given a question and it's four candidate options. Choose the correct answer from the options provided."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answers}
        ])
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convos]
    return {"text": texts}

def format_mhqa(examples):
    convos = []
    for query, answers in zip(examples['query'], examples['answers']):
        convos.append([
            {"role": "system", "content": "You are a medical expert. You are given a question and it's four candidate options. Choose the correct answer from the options provided."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answers}
        ])
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convos]
    return {"text": texts}


# Apply formatting
#dataset = dataset.map(formatting_prompts_func, batched=True)
#valid_dataset = valid_dataset.map(formatting_prompts_func, batched=True)

dataset_angst= dataset_angst.map(format_angst, batched=True)
valid_dataset_angst = valid_dataset_angst.map(format_angst, batched=True)
dataset_medmcqa = dataset_medmcqa.map(format_medmcqa, batched=True)
valid_dataset_medmcqa = valid_dataset_medmcqa.map(format_medmcqa, batched=True)
dataset_mhqa = dataset_mhqa.map(format_mhqa, batched=True)
valid_dataset_mhqa = valid_dataset_mhqa.map(format_mhqa, batched=True)



# Combine all formatted training datasets
dataset = concatenate_datasets([dataset_angst, dataset_medmcqa, dataset_mhqa])

# Combine all formatted validation datasets
valid_dataset = concatenate_datasets([valid_dataset_angst, valid_dataset_medmcqa, valid_dataset_mhqa])


# Training arguments
args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="wandb",
    logging_steps=1,
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=1600,
    load_best_model_at_end=True,
    save_only_model=False,
    eval_strategy="steps",
    eval_steps=200,
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=valid_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args
)

# GPU memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
print(f"GPU: {gpu_stats.name}. Max memory: {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Training
trainer_stats = unsloth_train(trainer)
#trainer_stats = trainer.train()
print(trainer_stats)
wandb.finish()

# Final memory stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round((used_memory / max_memory) * 100, 3)
lora_percentage = round((used_memory_for_lora / max_memory) * 100, 3)
print(f"Training time: {trainer_stats.metrics['train_runtime']} sec ({round(trainer_stats.metrics['train_runtime']/60, 2)} min).")
print(f"Peak reserved memory: {used_memory} GB.")
print(f"Training memory usage: {used_memory_for_lora} GB ({lora_percentage} % of max).")

# Save model locally
model.save_pretrained("DiagMent-LLM-1")
tokenizer.save_pretrained("DiagMent-LLM-1")

# Push model to Hugging Face Hub
# model.push_to_hub("<hf_username/lora_model_name>", token=hf_token)
# tokenizer.push_to_hub("<hf_username/lora_model_name>", token=hf_token)

# Merge LoRA into 16-bit
model.save_pretrained_merged("DiagMent-LLM-1", tokenizer, save_method="merged_16bit")
#model.push_to_hub_merged("<hf_username/model_name>", tokenizer, save_method="merged_16bit", token=hf_token)