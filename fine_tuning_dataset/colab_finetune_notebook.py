"""
Fine-Tuning a 0.5B Model on Serverlex
Automated script for Google Colab (T4 GPU recommended).
"""
import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# 1. Determine dataset path
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "serverlex_instruct_dataset.json")

if not os.path.exists(dataset_path):
    dataset_path = "serverlex_instruct_dataset.json"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Could not locate dataset at '{dataset_path}'")

print(f"Loading dataset from: {dataset_path}")
with open(dataset_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. Format into instruction-tuning prompt template
def format_prompts(data):
    formatted = []
    for item in data:
        if item.get("input") and str(item["input"]).strip():
            text = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
        else:
            text = f"""### Instruction:
{item['instruction']}

### Response:
{item['output']}"""
        formatted.append({"text": text})
    return formatted

formatted_data = format_prompts(raw_data)
dataset = Dataset.from_list(formatted_data)
print(f"Loaded {len(dataset)} training examples.")

# 3. Model & Tokenizer setup
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading base model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto" if torch.cuda.is_available() else None
)

# 4. LoRA Adapter configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./serverlex-0.5b-checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=20,
    save_strategy="epoch",
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    report_to="none"
)

# 6. SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config
)

print("\nStarting LoRA Fine-Tuning...")
trainer.train()

# 7. Save fine-tuned model
OUTPUT_DIR = "./serverlex-0.5b-final"
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel fine-tuning complete! Saved weights to: {OUTPUT_DIR}")

# 8. Interactive inference helper
def ask_serverlex_expert(instruction, error_log=""):
    if error_log:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{error_log}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=350, temperature=0.2)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

# Sample test run
sample_log = "ValueError: Bundle authenticity verification failed! HMAC mismatch for 'cancer_triage_bundle.joblib'"
print("\n--- Test Model Diagnosis on Serverlex Log ---")
print(ask_serverlex_expert("Analyze this Serverlex error and provide the fix.", sample_log))
