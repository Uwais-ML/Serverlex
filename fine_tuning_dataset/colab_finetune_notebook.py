# Fine-Tuning a 0.5B Model on Serverlex (Google Colab Notebook)

Run the cells below in Google Colab (select a free T4 GPU under `Runtime > Change runtime type`).

---

### Step 1: Install Required Libraries
```bash
!pip install -q torch transformers datasets trl peft accelerate bitsandbytes serverlex
```

---

### Step 2: Load the Dataset and Format Prompt
```python
import json
from datasets import Dataset

# Load the Serverlex instruction dataset
with open("serverlex_instruct_dataset.json", "r") as f:
    raw_data = json.load(f)

# Convert to Alpaca / Chat template
def format_prompts(batch):
    formatted = []
    for item in batch:
        if item.get("input"):
            prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
        else:
            prompt = f"""### Instruction:
{item['instruction']}

### Response:
{item['output']}"""
        formatted.append({"text": prompt})
    return formatted

dataset = Dataset.from_list(format_prompts(raw_data))
print(f"Loaded {len(dataset)} training examples")
print(dataset[0]["text"][:250])
```

---

### Step 3: Load 0.5B Model (e.g. Qwen2.5-0.5B or SmolLM-360M/0.5B)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # High performance 0.5B model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

# Apply LoRA for fast, lightweight fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

---

### Step 4: Run SFT Training
```python
training_args = TrainingArguments(
    output_dir="./serverlex-0.5b-expert",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    logging_steps=5,
    save_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config
)

trainer.train()
trainer.model.save_pretrained("./serverlex-0.5b-final")
tokenizer.save_pretrained("./serverlex-0.5b-final")
print("✅ Fine-tuning completed and saved to ./serverlex-0.5b-final")
```

---

### Step 5: Test the Fine-Tuned Model on Serverlex Logs
```python
def ask_serverlex_expert(instruction, error_log=""):
    if error_log:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{error_log}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.2)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

# Test diagnosis with a log
sample_log = "ValueError: Bundle authenticity verification failed! HMAC mismatch for 'cancer_triage_bundle.joblib'"
print(ask_serverlex_expert("Analyze this Serverlex log and output the fix.", sample_log))
```
