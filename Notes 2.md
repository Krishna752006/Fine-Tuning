pip install torch transformers datasets peft accelerate trl

Load Qwen3-0.6B
```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="float32"
)

print("Model loaded!")
print("Parameters:", sum(p.numel() for p in model.parameters()))
```

```
prompt = "Explain what machine learning is in simple terms."

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Add LoRA
```
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
```

| Module   | Meaning           |
| -------- | ----------------- |
| `q_proj` | Query projection  |
| `k_proj` | Key projection    |
| `v_proj` | Value projection  |
| `o_proj` | Output projection |

Attention
   │
   ├── Q → q_proj
   ├── K → k_proj
   ├── V → v_proj
   └── output → o_proj

We're telling LoRA:
"Put the trainable adapters here."

r=8, means the LoRA matrices have rank 8.

r = 4    → smaller adapter
r = 8    → small adapter
r = 16   → larger adapter
r = 32   → larger still

Higher r means:

more trainable parameters
potentially more capacity
more memory
potentially more ability to learn the task

lora_alpha=16

This controls the scaling of the LoRA contribution.

LoRA contribution ∝ alpha / r
We used:

alpha = 16
r     = 8

alpha / r = 2

Just remember:

r controls adapter capacity; alpha controls how strongly the adapter contribution is scaled.

output = W·x  +  (alpha / r) · B·A·x

A is r × d_in and B is d_out × r, so B·A has the same shape as W but is only "rank r".
Only A and B are trained. W never changes.

The config, line by line:

r=8: the rank, i.e. the size of the bottleneck. Higher means more capacity and more trainable parameters. 8 is a common, cheap default.
lora_alpha=16: a scaling factor. The LoRA output is multiplied by alpha / r = 16/8 = 2. It controls how strongly the adapter influences the output.
lora_dropout=0.05: randomly drops 5% of the inputs to the LoRA branch during training, which is light regularization against overfitting.
target_modules=[q_proj, k_proj, v_proj, o_proj]: which layers get adapters. These are the four linear layers in every attention block (query, key, value, output). The MLP layers (gate_proj, up_proj, down_proj) are left alone.
bias="none": don't train any bias terms.
task_type="CAUSAL_LM": tells PEFT this is next-token-prediction, so it wraps the model appropriately.

                 ┌── W ──────────────┐
                 │   frozen           │
Input ───────────┤                    ├──► output
                 │                    │
                 └── LoRA A → LoRA B ┘
                       trainable

Model loaded! trainable params: 2,293,760 || all params: 598,343,680 || trainable%: 0.3834

Create a file called:

train.jsonl
with eg:
{"text":"### Question:\nWhat is Python?\n### Answer:\nPython is a programming language known for its simple and readable syntax."}
{"text":"### Question:\nWhat is a database?\n### Answer:\nA database is a system used to store, organize, and retrieve data."}

train.py
```
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# 1. Load model
model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="float32"
)

# 2. Add LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# 3. Load dataset
dataset = load_dataset(
    "json",
    data_files="train.jsonl",
    split="train"
)

# 4. Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=["text"]
)

# 5. Training configuration
training_args = TrainingArguments(
    output_dir="./qwen3-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
)

# 7. Train!
trainer.train()

# 8. Save LoRA adapter
model.save_pretrained("./qwen3-lora")
tokenizer.save_pretrained("./qwen3-lora")

print("Training complete!")
```

num_train_epochs=3: three passes over the data.
per_device_train_batch_size=1, gradient_accumulation_steps=1: one example per step, and weights update after every example. This is noisy but fine for a small experiment. Raising accumulation (e.g. 8) simulates a bigger batch.
learning_rate=2e-4: much higher than full fine-tuning would use, but standard for LoRA since only small adapter matrices are learning.
logging_steps=1: print the loss every step.
save_strategy="no": no checkpoints during training.
report_to="none": disables W&B/TensorBoard.

DataCollatorForLanguageModeling(mlm=False), which:

Pads examples in a batch to the same length.
Creates labels as a copy of input_ids, with padding positions set to -100 so they're ignored in the loss.

mlm=False means causal LM (predict the next token) rather than BERT-style masked LM. The model shifts the labels internally, so each token is trained to predict the one after it.

eg:
```
{'loss': '0.7719', 'grad_norm': '4.159', 'learning_rate': '6.667e-06', 'epoch': '2.95'} {'loss': '1.806', 'grad_norm': '4.033', 'learning_rate': '3.333e-06', 'epoch': '3'} {'train_runtime': '192.1', 'train_samples_per_second': '0.312', 'train_steps_per_second': '0.312', 'train_loss': '1.346', 'epoch': '3'} 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [03:12<00:00, 3.20s/it] Training complete!
```

evaluate.py
```
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "Qwen/Qwen3-0.6B-Base"
adapter_path = "./qwen3-lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load original model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype="float32"
)

# Load LoRA adapter onto the base model
model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

prompt = """### Question:
What is Python?
### Answer:"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Today
Qwen3-0.6B
     ↓
PEFT
     ↓
LoRA
     ↓
10 examples
     ↓
CPU
     ↓
2.29M trainable parameters

Kaggle:
Qwen3-0.6B / larger model
     ↓
Unsloth
     ↓
LoRA / QLoRA
     ↓
thousands of examples
     ↓
GPU
     ↓
much faster training

LoRA is a fine-tuning method. The adapter is the thing LoRA creates and trains.
Fine-tuning
    │
    ├── Full fine-tuning
    │      └── Update the original model weights
    │
    └── Parameter-efficient fine-tuning (PEFT)
           │
           ├── LoRA
           │      └── Train LoRA adapters
           │
           ├── Prefix tuning
           └── Other methods

