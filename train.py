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