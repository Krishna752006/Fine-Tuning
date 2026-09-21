import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_PATH = "./deepseek-coder-1.3b-instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_PATH
)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    cache_dir=MODEL_PATH
)

print("\nModel loaded successfully.")
print(f"Parameter count: {model.num_parameters():,}")