import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
CACHE_DIR = "./deepseek-coder-1.3b-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

print("Model loaded.")

messages = [
    {
        "role": "user",
        "content": "Write a Python function that checks whether a string is a palindrome."
    }
]

# Explicitly request a dict of PyTorch tensors (input_ids, attention_mask)
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Move inputs to the model device if needed
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Slice using input_ids tensor length
prompt_length = inputs["input_ids"].shape[-1]
new_tokens = outputs[0][prompt_length:]

response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("\n===== BASELINE RESPONSE =====\n")
print(response)