import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TorchAoConfig
)
from torchao.quantization import Int8WeightOnlyConfig

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Same cache directory you used during download
CACHE_DIR = "./deepseek-coder-1.3b-instruct"

# Where the quantized model was saved
OUTPUT_PATH = "./deepseek-coder-1.3b-instruct-int8"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    OUTPUT_PATH,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

print("Loading INT8 model...")

quant_config = TorchAoConfig(
    quant_type=Int8WeightOnlyConfig()
)

model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_PATH,
    dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    device_map="cpu",
    quantization_config=quant_config,
    trust_remote_code=True
)

print("INT8 model loaded.")

messages = [
    {
        "role": "user",
        "content": "Write a Python function that checks whether a string is a palindrome."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# CPU, so no .to("cuda")
inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

prompt_length = inputs["input_ids"].shape[-1]

new_tokens = outputs[0][prompt_length:]

response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("\n===== INT8 RESPONSE =====\n")
print(response)