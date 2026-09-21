from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="float32"
)

print("Model loaded!")
# print("Parameters:", sum(p.numel() for p in model.parameters()))

# prompt = "Explain what machine learning is in simple terms."

# inputs = tokenizer(prompt, return_tensors="pt")

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=100
# )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

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