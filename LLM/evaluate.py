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
Explain me what is Fine Tuning?
### Answer:"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))