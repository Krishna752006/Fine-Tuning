import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
CACHE_DIR = "./deepseek-coder-1.3b-instruct"
OUTPUT_PATH = "./deepseek-coder-1.3b-instruct-int8"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)

quant_config = TorchAoConfig(quant_type=Int8WeightOnlyConfig())

print("Loading and quantizing to INT8...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    device_map="cpu",
    quantization_config=quant_config,
    trust_remote_code=True,
)

print("Saving...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=False)
tokenizer.save_pretrained(OUTPUT_PATH)