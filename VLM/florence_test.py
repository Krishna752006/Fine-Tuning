import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


# -------------------------
# 1. Configuration
# -------------------------

model_name = "microsoft/Florence-2-large"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Device:", device)
print("Dtype:", dtype)


# -------------------------
# 2. Load processor
# -------------------------

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)


# -------------------------
# 3. Load model
# -------------------------

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    trust_remote_code=True
).to(device)

print("Model loaded!")


# -------------------------
# 4. Load an image
# -------------------------

image = Image.open("test_image.png").convert("RGB")

print("Image size:", image.size)


# -------------------------
# 5. Give Florence a task
# -------------------------

prompt = "<DETAILED_CAPTION>"

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# -------------------------
# 6. Generate output
# -------------------------

with torch.no_grad():

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=100,
        num_beams=3,
        do_sample=False
    )


# -------------------------
# 7. Decode
# -------------------------

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=False
)[0]


# -------------------------
# 8. Parse Florence output
# -------------------------

result = processor.post_process_generation(
    generated_text,
    task=prompt,
    image_size=image.size
)

print("\nFlorence output:")
print(result)

print("\n" + "=" * 80)
print("DECODER ATTENTION PARAMETERS")
print("=" * 80)

for name, module in model.named_modules():
    if (
        "language_model.model.decoder" in name
        and (
            "self_attn" in name
            or "encoder_attn" in name
        )
        and isinstance(module, torch.nn.Linear)
    ):
        print(
            f"{name:100s} "
            f"in={module.in_features:5d} "
            f"out={module.out_features:5d}"
        )