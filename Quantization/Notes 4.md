# Quantization

Quantization means representing model weights using fewer bits.

Example weight:

```text
W = 0.372846
```

Normally, this could be stored as FP32.

```text
FP32  → 32 bits = 4 bytes
FP16  → 16 bits = 2 bytes
BF16  → 16 bits = 2 bytes
INT8  →  8 bits = 1 byte
INT4  →  4 bits = 0.5 byte
```

## Memory

For a model with 596M parameters:

```text
FP32:
596M × 4 bytes ≈ 2.38 GB

FP16/BF16:
596M × 2 bytes ≈ 1.19 GB

INT8:
596M × 1 byte ≈ 596 MB

INT4:
596M × 0.5 byte ≈ 298 MB
```

| Format    | Bits/parameter | Relative memory |
| --------- | -------------: | --------------: |
| FP32      |             32 |            100% |
| FP16/BF16 |             16 |             50% |
| INT8      |              8 |             25% |
| INT4      |              4 |           12.5% |

Quantization does **not** reduce the number of parameters.

```text
Before:
596M parameters → FP16

After:
596M parameters → INT4
```

It only changes how the weights are represented.

## Trade-off

```text
Lower precision
      ↓
Less memory
      +
Potentially faster computation
      ↓
More numerical approximation/error
```

The goal is to reduce memory while keeping model accuracy
as close as possible to the original.

## Quantization Tools

Common approaches/tools include:

```text
TorchAO
bitsandbytes
GPTQ
AWQ
```

## LoRA vs QLoRA

**LoRA:**

```text
FP16/BF16 Base Model
        +
LoRA adapters
        ↓
Train only adapters
```

**QLoRA:**

```text
4-bit Quantized Base Model
        +
LoRA adapters
        ↓
Train only adapters
```

So:

```text
Quantization = reduce weight precision/memory

LoRA = train small adapter parameters

QLoRA = Quantization + LoRA
```

With tools such as Unsloth:

```text
Pretrained model
      ↓
Load in 4-bit
      ↓
Quantized base model
      +
LoRA adapters
      ↓
Train adapters
```
