### LoRA Target Selection

**Text → Text models**

* Inspect the Transformer architecture.
* Common LoRA targets are the attention projection layers:

  * `q_proj`
  * `k_proj`
  * `v_proj`
  * `o_proj` / `out_proj`
* MLP layers can also be targeted:

  * `up_proj`
  * `down_proj`
  * `gate_proj`
* Attention projections are a common starting point.

**Image + Text → Text models**

* Don't blindly use `q/k/v/o`.
* First identify **which part of the multimodal model should be adapted**.
* Common dataset structure:

```text
dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── img003.jpg
└── captions.json
```

```json
[
  {
    "image": "img001.jpg",
    "text": "A green circuit board with a graphics card."
  },
  {
    "image": "img002.jpg",
    "text": "A dog sitting on grass."
  }
]
```

**Key idea:**
For text models, attention projections are a standard LoRA starting point. For multimodal models, first determine **which component needs adaptation**, then choose the appropriate layers.