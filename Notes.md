# LoRA and QLoRA

## 1. The Problem: Fine-Tuning Large Language Models

Suppose we have a **7-billion-parameter (7B) language model**.

In traditional **full fine-tuning**, we update all 7B parameters.

```text
              7B Base Model
              ┌─────────────┐
Input ───────►│   Trainable │──────► Output
              │     7B      │
              └─────────────┘
```

This can be extremely expensive because training requires memory for more than just the model weights.

We need memory for:

* Model weights
* Gradients
* Optimizer states
* Activations

So even though a model may "only" have 7B parameters, full fine-tuning can require **far more GPU memory** than simply storing the 7B weights.

This motivates **Parameter-Efficient Fine-Tuning (PEFT)**.

# 2. LoRA

**LoRA = Low-Rank Adaptation**

The main idea is simple:

> **Keep the original model frozen and train a small set of additional parameters.**

Instead of modifying all 7B parameters:

```text
7B Base Model
      ↓
Freeze ❄️
```

we add small trainable matrices.

```text
                  ┌─────────────────┐
                  │   7B Base Model │
Input ───────────►│      Frozen     │────┐
                  └─────────────────┘    │
                                         +──► Output
                  ┌─────────────────┐    │
Input ───────────►│   LoRA Adapter  │────┘
                  │     Trainable   │
                  └─────────────────┘
```

The base model remains unchanged. Only the LoRA parameters are trained.

# 3. Why is LoRA called "Low-Rank"?

Suppose a model contains a large weight matrix:

$$
W \in \mathbb{R}^{d \times k}
$$

Full fine-tuning would learn an update:

$$
W' = W + \Delta W
$$

where:

$$
\Delta W \in \mathbb{R}^{d \times k}
$$

The problem is that $\Delta W$ can be very large.

LoRA makes an assumption:

> We don't need to learn the entire update matrix. We can approximate it using two much smaller matrices.

Instead of directly learning:

$$
\Delta W
$$

LoRA represents it as:

$$
\Delta W = BA
$$

where:

$$
A \in \mathbb{R}^{r \times k}
$$

and

$$
B \in \mathbb{R}^{d \times r}
$$

Here, **$r$ is the rank**, and it is much smaller than $d$ and $k$.

Therefore:

$$
W' = W + BA
$$

The original $W$ is frozen.

Only $A$ and $B$ are trained.

# 4. How Much Does LoRA Add?

There is **no fixed number** such as "LoRA always adds 200M parameters."

The number depends on things such as:

* Model architecture
* Which layers receive LoRA
* LoRA rank $r$
* Dimensions of the target matrices

A LoRA adapter could contain:

```text
1M parameters
10M parameters
50M parameters
200M+ parameters
```

The important idea is:

> **LoRA trains far fewer parameters than full fine-tuning.**

For example:

```text
Full fine-tuning

   7B parameters
🔥All trainable 🔥


LoRA

  7B base parameters
     ❄️Frozen❄️ 
        +
  Small LoRA adapter
    🔥 Trainable
```

# 5. Why Is LoRA Useful?

The same base model can be adapted to many different tasks.

```text
                    ┌── Coding LoRA
                    │
                    ├── Medical LoRA
                    │
Base LLM ───────────┼── Legal LoRA
                    │
                    └── SQL LoRA
```

Instead of creating a completely new copy of the model for every task, we can keep one frozen base model and store relatively small adapters.

This can make fine-tuning and deployment more efficient.

# 6. Quantization

LoRA reduces the number of **trainable parameters**.

Quantization addresses a different problem:

> **Reduce the number of bits used to represent model weights.**

For example:

```text
FP32 → INT8 → INT4
```

Fewer bits per parameter means less memory is required to store the model.

# 7. Example: Model Weight Precision

Suppose a model contains weights such as:

```text
-1.72
-0.91
-0.24
 0.15
 0.63
 1.84
```

With FP32, each value uses 32 bits.

Quantization maps these values into a smaller representation.

Conceptually:

```text
-1.72 → -118
-0.91 →  -62
-0.24 →  -16
 0.15 →   10
 0.63 →   43
 1.84 →  127
```

A simplified quantization equation is:

$$
q = \operatorname{round}\left(\frac{x}{s}\right)
$$

where:

* $x$ = original value
* $q$ = quantized value
* $s$ = scale factor

To approximately recover the original value:

$$
x \approx q \times s
$$

Real quantization schemes can be more sophisticated than this simplified example.

# 8. Why Does Quantization Save Memory?

Each parameter requires fewer bits.

For a 7B-parameter model, ignoring additional overhead:

| Representation | Bits / parameter | Approx. weight memory |
| -------------- | ---------------: | --------------------: |
| FP32           |               32 |                 28 GB |
| FP16           |               16 |                 14 GB |
| INT8           |                8 |                  7 GB |
| INT4           |                4 |                3.5 GB |

The calculation is simply:

$$
\text{Memory} \approx
\frac{\text{Number of parameters} \times \text{Bits per parameter}}
{8}
$$

followed by conversion from bytes to GB.

### Important

These numbers represent **theoretical weight storage**.

Actual GPU memory usage can be higher because of:

* Quantization metadata
* Temporary tensors
* Activations
* Gradients
* Optimizer states
* Framework overhead

# 9. Dtype Conversion vs Quantization

These concepts are related, but they are not exactly the same.

### Simple dtype conversion

For example:

```text
FP64 → FP32
```

You are changing the numerical representation from 64-bit floating point to 32-bit floating point.

This reduces storage:

```text
64 bits/value
      ↓
32 bits/value
```

### Quantization

Quantization usually means mapping values into a lower-precision representation, often involving **integer or specialized low-bit formats**.

For example:

```text
FP32 → INT8
FP32 → INT4
```

So:

> **Changing dtype is not automatically the same thing as quantization.**

Quantization is a more specific form of reducing numerical precision/representation.

# 10. QLoRA

Now we can combine the two ideas.

**QLoRA = Quantized LoRA**

It combines:

```text
Quantization
      +
LoRA
```

The basic idea is:

1. Quantize the pretrained base model.
2. Keep the quantized base model frozen.
3. Add LoRA adapters.
4. Train only the LoRA parameters.

The key difference from standard LoRA is that the **base model is stored in a quantized format**.

# 11. LoRA vs QLoRA

|                         | LoRA                              | QLoRA                                        |
| ----------------------- | --------------------------------- | -------------------------------------------- |
| Base model              | Full precision / chosen precision | Quantized                                    |
| Base model trainable?   | ❌ No                              | ❌ No                                         |
| LoRA adapter trainable? | ✅ Yes                             | ✅ Yes                                        |
| Uses low-rank adapters? | ✅                                 | ✅                                            |
| Uses quantization?      | Not necessarily                   | ✅                                            |
| Memory usage            | Lower than full fine-tuning       | Typically even lower                         |
| Main idea               | Train fewer parameters            | Train fewer parameters + quantize base model |

# 12. Full Fine-Tuning vs LoRA vs QLoRA

```text
                    Fine-Tuning
                         │
          ┌──────────────┴──────────────┐
          │                             │
   Full Fine-Tuning                    PEFT
          │                             │
   Update all weights                  LoRA
                                        │
                              Add small trainable
                                  low-rank matrices
                                        │
                                      QLoRA
                                        │
                              Quantized base model
                              + LoRA adapters
```

Another way to see it:

| Method           | Base Model            | What Gets Trained?   | Main Memory Saving                          |
| ---------------- | --------------------- | -------------------- | ------------------------------------------- |
| Full Fine-Tuning | Full precision        | All model parameters | None                                        |
| LoRA             | Full/chosen precision | Small LoRA adapters  | Fewer trainable parameters                  |
| QLoRA            | Quantized             | Small LoRA adapters  | Quantized base + fewer trainable parameters |

# 13. An Intuitive Analogy

Imagine the original model is a huge book.

### Full Fine-Tuning

You rewrite the entire book.

```text
📕 Original Book
        ↓
Rewrite everything
```

You need to modify every page.

### LoRA

You keep the original book and attach a set of correction notes.

```text
📕 Original Book
+
📝 Adaptation Notes
```

The original book stays unchanged.

The notes contain the task-specific adaptation.

### QLoRA

First, compress the book.

Then attach the same type of adaptation notes.

```text
📕 Compressed Original Book
+
📝 Adaptation Notes
```

So:

> **LoRA reduces how much you train.**

> **Quantization reduces how much memory the base model requires.**

> **QLoRA combines both ideas.**

# 14. The Most Important Distinction

A common source of confusion is thinking that LoRA and quantization solve the same problem.

They don't.

### LoRA answers:

> **"How can I fine-tune a large model without updating all of its parameters?"**

Answer:

**Train small low-rank adapter matrices while keeping the base model frozen.**

### Quantization answers:

> **"How can I represent the model using less memory?"**

Answer:

**Use fewer bits to represent the model's weights.**

### QLoRA answers:

> **"Can I combine both?"**

Answer:

**Yes. Quantize the base model and train LoRA adapters on top of it.**

# 15. A Simple Mental Model

Remember these three lines:

```text
Full Fine-Tuning
→ Train everything


LoRA
→ Freeze the model + train small adapters


QLoRA
→ Quantize the model + freeze it + train small adapters
```

Or even shorter:

$$
\boxed{\text{LoRA = fewer trainable parameters}}
$$

$$
\boxed{\text{Quantization = fewer bits per parameter}}
$$

$$
\boxed{\text{QLoRA = quantized base model + LoRA}}
$$

## One-Sentence Summary

> **LoRA makes fine-tuning efficient by training small low-rank adapters, while QLoRA additionally quantizes the frozen base model to reduce memory usage.**
