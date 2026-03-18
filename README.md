# GPT-2 (124M) From Scratch

Pure PyTorch implementation of GPT-2 with KV cache — built to understand 
transformer internals at the tensor level. Every weight shape, every 
attention mask, every residual connection implemented and verified against 
HuggingFace (max numerical diff < 1e-5).

**Why build this:** Production inference optimizations like paged KV cache, 
FP8 KV quantization, and speculative decoding only make sense if you 
understand exactly what K and V tensors are, why they're reusable across 
decode steps, and what their memory footprint looks like at scale. This is 
that foundation.

---

## Verification

Output matches HuggingFace GPT-2 exactly:
```python
# HuggingFace reference
hf_logits = hf_model(input_ids).logits

# This implementation
my_logits = model(input_ids)

max_diff = (hf_logits - my_logits).abs().max().item()
# → 9.31e-06 ✓
```

---

## Architecture
```
Token Embedding  (50257 → 768)
Position Embedding (1024 → 768)
        ↓
12× Transformer Blocks:
    LayerNorm → Multi-Head Attention (12 heads, KV cache) → Residual
    LayerNorm → MLP (768 → 3072 → 768) → Residual
        ↓
LayerNorm → LM Head (768 → 50257, weight-tied to embedding)
```

| Parameter | Value |
|-----------|-------|
| Total params | 124M |
| Context length | 1024 tokens |
| Attention heads | 12 |
| Hidden dim | 768 |
| KV cache | Implemented |

---

## KV Cache Implementation

The core insight: K and V tensors for previously seen tokens never 
change. Only the new token's K and V need to be computed each step.
```
Without cache — recomputes all tokens every step, O(n²):
Step 1: [The, cat, sat]       → full attention over 3 tokens
Step 2: [The, cat, sat, on]   → full attention over 4 tokens
Step 3: [The, cat, sat, on, the] → full attention over 5 tokens

With cache — computes only new token, O(n):
Step 1: [The, cat, sat]  → compute K,V for 3 tokens, store in cache
Step 2: [on]             → compute K,V for 1 token, concat with cache
Step 3: [the]            → compute K,V for 1 token, concat with cache
```

**Connection to production systems:**  
This is exactly why paged KV cache (used in vLLM and TensorRT-LLM) 
matters at scale. As sequences grow, KV cache memory grows linearly 
per token per layer. For a 32B model with 64 layers and 8 KV heads 
at FP16, each token costs ~1MB of KV cache. At 10,000 concurrent 
users with 2K context, that's ~20TB — impossible to keep in GPU HBM. 
Paged KV cache solves this with virtual memory-style block allocation. 
FP8 KV cache halves that footprint. Understanding the tensor mechanics 
here is what makes those production tradeoffs legible.

---

## Key Implementation Details

**Weight tying**  
LM head shares weights with the token embedding matrix. HuggingFace 
stores this as a single tensor — loading requires recognizing the tie 
and not double-counting parameters.

**HuggingFace Conv1D transpose**  
HuggingFace GPT-2 uses `Conv1D` for QKV projections with transposed 
weight layout vs standard `nn.Linear`. Weight loading requires an 
explicit `.T` transpose or outputs diverge silently.

**Causal mask**  
Implemented as additive mask (`-inf` for future positions) rather than 
multiplicative. Additive mask composes correctly with softmax — 
`exp(-inf) = 0` cleanly zeros out future attention weights.

---

## Usage
```python
# Load with pretrained weights
model = GPT2(config)
model = load_hf_weights(model, "gpt2")

# Generate with KV cache
output = generate(model, "The meaning of life is", max_new_tokens=50)
print(output)
```

---

## Requirements
```
torch
transformers  # tokenizer and weight loading only
```

---

## File Structure
```
GPT_2.ipynb
├── Cell 1: Config and hyperparameters
├── Cell 2: CausalSelfAttention with KV cache
├── Cell 3: MLP block
├── Cell 4: Transformer block
├── Cell 5: Full GPT-2 model
├── Cell 6: HuggingFace weight loading
├── Cell 7: Text generation loop
└── Cell 8: Numerical verification vs HuggingFace
```

---

## References

- [Attention Is All You Need — Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [GPT-2 — Radford et al.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace GPT-2](https://huggingface.co/gpt2)

---

## Related Projects

This implementation informed the inference optimization work below:

- [Llama-3.1-8B on H100 — 1,700+ tok/s with TRT-LLM FP8](https://github.com/IneshReddy249/LLAMA-TRT-OPTIMIZATION)
- [Speculative Decoding — 2.26× latency reduction on Qwen 2.5](https://github.com/IneshReddy249/SPECULATIVE_DECODING)
- [Mixtral 8x7B MoE — 57→120 tok/s on dual A100s](https://github.com/IneshReddy249/vLLM-mixtral-MoE-optimization)

---

## Author

**Inesh Reddy Chappidi** — LLM Inference & Systems Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Inesh_Reddy-0077B5?logo=linkedin)](https://www.linkedin.com/in/inesh-reddy)
[![GitHub](https://img.shields.io/badge/GitHub-IneshReddy249-181717?logo=github)](https://github.com/IneshReddy249)
