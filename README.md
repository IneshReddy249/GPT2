# GPT-2 (124M) From Scratch

Pure PyTorch implementation of GPT-2 with KV cache for efficient inference.

## What's Inside

- **Complete GPT-2 architecture** built from scratch (no HuggingFace model code)
- **KV cache implementation** for fast autoregressive generation
- **HuggingFace weight loading** — uses pretrained GPT-2 weights
- **Verified output** — matches HuggingFace exactly (max diff < 1e-5)

## Architecture
```
Token Embedding (50257 → 768)
Position Embedding (1024 → 768)
        ↓
12x Transformer Blocks:
    LayerNorm → Multi-Head Attention (12 heads) → Residual
    LayerNorm → MLP (768 → 3072 → 768) → Residual
        ↓
LayerNorm → LM Head (768 → 50257)
```

## Key Features

| Feature | Description |
|---------|-------------|
| Parameters | 124M (matches GPT-2 small) |
| Context Length | 1024 tokens |
| Attention Heads | 12 |
| Hidden Dim | 768 |
| KV Cache | ✅ Implemented |

## KV Cache Speedup

Without cache: recompute all tokens every step → O(n²)

With cache: compute only new token, reuse old K,V → O(n)
```
First pass:  [The, cat, sat] → cache K,V for 3 tokens
Second pass: [on]            → compute 1 token, concat with cache
Third pass:  [the]           → compute 1 token, concat with cache
```

## Usage
```python
# Load model with pretrained weights
model = GPT2(config)
model = load_hf_weights(model, "gpt2")

# Generate text
output = generate(model, "The meaning of life is", max_new_tokens=50)
print(output)
```

## Requirements
```
torch
transformers (for tokenizer and weight loading only)
```

## File Structure
```
GPT_2.ipynb
├── Cell 1: Config
├── Cell 2: CausalSelfAttention (with KV cache)
├── Cell 3: MLP
├── Cell 4: Block
├── Cell 5: GPT2
├── Cell 6: Weight loading
├── Cell 7: Generation
└── Cell 8: Verification
```

## What I Learned

- Every tensor shape in transformer architecture
- Why KV cache works (K,V for old tokens don't change)
- How attention mask prevents looking at future tokens
- Weight tying between embedding and LM head
- HuggingFace uses Conv1D (need to transpose weights)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace GPT-2](https://huggingface.co/gpt2)
