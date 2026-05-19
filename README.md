# llm.c

A complete, from-scratch Language Model in pure C - no Python, no PyTorch, no dependencies except OpenBLAS.

Implements a modern Transformer with every optimisation needed to train on real data: OpenBLAS matrix ops, AVX2+FMA kernels, OpenMP parallelism, KV-cache inference, and a Nested Learning extension (persistent memory + multi-rate optimiser).

---

## Quick start

```bash
# Install the only dependency
sudo apt install libopenblas-dev     # Ubuntu/Debian
# brew install openblas              # macOS

# Build everything
make all

# Full pipeline: vocabulary → data → train → generate
./build/build_vocab  corpus.txt 4096 data/tokenizer.vocab
./build/prepare_data corpus.txt data/tokenizer.vocab
./build/train
./build/generate checkpoints/ckpt_XXXXX.bin data/tokenizer.vocab \
    --prompt "Once upon a time" --temp 0.8
```

---

## Project layout

```
llm.c/
├── include/
│   ├── config.h        Model and training hyper-parameters + presets
│   ├── matrix.h        Core float matrix type, all math primitives
│   ├── simd.h          AVX2/FMA/AVX-512 capability detection
│   ├── attention.h     Multi-head / Grouped Query Attention, RoPE, KV-cache
│   ├── transformer.h   Full model: blocks, MoE, forward/backward
│   ├── tokenizer.h     Byte-Pair Encoding tokenizer
│   ├── inference.h     KV-cache autoregressive engine + samplers
│   └── nested.h        Nested Learning: CMB memory + multi-rate optimiser
│
├── src/
│   ├── matrix.c        OpenBLAS sgemm + AVX2 fallbacks + polynomial expf
│   ├── attention.c     Fully sgemm-accelerated attention fwd + bwd
│   ├── transformer.c   Model init/save/load, block fwd, backward pass
│   ├── tokenizer.c     O(1) hash-map encode, priority-queue BPE merge
│   ├── inference.c     Single-token forward, top-k/top-p sampling
│   ├── nested.c        CMB forward/backward, deep optimiser
│   ├── train.c         AdamW, LR schedule, gradient clipping, train loop
│   └── main.c          CLI entry point, auto-config from corpus metadata
│
├── tools/
│   ├── build_vocab.c   Fast incremental BPE trainer (heap + linked-list)
│   ├── prepare_data.c  Tokenise corpus → train.bin / val.bin
│   └── train_nested.c  Nested Learning fine-tuning on a trained checkpoint
│
└── tests/
    ├── test_matrix.c       Unit tests - matrix ops (37 tests)
    ├── test_tokenizer.c    Unit tests - BPE encode/decode
    ├── bench_matmul.c      sgemm throughput benchmark
    └── bench_inference.c   Tokens/second benchmark
```

---

## Build targets

```bash
make all          # train + generate + build_vocab + prepare_data + train_nested
make train        # training binary only
make tools        # all tool binaries
make test         # run 37 unit tests
make bench        # throughput benchmarks
make debug        # ASAN + UBSan debug build
make clean
```

**Requirements:** GCC with C11, OpenBLAS (`-lopenblas`). OpenMP is used automatically when available. Everything else is in the standard C library.

---

## Full pipeline

### 1. Build a vocabulary

```bash
./build/build_vocab corpus.txt <vocab_size> data/tokenizer.vocab
```

| Corpus size | Recommended vocab_size |
|-------------|------------------------|
| < 1MB       | 512 – 1024             |
| 1 – 10MB    | 2048 – 4096            |
| > 10MB      | 8000 – 32000           |

Uses incremental BPE with a max-heap and doubly-linked corpus - 10–50× faster than naïve rescanning.

### 2. Tokenise the corpus

```bash
./build/prepare_data corpus.txt data/tokenizer.vocab [options]

Options:
  --train_split 0.9     fraction for training (default 0.9)
  --out_dir     data/   output directory
  --chunk_size  256     sequence length hint
  --add_bos 1           prepend <bos> to each document
  --add_eos 1           append  <eos> to each document
```

Writes `data/train.bin`, `data/val.bin`, `data/meta.txt`.

### 3. Train

```bash
./build/train [options]

Options:
  --tiny              force tiny model  (~1M params)
  --small             force small model (~7M params)
  --steps  N          override max training steps
  --lr     F          override peak learning rate
  --checkpoint PATH   resume from a saved checkpoint
```

`config_auto` reads `data/meta.txt` and automatically selects:

| Corpus tokens | Model    | Params | Typical steps |
|--------------|----------|--------|---------------|
| < 500K       | tiny     | 0.9M   | 2 000         |
| 500K – 5M    | small    | 6.6M   | ~50 000       |
| > 5M         | default  | 22M    | ~25 000       |

Checkpoints are saved to `checkpoints/ckpt_NNNNN.bin`. A final checkpoint is always written when training completes.

**Reading the training log:**

```
step    loss      smooth    val       lr        ms/step
0       8.86      8.86      ---       1.6e-06   2200    gnorm=1014
...
500     3.66      3.82      ---       2.9e-04   2100    gnorm=92
  [val] step=500   val_loss=3.86  train_smooth=3.82  gap=0.04
  [ckpt] saved: checkpoints/ckpt_00500.bin
```

| Column  | Meaning |
|---------|---------|
| `loss`  | Raw cross-entropy this step (noisy) |
| `smooth`| Exponential moving average of loss (trend) |
| `val`   | Validation loss (computed every `eval_every` steps) |
| `gap`   | `val - smooth`: positive = overfitting, negative = still learning |
| `gnorm` | Gradient L2 norm before clipping |

**Loss targets for coherent generation:**

| `smooth` loss | Perplexity | Quality |
|--------------|------------|---------|
| > 3.5        | > 33       | Incoherent - keep training |
| 2.5 – 3.5    | 12 – 33    | Improving - getting there |
| 2.0 – 2.5    | 7 – 12     | Good - coherent sentences |
| < 2.0        | < 7        | Strong - domain-specific fluency |

Use the checkpoint where **val_loss is lowest** (not the final one - that is often overfit).

### 4. Generate

```bash
./build/generate <checkpoint.bin> <tokenizer.vocab> [options]

Options:
  --prompt TEXT        input prompt
  --max_new INT        max tokens to generate    (default 256)
  --temp    FLOAT      sampling temperature      (default 0.8)
  --top_k   INT        top-k sampling cutoff     (default 40)
  --top_p   FLOAT      nucleus probability mass  (default 0.9)
  --greedy             argmax decoding (deterministic)
  --seed    INT        RNG seed for reproducibility
  --bench              print tokens/sec after output
  --chat               interactive multi-turn mode
```

Examples:

```bash
# Standard sampling
./build/generate checkpoints/ckpt_05000.bin data/tokenizer.vocab \
    --prompt "The transformer architecture"

# Greedy (fully deterministic, same output every run)
./build/generate checkpoints/ckpt_05000.bin data/tokenizer.vocab \
    --prompt "Once upon a time" --greedy

# Reproducible sampling
./build/generate checkpoints/ckpt_05000.bin data/tokenizer.vocab \
    --prompt "The model" --seed 42

# Interactive chat (KV cache persists between turns)
./build/generate checkpoints/ckpt_05000.bin data/tokenizer.vocab --chat
```

Inside `--chat` mode: `/reset` clears context, `/info` shows position, `exit` quits.

---

## Model configurations

Three presets in `include/config.h`:

```c
ModelConfig mcfg = model_config_tiny();    // 0.9M  - test/debug
ModelConfig mcfg = model_config_small();   // 6.6M  - small corpora
ModelConfig mcfg = model_config_default(); // 22M   - production
```

Or set fields directly:

```c
ModelConfig cfg = model_config_small();
cfg.vocab_size = 8000;   // override after calling preset
cfg.d_model    = 384;
cfg.n_layers   = 8;
```

**Architecture features:**
- Pre-norm Transformer (RMSNorm before each sublayer)
- Grouped Query Attention (GQA) - fewer KV heads than query heads
- Rotary Position Embeddings (RoPE)
- SwiGLU activation in FFN
- Sliding window attention (configurable)
- Weight tying between embedding and LM head (optional)
- Mixture of Experts FFN (set `n_experts > 1`)

---

## Nested Learning

An experimental extension based on Google Research NeurIPS 2025 paper:
*"Nested Learning: Why Transformer 2.0 Might Actually Be About Memory, Not Attention."*

Adds three things on top of a trained model:

1. **Continuum Memory Block (CMB)** - a persistent key-value memory bank inserted after each transformer layer. Unlike attention (which reads the current context window only), the CMB survives between sequences and accumulates knowledge over the entire fine-tuning run.

2. **Multi-rate parameter groups** - early layers and embeddings update at a lower learning rate than later layers. Mirrors the brain's offline consolidation - stable base knowledge changes slowly, recent associations change quickly.

3. **Deep optimiser** - adds a slow-moving third momentum term that tracks long-term gradient trends. Gradients consistently pulling in one direction get a persistent boost; oscillating gradients are damped.

```bash
# Fine-tune an existing checkpoint with Nested Learning
./build/train_nested <checkpoint.bin> <tokenizer.vocab> [options]

Options:
  --data_dir  PATH    where train.bin / val.bin live  (default: data/)
  --steps     INT     fine-tuning steps               (default: 1000)
  --lr        FLOAT   peak learning rate              (default: 1e-4)
  --cmb_slots INT     persistent memory slots         (default: 64)
  --cmb_dim   INT     CMB internal dimension          (default: 128)
  --cmb_freq  INT     CMB update cadence in steps     (default: 4)
  --no_deep_opt       use standard AdamW instead
  --save      PATH    output checkpoint path
  --cmb_init  PATH    load pre-trained CMB weights

# Resume a nested training run (CMB weights persist)
./build/train_nested ckpt_05000.bin tokenizer.vocab \
    --cmb_init checkpoints/nested_ckpt.bin \
    --steps 1000
```

CMB weights are saved separately as `<checkpoint>.cmb.layerN` so the base model checkpoint remains compatible with `./build/generate`.

---

## Performance

Tested on a single CPU core (Ubuntu 24, GCC 13, OpenBLAS 0.3.26):

**Training throughput** (small model, batch=4, seq=256):

| Operation | Before (scalar) | After (OpenBLAS) | Speedup |
|-----------|----------------|-------------------|---------|
| mat_mul   | 13 GFLOP/s     | 121 GFLOP/s       | 9×      |
| Step time | ~15 s/step     | ~0.15 s/step      | **100×** |
| Tokens/s  | ~300 tok/s     | ~15 000 tok/s     | 50×     |

**Inference** (small model, 1 CPU core):

| Sequence length | Tokens/second |
|----------------|---------------|
| 1 token cache  | ~960          |
| 256 token cache| ~800          |
| 512 token cache| ~650          |

**Key optimisations:**
- OpenBLAS `sgemm` for all matrix multiplies in forward and backward pass
- AVX2+FMA polynomial `expf` approximation (5 FLOPs vs 20 for libm)
- Pre-allocated `TrainScratch` - zero `malloc` calls during backward pass
- Attention backward fully sgemm-accelerated (was O(seq²) scalar loops)
- LM head backward via `sgemm` (was scalar triple loop)
- OpenMP outer-tile parallelism (set `OMP_NUM_THREADS=N`)
- Incremental BPE vocabulary trainer with max-heap and doubly-linked corpus

---

## Troubleshooting

**`make: No rule to make target 'tools/train_nested.c'`**
Timestamp skew - run `find . -name "*.c" -o -name "*.h" -o -name "Makefile" | xargs touch` then `make all`.

**`<unk>` tokens flooding generation output**
- Use the `--top_p 0.9 --top_k 40` flags (defaults)
- Use a later checkpoint - early checkpoints haven't learned to avoid UNK
- Target val_loss < 2.8 before generation quality is reliable
- Special tokens (UNK/BOS/PAD) are suppressed from sampling automatically since v2

**Loss stops improving / diverges at end of training**
- The model has memorised the data - use a larger corpus or smaller model
- Check the `gap` column: if `val_loss >> smooth`, you are overfitting
- Use the checkpoint with the **lowest val_loss**, not the final one

**Training is slow**
- Confirm OpenBLAS is linked: `ldd build/train | grep blas`
- Set `OMP_NUM_THREADS` to your core count: `OMP_NUM_THREADS=8 ./build/train`
- Use `--small` for corpora under 5M tokens (right-sizes batch and seq_len)

**`malloc(): corrupted top size` crash**
Likely a `MAX_GROUPS` overflow in `train_nested.c` if you have many layers. The formula is `1 + L*7 + 1 + L*4 + 16`. Check the constant at the top of `tools/train_nested.c`.

---

## Architecture details

### Matrix operations (`matrix.c`)

Three-tier fallback:
```
USE_OPENBLAS defined  →  cblas_sgemm (fastest, multi-threaded)
SIMD_AVX2FMA defined  →  hand-rolled AVX2+FMA tiled kernel
otherwise             →  scalar (i,k,j) loop with OpenMP
```

Activation functions use a degree-6 Cephes polynomial for `expf`:
```
expf(x) = exp2(x * log2e)
        = 2^n * polynomial(x - n)     n = floor(x * log2e)
```
Accuracy: max relative error < 2×10⁻⁷ (adequate for float32 softmax).

### Attention (`attention.c`)

Standard scaled dot-product attention with Grouped Query Attention (GQA):
```
Q = x · W_Q    [seq × (n_heads * head_dim)]
K = x · W_K    [seq × (n_kv_heads * head_dim)]
V = x · W_V    [seq × (n_kv_heads * head_dim)]

For each query head h:
  kv_h = h / gqa_ratio
  scores = Q[h] · K[kv_h]ᵀ / sqrt(head_dim)
  scores = softmax(causal_mask(scores))
  out[h] = scores · V[kv_h]
```

RoPE is applied to Q and K before score computation. Sliding window attention optionally limits each token to attending only `window_size` past positions.

### Tokenizer (`tokenizer.c`)

- Encoding: O(1) hash map for codepoint → token id lookup, priority-queue BPE merge loop O(n·k·log n)
- Decoding: O(1) via direct `id_to_text[]` array lookup
- Byte fallback: every raw byte has a token id so encoding never produces `<unk>` for ASCII text

---

## Roadmap

- [ ] AVX-512 VNNI mat_mul path (16 floats/op on supported CPUs)
- [ ] Flash Attention - fused kernel to eliminate O(seq²) memory bandwidth
- [ ] Multi-GPU training via MPI or NCCL
- [ ] GGUF/GGML weight export for llama.cpp compatibility
- [ ] Speculative decoding for 2-3× inference speedup
- [ ] Proper MoE load-balancing loss
- [ ] Full FFN backward through SwiGLU (currently partial)
