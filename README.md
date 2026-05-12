A lean, efficient Large Language Model built from scratch in pure C.

## Architecture Highlights

- Modern Transformer: RMSNorm + SwiGLU + RoPE + Grouped Query Attention
- Sliding window attention - O(n·w) not O(n²)
- Mixture of Experts (MoE) FFN layer support
- SIMD-accelerated matrix operations
- Knowledge distillation training support

## Project Structure

```
llm.c/
├── include/           # Header files (public API)
│   ├── matrix.h       # Matrix type and math operations
│   ├── attention.h    # Attention mechanisms
│   ├── transformer.h  # Transformer block and model
│   ├── tokenizer.h    # BPE tokenizer
│   └── config.h       # Hyperparameter config structs
├── src/               # Implementations
│   ├── matrix.c
│   ├── attention.c
│   ├── transformer.c
│   ├── tokenizer.c
│   └── train.c
├── tests/             # Unit tests per module
├── tools/             # Utilities (data prep, conversion)
└── docs/              # Notes and architecture diagrams
```

## Build

```bash
make            # optimized build
make debug      # with sanitizers
make test       # run unit tests
```
## Training

```
# Step 1 - build a small vocabulary (512 merges is plenty)

./build/build_vocab data/corpus.txt 512 data/tokenizer.vocab

# Step 2 - tokenise

./build/prepare_data data/corpus.txt data/tokenizer.vocab

# Step 3 - train (auto-configures to tiny model, ~2000 steps)

./build/train

# Step 4 - generate

./build/generate checkpoints/ckpt_00500.bin data/tokenizer.vocab --prompt "The transformer" --temp 0.7 
```

## Phase Roadmap

- [x] Phase 0: Project scaffold
- [ ] Phase 1: Matrix ops + forward pass
- [ ] Phase 2: Backprop + training loop
- [ ] Phase 3: Tokenizer + data pipeline
- [ ] Phase 4: SIMD optimizations
- [ ] Phase 5: MoE + distillation