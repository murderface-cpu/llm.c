/**
 * config.h
 *
 * Central configuration structs for the model architecture and training.
 * All hyperparameters live here — change them in one place only.
 *
 * Typical small model (fits on a single GPU / strong CPU):
 *   d_model=512, n_layers=8, n_heads=8, n_kv_heads=2, ffn_hidden=1536
 *   -> ~50M parameters
 */

#ifndef LLM_CONFIG_H
#define LLM_CONFIG_H

#include <stddef.h>
#include <stdio.h>

/* ---------------------------------------------------------------------------
 * ModelConfig
 * Describes the static shape of the Transformer architecture.
 * Set once at startup; never mutated during a run.
 * ------------------------------------------------------------------------- */
typedef struct {
    /* Vocabulary & context ------------------------------------------------ */
    int vocab_size;     /* Number of tokens in the vocabulary               */
    int max_seq_len;    /* Maximum sequence length (context window)          */

    /* Transformer dimensions --------------------------------------------- */
    int d_model;        /* Embedding / residual stream dimension (e.g. 512)  */
    int n_layers;       /* Number of stacked Transformer blocks              */

    /* Attention ---------------------------------------------------------- */
    int n_heads;        /* Number of query heads                             */
    int n_kv_heads;     /* Number of key/value heads (< n_heads => GQA)      */
    int head_dim;       /* d_model / n_heads  (computed, not set by user)    */
    int window_size;    /* Sliding-window attention span; 0 = full attention  */

    /* Feed-forward ------------------------------------------------------- */
    int ffn_hidden;     /* Inner dimension of the FFN / MoE expert           */
    int n_experts;      /* Number of MoE experts; 1 = standard dense FFN     */
    int top_k_experts;  /* How many experts activate per token (usually 1-2) */

    /* Norms & activations ------------------------------------------------ */
    float rms_norm_eps; /* Epsilon for RMSNorm numerical stability           */

    /* Positional encoding ------------------------------------------------ */
    float rope_theta;   /* RoPE base frequency (default 10000.0)             */
} ModelConfig;


/* ---------------------------------------------------------------------------
 * TrainConfig
 * Describes the training loop hyper-parameters.
 * ------------------------------------------------------------------------- */
typedef struct {
    /* Data --------------------------------------------------------------- */
    const char *train_data_path;  /* Path to tokenised binary training data  */
    const char *val_data_path;    /* Path to tokenised binary validation data */

    /* Optimiser (AdamW) -------------------------------------------------- */
    float learning_rate;    /* Peak learning rate (e.g. 3e-4)               */
    float lr_min;           /* Minimum LR after cosine decay                */
    float beta1;            /* Adam β₁ (default 0.9)                        */
    float beta2;            /* Adam β₂ (default 0.95)                       */
    float epsilon;          /* Adam ε  (default 1e-8)                       */
    float weight_decay;     /* AdamW weight decay (default 0.1)             */
    float grad_clip;        /* Gradient clipping norm (default 1.0)         */

    /* Schedule ----------------------------------------------------------- */
    int warmup_steps;       /* Linear LR warmup steps                       */
    int max_steps;          /* Total training steps                         */

    /* Batching ----------------------------------------------------------- */
    int batch_size;         /* Sequences per batch                          */
    int seq_len;            /* Sequence length per sample                   */

    /* Regularisation ----------------------------------------------------- */
    float dropout;          /* Dropout probability (0.0 = disabled)         */
    float label_smoothing;  /* Label smoothing epsilon (0.0 = disabled)     */

    /* Distillation ------------------------------------------------------- */
    int use_distillation;   /* 1 = distill from teacher logits file         */
    float distill_alpha;    /* Weight of distillation loss vs CE loss       */
    const char *teacher_logits_path; /* Pre-computed teacher logit file     */

    /* Checkpointing ------------------------------------------------------ */
    const char *checkpoint_dir; /* Where to save checkpoints               */
    int save_every;             /* Save checkpoint every N steps            */
    int eval_every;             /* Evaluate on val set every N steps        */
} TrainConfig;


/* ---------------------------------------------------------------------------
 * Convenience: fill a ModelConfig with sensible 50M-parameter defaults.
 * Call this then override only what you need.
 * ------------------------------------------------------------------------- */
/**
 * model_config_compute — derive fields that are determined by other fields.
 * Call this after setting d_model / n_heads / n_kv_heads manually.
 * Also validates that the configuration is self-consistent.
 */
static inline ModelConfig model_config_compute(ModelConfig cfg) {
    /* head_dim must always equal d_model / n_heads */
    if (cfg.n_heads > 0)
        cfg.head_dim = cfg.d_model / cfg.n_heads;
    /* Validate GQA ratio is an integer */
    if (cfg.n_heads % cfg.n_kv_heads != 0) {
        /* Round n_kv_heads down to the nearest divisor — safe fallback */
        while (cfg.n_heads % cfg.n_kv_heads != 0)
            cfg.n_kv_heads--;
    }
    return cfg;
}

static inline ModelConfig model_config_default(void) {
    ModelConfig cfg = {
        .vocab_size    = 4096,    /* matches build_vocab default output size  */
        .max_seq_len   = 512,     /* sensible default; increase for longer ctx */
        .d_model       = 256,     /* smaller default — trains on CPU          */
        .n_layers      = 4,
        .n_heads       = 4,
        .n_kv_heads    = 2,       /* GQA: 2 query heads share each KV head   */
        .head_dim      = 0,       /* computed below via model_config_compute  */
        .window_size   = 128,     /* sliding window attention span            */
        .ffn_hidden    = 768,     /* ~3× d_model                             */
        .n_experts     = 1,       /* start dense; flip to 8 for MoE later    */
        .top_k_experts = 1,
        .rms_norm_eps  = 1e-5f,
        .rope_theta    = 10000.0f,
    };
    return model_config_compute(cfg);
}

static inline TrainConfig train_config_default(void) {
    TrainConfig cfg = {
        .train_data_path    = "data/train.bin",
        .val_data_path      = "data/val.bin",
        .learning_rate      = 3e-4f,
        .lr_min             = 1e-5f,
        .beta1              = 0.9f,
        .beta2              = 0.95f,
        .epsilon            = 1e-8f,
        .weight_decay       = 0.1f,
        .grad_clip          = 1.0f,
        .warmup_steps       = 2000,
        .max_steps          = 100000,
        .batch_size         = 8,
        .seq_len            = 512,
        .dropout            = 0.0f,
        .label_smoothing    = 0.1f,
        .use_distillation   = 0,
        .distill_alpha      = 0.5f,
        .teacher_logits_path= NULL,
        .checkpoint_dir     = "checkpoints/",
        .save_every         = 1000,
        .eval_every         = 500,
    };
    return cfg;
}


/* ---------------------------------------------------------------------------
 * Config presets — call one of these instead of model_config_default()
 *
 *   config_tiny()    — fits a corpus of < 50K tokens. Trains in minutes
 *                      on a CPU. Good for experimenting and debugging.
 *
 *   config_small()   — fits a corpus of 50K–5M tokens. Trains overnight
 *                      on a CPU, in hours on a GPU.
 *
 *   config_default() — the original 50M-param config. Needs > 5M tokens
 *                      and a GPU or many hours on CPU.
 *
 * Rule of thumb for corpus size:
 *   You need at least 10–20 tokens per model parameter to avoid memorisation.
 *   tiny  (~1M params)  → needs ~10M  tokens minimum
 *   small (~7M params)  → needs ~70M  tokens minimum
 *
 *   For very small corpora (< 1M tokens) you are essentially memorising the
 *   text — which is fine for testing but don't expect generalisation.
 * ------------------------------------------------------------------------- */

/** config_tiny — ~1M params, trains in minutes on a single CPU core */
static inline ModelConfig model_config_tiny(void) {
    ModelConfig cfg = {
        .vocab_size    = 512,    /* small vocab — override with real vocab_size */
        .max_seq_len   = 256,
        .d_model       = 128,
        .n_layers      = 4,
        .n_heads       = 4,
        .n_kv_heads    = 2,
        .head_dim      = 32,     /* 128 / 4 */
        .window_size   = 128,
        .ffn_hidden    = 384,    /* 3× d_model */
        .n_experts     = 1,
        .top_k_experts = 1,
        .rms_norm_eps  = 1e-5f,
        .rope_theta    = 10000.0f,
    };
    return cfg;
}

/** config_small — ~7M params, trains well on a corpus of a few MB */
static inline ModelConfig model_config_small(void) {
    ModelConfig cfg = {
        .vocab_size    = 2048,
        .max_seq_len   = 512,
        .d_model       = 256,
        .n_layers      = 6,
        .n_heads       = 8,
        .n_kv_heads    = 2,
        .head_dim      = 32,     /* 256 / 8 */
        .window_size   = 256,
        .ffn_hidden    = 768,
        .n_experts     = 1,
        .top_k_experts = 1,
        .rms_norm_eps  = 1e-5f,
        .rope_theta    = 10000.0f,
    };
    return cfg;
}

/** train_config_tiny — aggressive schedule for fast iteration on tiny data */
static inline TrainConfig train_config_tiny(void) {
    TrainConfig cfg = {
        .train_data_path    = "data/train.bin",
        .val_data_path      = "data/val.bin",
        .learning_rate      = 1e-3f,   /* higher LR OK for tiny models     */
        .lr_min             = 1e-4f,
        .beta1              = 0.9f,
        .beta2              = 0.95f,
        .epsilon            = 1e-8f,
        .weight_decay       = 0.1f,
        .grad_clip          = 1.0f,
        .warmup_steps       = 100,
        .max_steps          = 2000,    /* few passes over small corpus      */
        .batch_size         = 4,
        .seq_len            = 128,
        .dropout            = 0.0f,
        .label_smoothing    = 0.0f,    /* no smoothing for tiny sets        */
        .use_distillation   = 0,
        .distill_alpha      = 0.5f,
        .teacher_logits_path= NULL,
        .checkpoint_dir     = "checkpoints/",
        .save_every         = 500,
        .eval_every         = 200,
    };
    return cfg;
}

/**
 * config_auto — choose model and training config based on corpus token count.
 * Pass n_train_tokens from the meta.txt file written by prepare_data.
 *
 * Sets *mcfg and *tcfg appropriately and prints a recommendation.
 */
static inline void config_auto(long n_train_tokens, int vocab_size,
                                ModelConfig *mcfg, TrainConfig *tcfg) {
    if (n_train_tokens < 50000L) {
        *mcfg = model_config_tiny();
        *tcfg = train_config_tiny();
        fprintf(stderr,
            "[config_auto] tiny corpus (%ld tokens) → tiny model (1M params)\n"
            "              Expect memorisation, not generalisation.\n",
            n_train_tokens);
    } else if (n_train_tokens < 2000000L) {
        *mcfg = model_config_small();
        *tcfg = train_config_default();
        tcfg->max_steps    = (int)(n_train_tokens / (tcfg->batch_size * tcfg->seq_len)) * 10;
        tcfg->warmup_steps = tcfg->max_steps / 20;
        fprintf(stderr,
            "[config_auto] small corpus (%ld tokens) → small model (7M params)\n"
            "              max_steps=%d\n",
            n_train_tokens, tcfg->max_steps);
    } else {
        *mcfg = model_config_default();
        *tcfg = train_config_default();
        tcfg->max_steps    = (int)(n_train_tokens / (tcfg->batch_size * tcfg->seq_len)) * 3;
        tcfg->warmup_steps = tcfg->max_steps / 20;
        fprintf(stderr,
            "[config_auto] large corpus (%ld tokens) → default model (50M params)\n"
            "              max_steps=%d\n",
            n_train_tokens, tcfg->max_steps);
    }
    mcfg->vocab_size = vocab_size;
}

#endif /* LLM_CONFIG_H */
