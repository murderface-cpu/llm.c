/**
 * transformer.h
 *
 * Full Transformer model: block definition, model struct, forward/backward,
 * parameter init, save/load, and the Mixture-of-Experts FFN.
 *
 * Block layout (per layer):
 *   x → RMSNorm → Attention → residual add
 *     → RMSNorm → FFN (MoE or dense) → residual add
 *
 * This is the pre-norm variant (norm before sublayer, not after).
 * Pre-norm training is more stable — used in LLaMA, GPT-NeoX, Mistral.
 */

#ifndef LLM_TRANSFORMER_H
#define LLM_TRANSFORMER_H

#include "matrix.h"
#include "attention.h"
#include "config.h"

/* ---------------------------------------------------------------------------
 * FFNWeights  —  dense feed-forward network (single expert)
 *
 * SwiGLU variant has THREE weight matrices instead of two:
 *   gate:  [d_model × ffn_hidden]   — the gating branch
 *   up:    [d_model × ffn_hidden]   — the value branch
 *   down:  [ffn_hidden × d_model]   — projection back to residual stream
 *
 * Forward:  h = SwiGLU(x·gate, x·up) = silu(x·gate) ⊙ (x·up)
 *           out = h · down
 *
 * No bias (LLaMA style).
 * ------------------------------------------------------------------------- */
typedef struct {
    Matrix W_gate;   /* [d_model   × ffn_hidden] gating projection           */
    Matrix W_up;     /* [d_model   × ffn_hidden] value projection            */
    Matrix W_down;   /* [ffn_hidden × d_model]   output projection           */
} FFNWeights;


/* ---------------------------------------------------------------------------
 * MoEWeights  —  Mixture of Experts FFN
 *
 * Contains n_experts independent FFN experts plus a router.
 * Router: [d_model × n_experts] — produces logits over experts.
 * Top-k experts are selected per token; others are zeroed out.
 *
 * During training: soft top-k (differentiable via straight-through estimator
 * or auxiliary load-balancing loss).
 * During inference: hard top-k (fast, greedy).
 * ------------------------------------------------------------------------- */
typedef struct {
    FFNWeights *experts;    /* Array of n_experts independent FFN weights     */
    Matrix      W_router;   /* [d_model × n_experts]  token → expert logits  */
    int         n_experts;
    int         top_k;      /* How many experts activate per token            */
} MoEWeights;


/* ---------------------------------------------------------------------------
 * BlockWeights  —  all learnable parameters for one Transformer layer
 * ------------------------------------------------------------------------- */
typedef struct {
    /* Pre-attention norm -------------------------------------------------- */
    float *rms_attn;   /* RMSNorm scale vector, length d_model               */

    /* Attention ----------------------------------------------------------- */
    AttentionWeights attn;

    /* Pre-FFN norm -------------------------------------------------------- */
    float *rms_ffn;    /* RMSNorm scale vector, length d_model               */

    /* Feed-forward (dense or MoE) ---------------------------------------- */
    int use_moe;       /* 0 = dense FFN, 1 = MoE                             */
    union {
        FFNWeights ffn;
        MoEWeights moe;
    };
} BlockWeights;


/* ---------------------------------------------------------------------------
 * ModelWeights  —  all parameters in the full model
 * ------------------------------------------------------------------------- */
typedef struct {
    /* Token embedding table ---------------------------------------------- */
    Matrix embed;           /* [vocab_size × d_model]                        */

    /* Transformer blocks -------------------------------------------------- */
    BlockWeights *blocks;   /* Array of n_layers blocks                      */
    int           n_layers;

    /* Final norm + LM head ----------------------------------------------- */
    float  *rms_final;      /* RMSNorm scale, length d_model                 */
    Matrix  lm_head;        /* [d_model × vocab_size]  logit projection      */
                            /* Tied to embed.data in weight-tying mode       */
} ModelWeights;


/* ---------------------------------------------------------------------------
 * ForwardCache
 *
 * Stores all intermediate activations needed for backprop.
 * One ForwardCache exists per layer. The caller pre-allocates these once
 * at model init, then reuses them across training steps.
 * ------------------------------------------------------------------------- */
typedef struct {
    Matrix x_norm_attn;   /* Input after RMSNorm (pre-attention)             */
    Matrix x_norm_ffn;    /* Input after RMSNorm (pre-FFN)                   */
    Matrix attn_out;      /* Output of attention sublayer (pre-residual)     */
    Matrix ffn_out;       /* Output of FFN sublayer (pre-residual)           */
    /* Attention scratch buffers (reused each step to avoid malloc) -------- */
    Matrix Q, K, V;
    Matrix attn_scores;   /* [n_heads × seq_len × seq_len]  — actually       */
                          /* stored flat: [n_heads*seq_len × seq_len]        */
    Matrix attn_weighted; /* [seq_len × (n_heads * head_dim)]                */
    /* FFN scratch — properly sized to ffn_hidden -------------------------*/
    Matrix gate_buf;      /* [seq_len × ffn_hidden]                          */
    Matrix up_buf;        /* [seq_len × ffn_hidden]                          */
    /* MoE router cache ---------------------------------------------------- */
    float *router_probs;  /* [seq_len × n_experts]  — softmax of router logi */
    int   *expert_ids;    /* [seq_len × top_k]      — selected expert ids    */
} ForwardCache;


/* ---------------------------------------------------------------------------
 * Model  —  top-level struct you interact with
 * ------------------------------------------------------------------------- */
typedef struct {
    ModelConfig   cfg;
    ModelWeights  weights;
    ForwardCache *caches;   /* One ForwardCache per layer                    */
    RoPEBuffer    rope;     /* Shared cos/sin tables                         */
    void         *train_scratch; /* Pre-allocated backward buffers (TrainScratch) */
} Model;


/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */

/**
 * model_init — allocate and randomly initialise a new model.
 * Uses scaled normal initialisation per layer depth.
 */
Model *model_init(const ModelConfig *cfg);

/** model_free — release all memory. */
void model_free(Model *m);

/**
 * model_save — write weights to a binary checkpoint file.
 * Format: [ModelConfig][weight tensors in declaration order]
 */
int model_save(const Model *m, const char *path);

/**
 * model_load — read a checkpoint produced by model_save.
 * Returns NULL on failure.
 */
Model *model_load(const char *path);

/** model_param_count — return total number of scalar parameters. */
long model_param_count(const Model *m);


/* ---------------------------------------------------------------------------
 * Forward pass
 * ------------------------------------------------------------------------- */

/**
 * model_forward — run the full forward pass and return logits.
 *
 * @tokens:   integer array of length seq_len
 * @seq_len:  number of tokens
 * @logits:   output [seq_len × vocab_size] — must be pre-allocated
 *
 * Intermediate activations saved into m->caches for use by backward pass.
 */
void model_forward(Model *m, const int *tokens, int seq_len, Matrix *logits);

/**
 * block_forward — forward pass for a single Transformer block.
 * Called internally by model_forward; exposed here for testing.
 *
 * @x:     input/output residual stream [seq_len × d_model]
 *         MODIFIED IN PLACE (residual connections applied here).
 * @bw:    this block's weights
 * @cache: this block's activation cache
 * @rope:  shared RoPE tables
 * @cfg:   model config
 * @layer: layer index (for debugging / logging)
 */
void block_forward(Matrix *x,
                   const BlockWeights *bw,
                   ForwardCache *cache,
                   const RoPEBuffer *rope,
                   const ModelConfig *cfg,
                   int layer);


/* ---------------------------------------------------------------------------
 * Backward pass
 * ------------------------------------------------------------------------- */

/**
 * model_backward — backprop from cross-entropy loss through the full model.
 *
 * Computes gradients for all parameters (stored in weight .grad fields)
 * given the token targets.
 *
 * @tokens:  input token ids  [seq_len]
 * @targets: target token ids [seq_len]  (shifted by 1 for next-token prediction)
 * @seq_len: sequence length
 *
 * Returns the mean cross-entropy loss (scalar).
 */
float model_backward(Model *m,
                     const int *tokens,
                     const int *targets,
                     int seq_len);

/**
 * model_zero_grads — zero all .grad buffers before a new backward pass.
 * Call this at the start of each training step.
 */
void model_zero_grads(Model *m);


/* ---------------------------------------------------------------------------
 * FFN helpers (exposed for unit tests)
 * ------------------------------------------------------------------------- */

/**
 * ffn_backward — backprop through the dense SwiGLU FFN.
 * @gate_pre: saved pre-activation of gate branch (x · W_gate)
 * @up_pre:   saved pre-activation of up branch   (x · W_up)
 * @h:        saved SwiGLU output (silu(gate_pre) ⊙ up_pre)
 * @d_out:    incoming gradient [seq × d_model]
 * @d_x:      output gradient  [seq × d_model]  (zeroed then filled)
 */
void ffn_backward(const Matrix *x,
                  const FFNWeights *w,
                  const Matrix *gate_pre,
                  const Matrix *up_pre,
                  const Matrix *h,
                  const Matrix *d_out,
                  Matrix *d_x);

/** ffn_weights_alloc */
FFNWeights ffn_weights_alloc(int d_model, int ffn_hidden, int with_grad);

/** ffn_weights_free */
void ffn_weights_free(FFNWeights *w);

/**
 * ffn_forward — dense SwiGLU FFN forward.
 * @x:   [seq_len × d_model]
 * @out: [seq_len × d_model]  pre-allocated
 */
void ffn_forward(const Matrix *x, const FFNWeights *w,
                 const ModelConfig *cfg, Matrix *gate_buf,
                 Matrix *up_buf, Matrix *out);

/**
 * moe_forward — MoE FFN forward.
 * Routes each token to top_k experts and combines their outputs.
 */
void moe_forward(const Matrix *x, const MoEWeights *w,
                 const ModelConfig *cfg, ForwardCache *cache,
                 Matrix *out);

#endif /* LLM_TRANSFORMER_H */