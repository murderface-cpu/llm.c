/**
 * attention.h
 *
 * Attention mechanisms:
 *   - Scaled Dot-Product Attention (the base primitive)
 *   - Multi-Head Attention with Grouped Query Attention (GQA) support
 *   - Sliding Window Attention (O(n·w) instead of O(n²))
 *   - Rotary Position Embeddings (RoPE)
 *
 * Grouped Query Attention (GQA) explanation:
 *   Standard MHA: every head has its own Q, K, V projections.
 *   GQA:          n_heads query heads share n_kv_heads K/V heads.
 *                 e.g. 8 Q heads with 2 KV heads → each KV head serves 4 Q heads.
 *   Benefit:      Reduces KV cache memory and K/V projection compute by 4×.
 *                 Used in LLaMA 3, Mistral, Gemma.
 *
 * Sliding Window Attention explanation:
 *   Each token only attends to the `window_size` most recent tokens.
 *   Reduces attention compute from O(n²) to O(n·w).
 *   Long-range dependencies are still learned through layer stacking.
 */

#ifndef LLM_ATTENTION_H
#define LLM_ATTENTION_H

#include "matrix.h"
#include "config.h"

/* ---------------------------------------------------------------------------
 * AttentionWeights
 *
 * Learnable projection matrices for one attention layer.
 * W_Q: [d_model × (n_heads    * head_dim)]
 * W_K: [d_model × (n_kv_heads * head_dim)]
 * W_V: [d_model × (n_kv_heads * head_dim)]
 * W_O: [(n_heads * head_dim) × d_model]
 *
 * No bias terms — following LLaMA design (cleaner, rarely hurts quality).
 * ------------------------------------------------------------------------- */
typedef struct {
    Matrix W_Q;   /* Query projection                                        */
    Matrix W_K;   /* Key projection    (smaller if n_kv_heads < n_heads)    */
    Matrix W_V;   /* Value projection  (smaller if n_kv_heads < n_heads)    */
    Matrix W_O;   /* Output projection — projects concatenated heads back    */
} AttentionWeights;


/* ---------------------------------------------------------------------------
 * AttentionCache
 *
 * KV cache for inference: stores pre-computed K and V tensors for each
 * past token so we don't recompute them on every generation step.
 *
 * Shape: [max_seq_len × (n_kv_heads * head_dim)]
 * Only used during inference (not training — we keep full activations then).
 * ------------------------------------------------------------------------- */
typedef struct {
    Matrix K_cache;     /* Cached keys   for all past positions              */
    Matrix V_cache;     /* Cached values for all past positions              */
    int    head;        /* Which KV-head group this cache belongs to         */
    int    seq_pos;     /* Number of tokens currently in the cache           */
} AttentionCache;


/* ---------------------------------------------------------------------------
 * RoPEBuffer
 *
 * Pre-computed sine/cosine tables for Rotary Position Embeddings.
 * Computed once at model init, reused across all layers and steps.
 *
 * cos_table[pos][i] = cos(pos / theta^(2i/head_dim))
 * sin_table[pos][i] = sin(pos / theta^(2i/head_dim))
 * Both: [max_seq_len × (head_dim/2)]
 * ------------------------------------------------------------------------- */
typedef struct {
    float *cos_table;  /* [max_seq_len × head_dim/2]                         */
    float *sin_table;  /* [max_seq_len × head_dim/2]                         */
    int    max_seq_len;
    int    head_dim;
} RoPEBuffer;


/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */

/**
 * attn_weights_alloc — allocate and He-initialise all four projection matrices.
 */
AttentionWeights attn_weights_alloc(const ModelConfig *cfg);

/** attn_weights_free — release all projection matrices. */
void attn_weights_free(AttentionWeights *w);

/**
 * rope_buffer_init — pre-compute the cos/sin tables.
 * Call once after loading / initialising the model config.
 */
RoPEBuffer rope_buffer_init(int max_seq_len, int head_dim, float theta);

/** rope_buffer_free */
void rope_buffer_free(RoPEBuffer *rb);

/**
 * attn_cache_alloc — allocate a KV cache for one layer.
 * `kv_dim` = n_kv_heads * head_dim
 */
AttentionCache attn_cache_alloc(int max_seq_len, int kv_dim);

/** attn_cache_free */
void attn_cache_free(AttentionCache *c);

/** attn_cache_reset — zero the cache and reset seq_pos to 0. */
void attn_cache_reset(AttentionCache *c);


/* ---------------------------------------------------------------------------
 * RoPE application
 * ------------------------------------------------------------------------- */

/**
 * rope_apply — apply rotary position embeddings to Q or K in-place.
 *
 * For each head vector h of dimension head_dim, split into pairs (h[2i], h[2i+1])
 * and rotate by the angle for position `pos`:
 *   h[2i]   = h[2i]   * cos(pos,i) - h[2i+1] * sin(pos,i)
 *   h[2i+1] = h[2i+1] * cos(pos,i) + h[2i]   * sin(pos,i)
 *
 * `qk`:     [seq_len × (n_heads * head_dim)]  (Q or K projection output)
 * `rb`:     the pre-computed tables
 * `offset`: token position offset (0 for training; seq_pos for inference)
 */
void rope_apply(Matrix *qk, const RoPEBuffer *rb,
                int n_heads, int head_dim, int offset);


/* ---------------------------------------------------------------------------
 * Core attention forward pass
 * ------------------------------------------------------------------------- */

/**
 * attention_forward — full multi-head attention forward pass.
 *
 * Steps:
 *   1. Project input x to Q, K, V via W_Q, W_K, W_V
 *   2. Apply RoPE to Q and K
 *   3. For each query head, compute scaled dot-product attention over
 *      the corresponding KV group (GQA)
 *   4. Apply sliding window mask if cfg->window_size > 0
 *   5. Softmax the attention scores
 *   6. Weighted sum of V
 *   7. Concatenate heads and project through W_O
 *
 * @x:      input  [seq_len × d_model]
 * @w:      layer weights
 * @rb:     RoPE tables
 * @cfg:    model config (for n_heads, n_kv_heads, head_dim, window_size)
 * @out:    output [seq_len × d_model]  — must be pre-allocated
 *
 * Scratch buffers (must be pre-allocated by caller to avoid malloc in the hot path):
 * @Q:      [seq_len × (n_heads    * head_dim)]
 * @K:      [seq_len × (n_kv_heads * head_dim)]
 * @V:      [seq_len × (n_kv_heads * head_dim)]
 * @scores: [n_heads × seq_len × seq_len]  — attention score matrix
 * @attn:   [seq_len × (n_heads * head_dim)]  — post-softmax weighted values
 */
void attention_forward(const Matrix *x,
                       const AttentionWeights *w,
                       const RoPEBuffer *rb,
                       const ModelConfig *cfg,
                       Matrix *Q, Matrix *K, Matrix *V,
                       Matrix *scores, Matrix *attn,
                       Matrix *out);

/**
 * attention_backward — backprop through the attention layer.
 *
 * Accumulates gradients into w->W_Q.grad, w->W_K.grad, w->W_V.grad, w->W_O.grad
 * and into d_x (gradient w.r.t. the input x).
 *
 * Requires the saved forward activations Q, K, V, scores, attn.
 */
void attention_backward(const Matrix *x,
                        const AttentionWeights *w,
                        const ModelConfig *cfg,
                        const Matrix *Q, const Matrix *K, const Matrix *V,
                        const Matrix *scores, const Matrix *attn,
                        const Matrix *d_out,
                        Matrix *d_x);


/* ---------------------------------------------------------------------------
 * Causal mask helper
 * ------------------------------------------------------------------------- */

/**
 * apply_causal_mask — set scores[i][j] = -inf where j > i (future tokens).
 * Also applies the sliding window: set -inf where i - j >= window_size.
 * `scores`: [seq_len × seq_len] attention score matrix for one head.
 * `window_size`: 0 means no window limit (full causal attention).
 */
void apply_causal_mask(Matrix *scores, int window_size);

#endif /* LLM_ATTENTION_H */