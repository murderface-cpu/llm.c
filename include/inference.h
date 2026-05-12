/**
 * inference.h
 *
 * Autoregressive inference (text generation) with KV cache.
 *
 * During training, every forward pass processes a full sequence of length T
 * and recomputes Q, K, V for every layer from scratch.
 *
 * During inference (token-by-token generation) that would be wasteful:
 * at step t we only need to compute Q, K, V for the ONE new token, then
 * attend it against all previously computed K and V vectors.
 *
 * The KV cache stores these past K and V tensors so we never recompute them.
 *
 *   Training:   process all T tokens at once   — O(T²) attention per layer
 *   Inference:  process 1 token per step       — O(T)  attention per step
 *               past K,V reused from cache     — total O(T²) across all steps
 *
 * Cache memory:
 *   Per layer: 2 × max_seq_len × (n_kv_heads × head_dim) × 4 bytes
 *   Default (8 layers, 2 KV heads, head_dim=64, seq=2048):
 *     2 × 2048 × 128 × 4 × 8 = 16 MB  — negligible.
 *
 * Sampling strategies provided:
 *   - Greedy  (argmax)  — deterministic, highest-probability token
 *   - Top-k   sampling  — sample from the k highest-prob tokens
 *   - Top-p   (nucleus) — sample from the smallest set summing to prob p
 *   - Temperature       — scale logits before softmax (< 1.0 = sharper)
 */

#ifndef LLM_INFERENCE_H
#define LLM_INFERENCE_H

#include "transformer.h"
#include "tokenizer.h"
#include <stdint.h>
#include "config.h"

/* ---------------------------------------------------------------------------
 * LayerKVCache — K and V cache for one transformer layer
 *
 * Stores keys and values for all past token positions.
 * Shape per tensor: [max_seq_len × (n_kv_heads * head_dim)]
 * ------------------------------------------------------------------------- */
typedef struct {
    float *K;   /* [max_seq_len × kv_dim]  — key   vectors for past tokens  */
    float *V;   /* [max_seq_len × kv_dim]  — value vectors for past tokens  */
} LayerKVCache;

/* ---------------------------------------------------------------------------
 * InferenceState — everything needed for one generation run
 * ------------------------------------------------------------------------- */
typedef struct {
    /* Per-layer KV caches */
    LayerKVCache *kv_cache;  /* [n_layers]                                   */
    int           n_layers;

    /* Current sequence length (tokens generated so far, including prompt)   */
    int seq_pos;

    /* Dimensions (copied from ModelConfig for convenience) */
    int n_kv_heads;
    int head_dim;
    int kv_dim;        /* n_kv_heads * head_dim                              */
    int max_seq_len;

    /* Scratch buffers for the single-token forward pass */
    float *x;          /* residual stream  [d_model]                         */
    float *x_norm;     /* post-RMSNorm     [d_model]                         */
    float *q;          /* query vector     [n_heads * head_dim]              */
    float *k;          /* key vector       [kv_dim]  — written to cache      */
    float *v;          /* value vector     [kv_dim]  — written to cache      */
    float *attn_buf;   /* attention scores [max_seq_len]  per head           */
    float *head_out;   /* weighted sum     [n_heads * head_dim]              */
    float *logits;     /* final logits     [vocab_size]                      */
} InferenceState;

/* ---------------------------------------------------------------------------
 * SamplerConfig — controls how the next token is chosen from logits
 * ------------------------------------------------------------------------- */
typedef struct {
    float temperature;   /* > 0.0; 1.0 = no change, < 1.0 = sharper dist   */
    float top_p;         /* nucleus probability mass, 0.0 = disabled        */
    int   top_k;         /* keep only top-k tokens, 0 = disabled            */
    unsigned int seed;   /* RNG seed; 0 = use time-based seed               */
} SamplerConfig;

static inline SamplerConfig sampler_default(void) {
    return (SamplerConfig){
        .temperature = 0.8f,
        .top_p       = 0.9f,
        .top_k       = 40,
        .seed        = 0,
    };
}

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */

/**
 * inference_init — allocate KV caches and scratch buffers for one model.
 * Must be called once before any generate() call.
 * Returns NULL on allocation failure.
 */
InferenceState *inference_init(const Model *m);

/**
 * inference_free — release all memory owned by the InferenceState.
 */
void inference_free(InferenceState *state);

/**
 * inference_reset — clear KV cache and reset seq_pos to 0.
 * Call between independent generation runs (new conversations / prompts).
 */
void inference_reset(InferenceState *state);

/* ---------------------------------------------------------------------------
 * Single-token forward pass (the KV-cache-aware inference step)
 * ------------------------------------------------------------------------- */

/**
 * inference_step — run one autoregressive step.
 *
 * Given the token id of the most recently generated token (`token`),
 * computes the full transformer forward pass for that single position,
 * updating the KV cache for all layers.
 *
 * After the call, state->logits contains the unnormalised log-probabilities
 * over the vocabulary for the NEXT token.
 *
 * @m:      the model (weights are read-only during inference)
 * @state:  the running inference state (KV cache, seq_pos, scratch buffers)
 * @token:  the input token id for this step
 */
void inference_step(const Model *m, InferenceState *state, int token);

/* ---------------------------------------------------------------------------
 * Sampling
 * ------------------------------------------------------------------------- */

/**
 * sample_token — choose the next token id from logits.
 *
 * Applies temperature scaling, then top-k and/or top-p filtering,
 * then samples from the resulting distribution.
 *
 * @logits:     [vocab_size] raw logits (not softmax-ed; modified in-place)
 * @vocab_size: number of tokens
 * @cfg:        sampling parameters
 * @rng_state:  pointer to uint64_t RNG state (updated each call)
 *
 * Returns the sampled token id.
 */
int sample_token(float *logits, int vocab_size,
                 const SamplerConfig *cfg, uint64_t *rng_state);

/**
 * sample_argmax — greedy decoding: return the highest-logit token.
 * No randomness; always deterministic.
 */
int sample_argmax(const float *logits, int vocab_size);

/* ---------------------------------------------------------------------------
 * High-level generation loop
 * ------------------------------------------------------------------------- */

/**
 * GenerateConfig — parameters for a full generation run.
 */
typedef struct {
    int   max_new_tokens;  /* maximum tokens to generate                    */
    int   min_new_tokens;  /* don't stop before this many tokens            */
    int  *stop_tokens;     /* array of token ids that end generation        */
    int   n_stop_tokens;   /* length of stop_tokens array                   */
    SamplerConfig sampler; /* sampling parameters                           */

    /* Optional streaming callback: called after each generated token.
     * Return 0 to continue, non-zero to stop generation early.
     * `user_data` is passed through unchanged. */
    int  (*on_token)(int token_id, const char *token_text, void *user_data);
    void  *user_data;
} GenerateConfig;

static inline GenerateConfig generate_config_default(void) {
    return (GenerateConfig){
        .max_new_tokens = 256,
        .min_new_tokens = 1,
        .stop_tokens    = NULL,
        .n_stop_tokens  = 0,
        .sampler        = { .temperature=0.8f, .top_p=0.9f,
                            .top_k=40, .seed=0 },
        .on_token       = NULL,
        .user_data      = NULL,
    };
}

/**
 * generate — run the full autoregressive generation loop.
 *
 * Encodes `prompt`, runs prefill (forward pass over all prompt tokens
 * to fill the KV cache), then generates up to max_new_tokens new tokens.
 *
 * @m:      model (read-only)
 * @tok:    tokenizer for encode/decode
 * @state:  inference state (KV cache, scratch); reset internally if needed
 * @prompt: null-terminated UTF-8 input prompt
 * @cfg:    generation configuration
 *
 * Returns a heap-allocated null-terminated string with the generated text
 * (not including the prompt).  Caller must free().
 * Returns NULL on failure.
 */
char *generate(const Model *m, const Tokenizer *tok,
               InferenceState *state,
               const char *prompt,
               const GenerateConfig *cfg);

#endif /* LLM_INFERENCE_H */
