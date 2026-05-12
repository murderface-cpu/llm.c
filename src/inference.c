/**
 * inference.c
 *
 * KV-cache autoregressive inference engine.
 *
 * Key difference from the training forward pass
 * ─────────────────────────────────────────────
 * Training:    processes a full sequence at once.
 *              attention_forward() receives x[seq × d_model].
 *              Q, K, V are all [seq × dim].
 *
 * Inference:   processes ONE token per step.
 *              x is a single row vector [1 × d_model] = [d_model].
 *              Q  = x · W_Q   → [1 × (n_heads*head_dim)]  (new query)
 *              K  = x · W_K   → [1 × kv_dim]              (new key, saved to cache)
 *              V  = x · W_V   → [1 × kv_dim]              (new val, saved to cache)
 *
 *              Attention for head h at position t:
 *                scores[j] = Q[h] · K_cache[j, kv_h] / sqrt(head_dim)
 *                            for j in [0, t+1)
 *                out[h] = softmax(scores) · V_cache[:t+1, kv_h]
 *
 * This gives O(t) attention per step vs O(t²) without the cache.
 *
 * RoPE for inference
 * ──────────────────
 * We apply RoPE at position `seq_pos` (the absolute token index) rather
 * than at position 0 (which is what training does for each local position).
 * The cos/sin tables in RoPEBuffer are pre-computed for all positions up
 * to max_seq_len, so we just index by seq_pos.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#include "../include/inference.h"
#include "../include/simd.h"

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "[inference] OOM %zu bytes\n", n); abort(); }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "[inference] OOM\n"); abort(); }
    return p;
}

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

InferenceState *inference_init(const Model *m) {
    const ModelConfig *cfg = &m->cfg;
    InferenceState *s = (InferenceState *)xcalloc(1, sizeof(InferenceState));

    s->n_layers    = cfg->n_layers;
    s->n_kv_heads  = cfg->n_kv_heads;
    s->head_dim    = cfg->head_dim;
    s->kv_dim      = cfg->n_kv_heads * cfg->head_dim;
    s->max_seq_len = cfg->max_seq_len;
    s->seq_pos     = 0;

    /* Per-layer KV caches */
    s->kv_cache = (LayerKVCache *)xcalloc(cfg->n_layers, sizeof(LayerKVCache));
    for (int l = 0; l < cfg->n_layers; l++) {
        size_t cache_floats = (size_t)cfg->max_seq_len * s->kv_dim;
        s->kv_cache[l].K = (float *)xcalloc(cache_floats, sizeof(float));
        s->kv_cache[l].V = (float *)xcalloc(cache_floats, sizeof(float));
    }

    /* Single-token scratch buffers */
    int d  = cfg->d_model;
    int nh = cfg->n_heads;
    int hd = cfg->head_dim;
    int V  = cfg->vocab_size;

    s->x        = (float *)xmalloc(d          * sizeof(float));
    s->x_norm   = (float *)xmalloc(d          * sizeof(float));
    s->q        = (float *)xmalloc(nh * hd    * sizeof(float));
    s->k        = (float *)xmalloc(s->kv_dim  * sizeof(float));
    s->v        = (float *)xmalloc(s->kv_dim  * sizeof(float));
    s->attn_buf = (float *)xmalloc(cfg->max_seq_len * sizeof(float));
    s->head_out = (float *)xmalloc(nh * hd    * sizeof(float));
    s->logits   = (float *)xmalloc(V          * sizeof(float));

    return s;
}

void inference_free(InferenceState *s) {
    if (!s) return;
    for (int l = 0; l < s->n_layers; l++) {
        free(s->kv_cache[l].K);
        free(s->kv_cache[l].V);
    }
    free(s->kv_cache);
    free(s->x);
    free(s->x_norm);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->attn_buf);
    free(s->head_out);
    free(s->logits);
    free(s);
}

void inference_reset(InferenceState *s) {
    for (int l = 0; l < s->n_layers; l++) {
        size_t bytes = (size_t)s->max_seq_len * s->kv_dim * sizeof(float);
        memset(s->kv_cache[l].K, 0, bytes);
        memset(s->kv_cache[l].V, 0, bytes);
    }
    s->seq_pos = 0;
}

/* =========================================================================
 * Vector dot product (uses SIMD if available)
 * ====================================================================== */

static inline float vec_dot(const float *a, const float *b, int n) {
#if defined(SIMD_AVX2FMA)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),
                               _mm256_loadu_ps(b+i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),
                               _mm256_loadu_ps(b+i+8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i <= n - 8; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),
                               _mm256_loadu_ps(b+i), acc0);
    float result = hsum256(acc0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* AXPY: y += scale * x,  length n */
static inline void vec_axpy(float *y, const float *x, float scale, int n) {
#if defined(SIMD_AVX2FMA)
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(y+i,
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(x+i),
                            _mm256_loadu_ps(y+i)));
    for (; i < n; i++) y[i] += scale * x[i];
#else
    for (int i = 0; i < n; i++) y[i] += scale * x[i];
#endif
}

/* Matrix-vector product: out[N] = mat[M×N]ᵀ · vec[M]   (out = vecᵀ · mat) */
static void matvec(const float *mat, const float *vec,
                   float *out, int M, int N) {
    memset(out, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float vi = vec[i];
        if (vi == 0.0f) continue;
        vec_axpy(out, mat + i * N, vi, N);
    }
}

/* =========================================================================
 * RoPE for a single position
 *
 * Rotates a [n_heads × head_dim] flat array at absolute position `pos`.
 * ====================================================================== */
static void rope_apply_pos(float *qk, const RoPEBuffer *rb,
                            int n_heads, int head_dim, int pos) {
    int half = head_dim / 2;
    const float *cos_row = rb->cos_table + pos * half;
    const float *sin_row = rb->sin_table + pos * half;

    for (int h = 0; h < n_heads; h++) {
        float *v = qk + h * head_dim;
        for (int i = 0; i < half; i++) {
            float v0 =  v[2*i]   * cos_row[i] - v[2*i+1] * sin_row[i];
            float v1 =  v[2*i]   * sin_row[i] + v[2*i+1] * cos_row[i];
            v[2*i]   = v0;
            v[2*i+1] = v1;
        }
    }
}

/* =========================================================================
 * RMSNorm for a single vector
 * ====================================================================== */
static void rmsnorm_vec(const float *x, const float *w,
                         float eps, float *out, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv_rms * w[i];
}

/* Softmax in-place for a vector of length n */
static void softmax_vec(float *v, int n) {
    float vmax = v[0];
    for (int i = 1; i < n; i++) if (v[i] > vmax) vmax = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - vmax); sum += v[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) v[i] *= inv;
}

/* =========================================================================
 * inference_step — single-token transformer forward pass with KV cache
 * ====================================================================== */

void inference_step(const Model *m, InferenceState *s, int token) {
    const ModelConfig *cfg = &m->cfg;
    int d       = cfg->d_model;
    int nh      = cfg->n_heads;
    int nkv     = cfg->n_kv_heads;
    int hd      = cfg->head_dim;
    int kv_dim  = s->kv_dim;
    int gqa     = nh / nkv;           /* query heads per KV head            */
    int pos     = s->seq_pos;
    float scale = 1.0f / sqrtf((float)hd);

    /* ------------------------------------------------------------------ */
    /* 1. Token embedding lookup: x = embed[token]                         */
    memcpy(s->x, m->weights.embed.data + token * d, d * sizeof(float));

    /* ------------------------------------------------------------------ */
    /* 2. Run through all transformer blocks                                */
    for (int l = 0; l < cfg->n_layers; l++) {
        const BlockWeights *bw   = &m->weights.blocks[l];
        LayerKVCache       *kvc  = &s->kv_cache[l];

        /* 2a. Pre-attention RMSNorm */
        rmsnorm_vec(s->x, bw->rms_attn, cfg->rms_norm_eps, s->x_norm, d);

        /* 2b. Project Q, K, V for this single token
         *     Q = x_norm · W_Q  →  [nh * hd]
         *     K = x_norm · W_K  →  [kv_dim]
         *     V = x_norm · W_V  →  [kv_dim]                               */
        matvec(bw->attn.W_Q.data, s->x_norm, s->q, d, nh * hd);
        matvec(bw->attn.W_K.data, s->x_norm, s->k, d, kv_dim);
        matvec(bw->attn.W_V.data, s->x_norm, s->v, d, kv_dim);

        /* 2c. Apply RoPE at absolute position `pos` */
        rope_apply_pos(s->q, &m->rope, nh,  hd, pos);
        rope_apply_pos(s->k, &m->rope, nkv, hd, pos);

        /* 2d. Write K and V into the cache at position `pos` */
        float *k_slot = kvc->K + pos * kv_dim;
        float *v_slot = kvc->V + pos * kv_dim;
        memcpy(k_slot, s->k, kv_dim * sizeof(float));
        memcpy(v_slot, s->v, kv_dim * sizeof(float));

        /* 2e. Grouped Query Attention over cache positions [0..pos]
         *
         * For each query head qh:
         *   kv_h = qh / gqa  (which KV cache slot to attend)
         *   scores[j] = Q[qh] · K_cache[j, kv_h] * scale   j in [0,pos+1)
         *   softmax(scores)
         *   out[qh] = sum_j scores[j] * V_cache[j, kv_h]    */
        memset(s->head_out, 0, (size_t)nh * hd * sizeof(float));

        for (int qh = 0; qh < nh; qh++) {
            int kv_h = qh / gqa;
            float *q_head  = s->q + qh * hd;
            float *out_head = s->head_out + qh * hd;

            /* Compute attention scores for all past positions */
            for (int j = 0; j <= pos; j++) {
                /* K_cache[j] for kv_h: row j of cache, slice for kv_h */
                const float *k_j = kvc->K + j * kv_dim + kv_h * hd;
                s->attn_buf[j] = vec_dot(q_head, k_j, hd) * scale;
            }

            /* Causal: all positions [0..pos] are valid (no future masking needed) */
            softmax_vec(s->attn_buf, pos + 1);

            /* Weighted sum of V */
            for (int j = 0; j <= pos; j++) {
                const float *v_j = kvc->V + j * kv_dim + kv_h * hd;
                vec_axpy(out_head, v_j, s->attn_buf[j], hd);
            }
        }

        /* 2f. Output projection: attn_out = head_out · W_O
         *     Then residual: x += attn_out                                 */
        float *attn_out = s->x_norm;   /* reuse x_norm as temp scratch      */
        matvec(bw->attn.W_O.data, s->head_out, attn_out, nh * hd, d);
        for (int i = 0; i < d; i++) s->x[i] += attn_out[i];

        /* 2g. Pre-FFN RMSNorm */
        rmsnorm_vec(s->x, bw->rms_ffn, cfg->rms_norm_eps, s->x_norm, d);

        /* 2h. FFN (dense SwiGLU) for a single token
         *     gate = x_norm · W_gate  [ffn_hidden]
         *     up   = x_norm · W_up    [ffn_hidden]
         *     h    = silu(gate) * up
         *     out  = h · W_down       [d_model]                            */
        if (!bw->use_moe) {
            int ffnh = cfg->ffn_hidden;
            float *gate = (float *)malloc(ffnh * sizeof(float));
            float *up   = (float *)malloc(ffnh * sizeof(float));
            float *h    = (float *)malloc(ffnh * sizeof(float));
            float *ffn_out = s->q;   /* reuse q scratch (nh*hd >= d_model usually) */

            matvec(bw->ffn.W_gate.data, s->x_norm, gate, d, ffnh);
            matvec(bw->ffn.W_up.data,   s->x_norm, up,   d, ffnh);

            /* h = silu(gate) * up */
            for (int i = 0; i < ffnh; i++) {
                float g = gate[i];
                h[i] = (g / (1.0f + expf(-g))) * up[i];
            }

            /* ffn_out = h · W_down */
            matvec(bw->ffn.W_down.data, h, ffn_out, ffnh, d);

            /* Residual */
            for (int i = 0; i < d; i++) s->x[i] += ffn_out[i];

            free(gate); free(up); free(h);
        }
        /* Note: MoE inference path is a future TODO — falls through as
         * identity (no FFN update) when use_moe=1 for now.             */
    }

    /* ------------------------------------------------------------------ */
    /* 3. Final RMSNorm                                                     */
    rmsnorm_vec(s->x, m->weights.rms_final, cfg->rms_norm_eps,
                s->x_norm, d);

    /* ------------------------------------------------------------------ */
    /* 4. LM head: logits = x_norm · W_lm_head  [vocab_size]               */
    matvec(m->weights.lm_head.data, s->x_norm, s->logits,
           d, cfg->vocab_size);

    /* Advance position */
    s->seq_pos++;
}

/* =========================================================================
 * Sampling
 * ====================================================================== */

/* xorshift64 — fast, high-quality 64-bit PRNG */
static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Uniform float in [0, 1) */
static inline float rand_float(uint64_t *rng) {
    return (float)(xorshift64(rng) >> 11) / (float)(1ULL << 53);
}

int sample_argmax(const float *logits, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/* Comparison function for qsort (descending by value) */
typedef struct { float val; int idx; } IndexedFloat;
static int cmp_desc(const void *a, const void *b) {
    float da = ((IndexedFloat *)a)->val;
    float db = ((IndexedFloat *)b)->val;
    return (da < db) - (da > db);
}

int sample_token(float *logits, int vocab_size,
                 const SamplerConfig *cfg, uint64_t *rng_state) {

    /* Temperature scaling */
    if (cfg->temperature <= 0.0f || cfg->temperature == 1.0f) {
        /* temperature=0 → greedy */
        if (cfg->temperature <= 0.0f) return sample_argmax(logits, vocab_size);
    } else {
        float inv_temp = 1.0f / cfg->temperature;
        for (int i = 0; i < vocab_size; i++) logits[i] *= inv_temp;
    }

    /* Softmax */
    float vmax = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > vmax) vmax = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) { logits[i] = expf(logits[i]-vmax); sum += logits[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) logits[i] *= inv;

    /* Top-k: zero out all but the k highest-prob tokens */
    int k = cfg->top_k;
    if (k > 0 && k < vocab_size) {
        /* Partial sort: find k-th largest threshold */
        IndexedFloat *sorted = (IndexedFloat *)malloc(
                                   vocab_size * sizeof(IndexedFloat));
        for (int i = 0; i < vocab_size; i++) {
            sorted[i].val = logits[i]; sorted[i].idx = i;
        }
        qsort(sorted, vocab_size, sizeof(IndexedFloat), cmp_desc);

        /* Zero everything below rank k */
        float threshold = sorted[k-1].val;
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < threshold) logits[i] = 0.0f;

        /* Renormalise */
        sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += logits[i];
        if (sum > 0.0f) { inv = 1.0f / sum;
                          for (int i=0; i<vocab_size; i++) logits[i] *= inv; }
        free(sorted);
    }

    /* Top-p (nucleus): keep the smallest set of tokens whose cumulative
     * probability exceeds p, zero the rest */
    float p = cfg->top_p;
    if (p > 0.0f && p < 1.0f) {
        IndexedFloat *sorted = (IndexedFloat *)malloc(
                                   vocab_size * sizeof(IndexedFloat));
        for (int i = 0; i < vocab_size; i++) {
            sorted[i].val = logits[i]; sorted[i].idx = i;
        }
        qsort(sorted, vocab_size, sizeof(IndexedFloat), cmp_desc);

        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += sorted[i].val;
            if (cumsum >= p) { cutoff = i + 1; break; }
        }

        /* Zero tokens beyond cutoff */
        for (int i = cutoff; i < vocab_size; i++) logits[sorted[i].idx] = 0.0f;

        /* Renormalise */
        sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += logits[i];
        if (sum > 0.0f) { inv = 1.0f / sum;
                          for (int i=0; i<vocab_size; i++) logits[i] *= inv; }
        free(sorted);
    }

    /* Sample from the filtered distribution */
    float r = rand_float(rng_state);
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return vocab_size - 1;   /* fallback */
}

/* =========================================================================
 * High-level generate()
 * ====================================================================== */

char *generate(const Model *m, const Tokenizer *tok,
               InferenceState *state,
               const char *prompt,
               const GenerateConfig *gcfg) {

    const ModelConfig *mcfg = &m->cfg;

    /* Initialise RNG */
    uint64_t rng = gcfg->sampler.seed != 0
                 ? (uint64_t)gcfg->sampler.seed
                 : (uint64_t)time(NULL);
    xorshift64(&rng);   /* warm up */

    /* Encode the prompt */
    int max_prompt_len = mcfg->max_seq_len;
    int *prompt_tokens = (int *)malloc((max_prompt_len + 2) * sizeof(int));
    int  n_prompt      = 0;

    if (tok_encode(tok, prompt, /*add_bos=*/1, /*add_eos=*/0,
                   prompt_tokens, &n_prompt) < 0) {
        fprintf(stderr, "[generate] tokenize failed\n");
        free(prompt_tokens); return NULL;
    }

    if (n_prompt >= mcfg->max_seq_len) {
        fprintf(stderr, "[generate] prompt too long (%d >= %d)\n",
                n_prompt, mcfg->max_seq_len);
        free(prompt_tokens); return NULL;
    }

    /* Output buffer — grows as we generate */
    size_t out_cap  = 4096;
    char  *out_text = (char *)malloc(out_cap);
    size_t out_len  = 0;
    out_text[0] = '\0';

    /* ------------------------------------------------------------------ */
    /* Prefill: run forward pass over all prompt tokens to populate cache  */
    /* The last prompt token's logits seed the first new token.            */
    /* ------------------------------------------------------------------ */
    int last_token = prompt_tokens[0];
    for (int i = 0; i < n_prompt; i++) {
        inference_step(m, state, prompt_tokens[i]);
        last_token = prompt_tokens[i];
    }

    /* ------------------------------------------------------------------ */
    /* Generate new tokens                                                  */
    /* ------------------------------------------------------------------ */
    int generated = 0;
    while (generated < gcfg->max_new_tokens &&
           state->seq_pos < mcfg->max_seq_len) {

        /* Sample the next token from the last step's logits */
        /* (copy logits so sample_token can modify them in-place) */
        float *logits_copy = (float *)malloc(mcfg->vocab_size * sizeof(float));
        memcpy(logits_copy, state->logits, mcfg->vocab_size * sizeof(float));

        int next_token = sample_token(logits_copy, mcfg->vocab_size,
                                      &gcfg->sampler, &rng);
        free(logits_copy);

        /* Check stop tokens */
        int stop = 0;
        for (int i = 0; i < gcfg->n_stop_tokens; i++)
            if (next_token == gcfg->stop_tokens[i]) { stop = 1; break; }
        if (stop && generated >= gcfg->min_new_tokens) break;

        /* Decode and append to output */
        const char *tok_str = tok_decode_token(tok, next_token);
        size_t tok_len = strlen(tok_str);

        /* Grow output buffer if needed */
        while (out_len + tok_len + 1 > out_cap) {
            out_cap *= 2;
            out_text = (char *)realloc(out_text, out_cap);
        }
        memcpy(out_text + out_len, tok_str, tok_len);
        out_len += tok_len;
        out_text[out_len] = '\0';

        /* Streaming callback */
        if (gcfg->on_token) {
            if (gcfg->on_token(next_token, tok_str, gcfg->user_data) != 0)
                break;
        }

        /* Advance: run the forward pass with the new token */
        inference_step(m, state, next_token);
        last_token = next_token;
        generated++;
    }

    free(prompt_tokens);
    (void)last_token;
    return out_text;
}
