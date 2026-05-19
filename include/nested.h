/**
 * nested.h — Nested Learning extension for llm.c
 *
 * Based on Google Research NeurIPS 2025 paper:
 * "Nested Learning: The Illusion of Deep Learning Architectures"
 *
 * Three ideas implemented:
 *
 *  1. ContinuumMemoryBlock (CMB)
 *     Persistent key-value memory bank alongside attention.
 *     Unlike attention (reads context window), CMB survives across sequences.
 *     Updated at a configurable slow rate.
 *
 *  2. Multi-rate parameter groups
 *     Early layers / embeddings update less frequently and with a smaller LR.
 *     Mirrors the brain's "offline consolidation" — stable base knowledge
 *     changes slowly, recent associations change quickly.
 *
 *  3. Deep optimiser
 *     Adds a slow-moving third momentum term (m3) that tracks long-term
 *     gradient trends. Acts as an internal memory of the optimisation
 *     trajectory, not just the gradient.
 */

#ifndef LLM_NESTED_H
#define LLM_NESTED_H

#include "matrix.h"
#include "config.h"
#include <stdint.h>

/* ── Nested Learning configuration ───────────────────────────────────────── */
typedef struct {
    int   use_cmb;         /* 0=off, 1=add CMB to each block              */
    int   cmb_slots;       /* persistent memory slots (key-value pairs)   */
    int   cmb_dim;         /* CMB internal dimension                      */
    float cmb_lr_scale;    /* LR multiplier for CMB (default 0.1)         */
    int   cmb_update_freq; /* update CMB every N steps (default 4)        */

    float embed_lr_scale;  /* LR scale for embeddings (default 0.5)       */
    float early_lr_scale;  /* LR scale for first half of layers (0.3)     */
    float late_lr_scale;   /* LR scale for second half + LM head (1.0)    */

    int   use_deep_optim;  /* 1 = add slow m3 momentum term               */
} NLConfig;

static inline NLConfig nl_config_default(void) {
    return (NLConfig){
        .use_cmb         = 1,
        .cmb_slots       = 64,
        .cmb_dim         = 128,
        .cmb_lr_scale    = 0.1f,
        .cmb_update_freq = 4,
        .embed_lr_scale  = 0.5f,
        .early_lr_scale  = 0.3f,
        .late_lr_scale   = 1.0f,
        .use_deep_optim  = 1,
    };
}

/* ── Continuum Memory Block ───────────────────────────────────────────────── */
typedef struct {
    Matrix keys;    /* [cmb_slots × cmb_dim]  persistent memory keys        */
    Matrix values;  /* [cmb_slots × cmb_dim]  persistent memory values      */
    Matrix W_q;     /* [d_model   × cmb_dim]  query projection              */
    Matrix W_out;   /* [cmb_dim   × d_model]  output projection             */
    float *rms_w;   /* [d_model]  RMSNorm scale for CMB input               */
} CMBWeights;

typedef struct {
    Matrix x_norm;    /* [seq × d_model]    input after RMSNorm             */
    Matrix queries;   /* [seq × cmb_dim]    query vectors                   */
    Matrix scores;    /* [seq × cmb_slots]  softmax attention scores        */
    Matrix retrieved; /* [seq × cmb_dim]    weighted value retrieval        */
    Matrix pre_out;   /* [seq × d_model]    before residual add             */
} CMBCache;

/* ── Multi-rate AdamW slot ────────────────────────────────────────────────── */
typedef struct {
    float  lr_scale;    /* effective_lr = base_lr × lr_scale              */
    int    update_freq; /* only update on steps where step%freq==offset   */
    int    step_offset;
    float *m1, *m2;     /* Adam first/second moments                      */
    float *m3;          /* deep optimiser: slow trend moment (or NULL)    */
    int    n;
} MRSlot;

typedef struct {
    MRSlot *slots;
    int     n_slots;
    int     step;
} MRAdamState;

/* ── Lifecycle ────────────────────────────────────────────────────────────── */
CMBWeights  cmb_alloc(int d_model, int cmb_slots, int cmb_dim);
void        cmb_free(CMBWeights *c);
CMBCache    cmb_cache_alloc(int seq_len, int cmb_slots, int cmb_dim, int d_model);
void        cmb_cache_free(CMBCache *c);

/* ── Forward / backward ───────────────────────────────────────────────────── */
void cmb_forward (Matrix *x, const CMBWeights *cmb, CMBCache *cache, float eps);
void cmb_backward(const Matrix *x_in, CMBWeights *cmb,
                  const CMBCache *cache, Matrix *d_x, float eps);

/* ── Multi-rate optimiser ─────────────────────────────────────────────────── */
MRAdamState *mr_adam_init(int n_params, float *lr_scales, int *update_freqs,
                           int *ns, int use_deep_optim);
void         mr_adam_free(MRAdamState *s);
void         mr_adam_step(MRAdamState *s,
                           float base_lr, float beta1, float beta2,
                           float eps, float wd,
                           float **data, float **grad, int n_params);

/* ── Persistence ─────────────────────────────────────────────────────────── */
int cmb_save(const CMBWeights *cmb, const char *path);
int cmb_load(CMBWeights *cmb, const char *path);

#endif /* LLM_NESTED_H */
