/**
 * nested.c — Nested Learning implementation
 *
 * CMB forward/backward, multi-rate AdamW, deep optimiser.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "../include/nested.h"
#include "../include/simd.h"

#ifdef USE_OPENBLAS
#  include <cblas.h>
#endif

static void *xc(size_t n) {
    void *p = calloc(1, n);
    if (!p) { fprintf(stderr, "[nested] OOM\n"); abort(); }
    return p;
}

/* =========================================================================
 * Continuum Memory Block
 * ====================================================================== */

CMBWeights cmb_alloc(int d_model, int cmb_slots, int cmb_dim) {
    CMBWeights c;
    c.keys   = mat_alloc(cmb_slots, cmb_dim, 1);
    c.values = mat_alloc(cmb_slots, cmb_dim, 1);
    c.W_q    = mat_alloc(d_model,   cmb_dim, 1);
    c.W_out  = mat_alloc(cmb_dim,   d_model, 1);
    c.rms_w  = (float *)malloc(d_model * sizeof(float));

    mat_randn(&c.keys,   sqrtf(1.f / cmb_dim));
    mat_randn(&c.values, sqrtf(1.f / cmb_dim));
    mat_randn(&c.W_q,    sqrtf(2.f / d_model));
    mat_randn(&c.W_out,  sqrtf(2.f / cmb_dim));
    for (int i = 0; i < d_model; i++) c.rms_w[i] = 1.f;
    return c;
}

void cmb_free(CMBWeights *c) {
    mat_free(&c->keys); mat_free(&c->values);
    mat_free(&c->W_q);  mat_free(&c->W_out);
    free(c->rms_w); c->rms_w = NULL;
}

CMBCache cmb_cache_alloc(int seq, int slots, int cmb_dim, int d_model) {
    CMBCache c;
    c.x_norm    = mat_alloc(seq, d_model, 0);
    c.queries   = mat_alloc(seq, cmb_dim, 0);
    c.scores    = mat_alloc(seq, slots,   0);
    c.retrieved = mat_alloc(seq, cmb_dim, 0);
    c.pre_out   = mat_alloc(seq, d_model, 0);
    return c;
}

void cmb_cache_free(CMBCache *c) {
    mat_free(&c->x_norm); mat_free(&c->queries);
    mat_free(&c->scores); mat_free(&c->retrieved);
    mat_free(&c->pre_out);
}

/* ── CMB forward ─────────────────────────────────────────────────────────────
 *
 * 1. x_norm = RMSNorm(x)
 * 2. queries = x_norm · W_q          [seq × cmb_dim]
 * 3. scores  = softmax(queries · keysᵀ) / sqrt(cmb_dim)  [seq × slots]
 * 4. retrieved = scores · values     [seq × cmb_dim]
 * 5. pre_out   = retrieved · W_out   [seq × d_model]
 * 6. x += pre_out                    (residual)
 */
void cmb_forward(Matrix *x, const CMBWeights *cmb,
                 CMBCache *cache, float eps) {
    int seq   = x->rows;
    int d     = x->cols;
    int slots = cmb->keys.rows;
    int cdim  = cmb->keys.cols;
    float scale = 1.f / sqrtf((float)cdim);

    /* 1. RMSNorm */
    rms_norm(x, cmb->rms_w, eps, &cache->x_norm);

    /* 2. queries = x_norm · W_q */
#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq, cdim, d, 1.f,
                cache->x_norm.data, d, cmb->W_q.data, cdim,
                0.f, cache->queries.data, cdim);
#else
    mat_mul(&cache->x_norm, &cmb->W_q, &cache->queries);
#endif

    /* 3. scores = softmax(queries · keysᵀ * scale) */
#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, slots, cdim, scale,
                cache->queries.data, cdim, cmb->keys.data, cdim,
                0.f, cache->scores.data, slots);
#else
    mat_mul_T(&cache->queries, &cmb->keys, &cache->scores);
    mat_scale(&cache->scores, scale);
#endif
    softmax_rows(&cache->scores);

    /* 4. retrieved = scores · values */
#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq, cdim, slots, 1.f,
                cache->scores.data, slots, cmb->values.data, cdim,
                0.f, cache->retrieved.data, cdim);
#else
    mat_mul(&cache->scores, &cmb->values, &cache->retrieved);
#endif

    /* 5. pre_out = retrieved · W_out */
#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq, d, cdim, 1.f,
                cache->retrieved.data, cdim, cmb->W_out.data, d,
                0.f, cache->pre_out.data, d);
#else
    mat_mul(&cache->retrieved, &cmb->W_out, &cache->pre_out);
#endif

    /* 6. residual add */
    mat_add_inplace(x, &cache->pre_out);
}

/* ── CMB backward ────────────────────────────────────────────────────────────
 *
 * Backprop through all 5 steps and accumulate weight gradients.
 * d_x is the gradient flowing in from above (modified in-place).
 */
void cmb_backward(const Matrix *x_in, CMBWeights *cmb,
                  const CMBCache *cache, Matrix *d_x, float eps) {
    int seq   = x_in->rows;
    int d     = x_in->cols;
    int slots = cmb->keys.rows;
    int cdim  = cmb->keys.cols;
    float scale = 1.f / sqrtf((float)cdim);

    /* Step 6 gradient: d_pre_out = d_x (residual path passes through) */
    /* d_pre_out IS d_x since pre_out was added to x */

    /* Step 5: pre_out = retrieved · W_out
     *  dW_out    += retrievedᵀ · d_x        [cdim×seq]×[seq×d]
     *  d_retrieved = d_x · W_outᵀ           [seq×d]×[d×cdim]   */
    float *d_retrieved = (float *)xc((size_t)seq * cdim * sizeof(float));
    float *d_scores    = (float *)xc((size_t)seq * slots * sizeof(float));
    float *d_queries   = (float *)xc((size_t)seq * cdim * sizeof(float));
    float *d_xnorm     = (float *)xc((size_t)seq * d * sizeof(float));

#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                cdim, d, seq, 1.f,
                cache->retrieved.data, cdim, d_x->data, d,
                1.f, cmb->W_out.grad, d);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, cdim, d, 1.f,
                d_x->data, d, cmb->W_out.data, d,
                0.f, d_retrieved, cdim);

    /* Step 4: retrieved = scores · values
     *  dW_values  += scoresᵀ · d_retrieved   [slots×seq]×[seq×cdim]
     *  d_scores   += d_retrieved · valuesᵀ   [seq×cdim]×[cdim×slots]  */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                slots, cdim, seq, 1.f,
                cache->scores.data, slots, d_retrieved, cdim,
                1.f, cmb->values.grad, cdim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, slots, cdim, 1.f,
                d_retrieved, cdim, cmb->values.data, cdim,
                0.f, d_scores, slots);
#else
    /* Fallback: manual accumulation */
    for (int j = 0; j < cdim; j++)
        for (int k = 0; k < d; k++) {
            float s = 0.f;
            for (int i = 0; i < seq; i++) s += cache->retrieved.data[i*cdim+j] * d_x->data[i*d+k];
            cmb->W_out.grad[j*d+k] += s;
        }
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < cdim; j++) {
            float s = 0.f;
            for (int k = 0; k < d; k++) s += d_x->data[i*d+k] * cmb->W_out.data[j*d+k];
            d_retrieved[i*cdim+j] = s;
        }
    for (int t = 0; t < slots; t++)
        for (int k = 0; k < cdim; k++) {
            float s = 0.f;
            for (int i = 0; i < seq; i++) s += cache->scores.data[i*slots+t] * d_retrieved[i*cdim+k];
            cmb->values.grad[t*cdim+k] += s;
        }
    for (int i = 0; i < seq; i++)
        for (int t = 0; t < slots; t++) {
            float s = 0.f;
            for (int k = 0; k < cdim; k++) s += d_retrieved[i*cdim+k] * cmb->values.data[t*cdim+k];
            d_scores[i*slots+t] = s;
        }
#endif

    /* Step 3: softmax backward
     *   d_raw[i][t] = scores[i][t] * (d_scores[i][t] - Σ_j scores[i][j]*d_scores[i][j])
     *   then multiply by scale */
    for (int i = 0; i < seq; i++) {
        const float *si = cache->scores.data + i * slots;
        float       *di = d_scores           + i * slots;
        float dot = 0.f;
        for (int t = 0; t < slots; t++) dot += si[t] * di[t];
        for (int t = 0; t < slots; t++) di[t] = si[t] * (di[t] - dot) * scale;
    }

#ifdef USE_OPENBLAS
    /* dW_keys += d_scoresᵀ · queries   [slots×seq]×[seq×cdim] */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                slots, cdim, seq, 1.f,
                d_scores, slots, cache->queries.data, cdim,
                1.f, cmb->keys.grad, cdim);
    /* d_queries = d_scores · keys       [seq×slots]×[slots×cdim] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq, cdim, slots, 1.f,
                d_scores, slots, cmb->keys.data, cdim,
                0.f, d_queries, cdim);
    /* Step 2: dW_q += x_normᵀ · d_queries */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, cdim, seq, 1.f,
                cache->x_norm.data, d, d_queries, cdim,
                1.f, cmb->W_q.grad, cdim);
    /* d_xnorm = d_queries · W_qᵀ */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, d, cdim, 1.f,
                d_queries, cdim, cmb->W_q.data, cdim,
                0.f, d_xnorm, d);
#else
    for (int t = 0; t < slots; t++)
        for (int k = 0; k < cdim; k++) {
            float s = 0.f;
            for (int i = 0; i < seq; i++) s += d_scores[i*slots+t] * cache->queries.data[i*cdim+k];
            cmb->keys.grad[t*cdim+k] += s;
        }
    for (int i = 0; i < seq; i++)
        for (int k = 0; k < cdim; k++) {
            float s = 0.f;
            for (int t = 0; t < slots; t++) s += d_scores[i*slots+t] * cmb->keys.data[t*cdim+k];
            d_queries[i*cdim+k] = s;
        }
    for (int j = 0; j < d; j++)
        for (int k = 0; k < cdim; k++) {
            float s = 0.f;
            for (int i = 0; i < seq; i++) s += cache->x_norm.data[i*d+j] * d_queries[i*cdim+k];
            cmb->W_q.grad[j*cdim+k] += s;
        }
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < d; j++) {
            float s = 0.f;
            for (int k = 0; k < cdim; k++) s += d_queries[i*cdim+k] * cmb->W_q.data[j*cdim+k];
            d_xnorm[i*d+j] = s;
        }
#endif

    /* Step 1: RMSNorm backward → accumulates into d_x */
    float *d_rms = (float *)xc(d * sizeof(float));
    Matrix d_xn_mat = { d_xnorm, NULL, seq, d };
    Matrix d_x_rms  = { (float *)xc((size_t)seq*d*sizeof(float)), NULL, seq, d };
    rms_norm_backward(x_in, cmb->rms_w, eps, &d_xn_mat, &d_x_rms, d_rms);
    /* Accumulate d_x_rms into d_x */
    for (int i = 0; i < seq * d; i++) d_x->data[i] += d_x_rms.data[i];

    free(d_x_rms.data); free(d_rms);
    free(d_retrieved); free(d_scores); free(d_queries); free(d_xnorm);
}

/* =========================================================================
 * Multi-rate AdamW  +  Deep Optimiser
 * ====================================================================== */

MRAdamState *mr_adam_init(int n_params, float *lr_scales, int *update_freqs,
                           int *ns, int use_deep_optim) {
    MRAdamState *s = (MRAdamState *)xc(sizeof(MRAdamState));
    s->n_slots = n_params;
    s->slots   = (MRSlot *)xc((size_t)n_params * sizeof(MRSlot));
    s->step    = 0;

    for (int i = 0; i < n_params; i++) {
        MRSlot *sl = &s->slots[i];
        sl->lr_scale    = lr_scales    ? lr_scales[i]    : 1.0f;
        sl->update_freq = update_freqs ? update_freqs[i] : 1;
        sl->step_offset = i % (update_freqs ? update_freqs[i] : 1);
        sl->n           = ns[i];
        sl->m1 = (float *)xc((size_t)ns[i] * sizeof(float));
        sl->m2 = (float *)xc((size_t)ns[i] * sizeof(float));
        sl->m3 = use_deep_optim
               ? (float *)xc((size_t)ns[i] * sizeof(float))
               : NULL;
    }
    return s;
}

void mr_adam_free(MRAdamState *s) {
    if (!s) return;
    for (int i = 0; i < s->n_slots; i++) {
        free(s->slots[i].m1);
        free(s->slots[i].m2);
        free(s->slots[i].m3);
    }
    free(s->slots);
    free(s);
}

/**
 * mr_adam_step — one multi-rate AdamW step.
 *
 * Deep optimiser (when m3 != NULL):
 *   m3 = beta_slow * m3 + (1 - beta_slow) * m1
 *   effective_grad = m_hat * (1 - gamma) + m3_hat * gamma
 *
 * This blends the standard Adam estimate with a very slow-moving trend,
 * letting the optimiser "remember" the long-term gradient direction.
 * Parameters pulled consistently in one direction get a persistent boost;
 * oscillating gradients are damped.
 */
void mr_adam_step(MRAdamState *s,
                  float base_lr, float beta1, float beta2,
                  float eps, float wd,
                  float **data, float **grad, int n_params) {
    s->step++;
    float t     = (float)s->step;
    float bc1   = 1.f - powf(beta1, t);
    float bc2   = 1.f - powf(beta2, t);
    float beta_slow = 0.995f;  /* slow momentum for m3 */
    float gamma     = 0.1f;    /* blend weight: 0=pure Adam, 1=pure slow   */

    assert(n_params == s->n_slots);

    for (int i = 0; i < n_params; i++) {
        MRSlot *sl = &s->slots[i];

        /* Skip if this slot is not due for update this step */
        if (sl->update_freq > 1 &&
            (s->step % sl->update_freq) != sl->step_offset)
            continue;

        float  effective_lr = base_lr * sl->lr_scale;
        float *p = data[i];
        float *g = grad[i];
        int    n = sl->n;

        for (int j = 0; j < n; j++) {
            float gj = g[j];

            /* Standard AdamW moments */
            sl->m1[j] = beta1 * sl->m1[j] + (1.f - beta1) * gj;
            sl->m2[j] = beta2 * sl->m2[j] + (1.f - beta2) * gj * gj;

            float m1h = sl->m1[j] / bc1;
            float m2h = sl->m2[j] / bc2;

            float update;
            if (sl->m3) {
                /* Deep optimiser: blend Adam estimate with slow trend */
                sl->m3[j] = beta_slow * sl->m3[j]
                           + (1.f - beta_slow) * sl->m1[j];
                float bc3  = 1.f - powf(beta_slow, t);
                float m3h  = sl->m3[j] / bc3;
                /* Blended update direction */
                float adam_dir = m1h / (sqrtf(m2h) + eps);
                float slow_dir = m3h / (sqrtf(m2h) + eps);
                update = (1.f - gamma) * adam_dir + gamma * slow_dir;
            } else {
                update = m1h / (sqrtf(m2h) + eps);
            }

            /* AdamW weight decay (decoupled from gradient) */
            p[j] *= (1.f - effective_lr * wd);
            p[j] -= effective_lr * update;
        }
    }
}

/* =========================================================================
 * CMB persistence
 * ====================================================================== */

int cmb_save(const CMBWeights *cmb, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    int32_t slots = cmb->keys.rows;
    int32_t cdim  = cmb->keys.cols;
    int32_t d     = cmb->W_q.rows;
    fwrite(&slots, 4, 1, f); fwrite(&cdim, 4, 1, f); fwrite(&d, 4, 1, f);
    fwrite(cmb->keys.data,   sizeof(float), (size_t)slots*cdim, f);
    fwrite(cmb->values.data, sizeof(float), (size_t)slots*cdim, f);
    fwrite(cmb->W_q.data,    sizeof(float), (size_t)d*cdim,     f);
    fwrite(cmb->W_out.data,  sizeof(float), (size_t)cdim*d,     f);
    fwrite(cmb->rms_w,       sizeof(float), (size_t)d,          f);
    fclose(f);
    return 0;
}

int cmb_load(CMBWeights *cmb, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    int32_t slots, cdim, d;
    if (fread(&slots,4,1,f)!=1||fread(&cdim,4,1,f)!=1||fread(&d,4,1,f)!=1) {
        fclose(f); return -1;
    }
    int ok = 1;
    ok &= (fread(cmb->keys.data,   sizeof(float),(size_t)slots*cdim,f)==(size_t)slots*cdim);
    ok &= (fread(cmb->values.data, sizeof(float),(size_t)slots*cdim,f)==(size_t)slots*cdim);
    ok &= (fread(cmb->W_q.data,    sizeof(float),(size_t)d*cdim,    f)==(size_t)d*cdim);
    ok &= (fread(cmb->W_out.data,  sizeof(float),(size_t)cdim*d,    f)==(size_t)cdim*d);
    ok &= (fread(cmb->rms_w,       sizeof(float),(size_t)d,         f)==(size_t)d);
    if (!ok) { fprintf(stderr,"[cmb_load] short read\n"); fclose(f); return -1; }
    fclose(f);
    return 0;
}
