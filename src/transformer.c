/**
 * transformer.c
 *
 * Implementation of:
 *   - Model init / free / save / load
 *   - Transformer block forward (pre-norm, residual)
 *   - Dense SwiGLU FFN forward / backward
 *   - Mixture-of-Experts FFN forward
 *   - Full model forward pass
 *   - Full model backward pass (cross-entropy loss)
 *   - Gradient zeroing
 */

#define _GNU_SOURCE  /* expose M_PI on Linux */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>  /* uint32_t for save/load magic number */

#include "../include/transformer.h"

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "[transformer] OOM\n"); abort(); }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "[transformer] OOM\n"); abort(); }
    return p;
}


/* =========================================================================
 * TrainScratch — pre-allocated backward-pass buffers.
 *
 * model_backward() previously called mat_alloc/mat_free ~30 times per step.
 * On a 5000-word corpus with batch_size=4 that's thousands of malloc calls
 * per second — pure overhead.  We allocate once at model_init() and reuse.
 * ====================================================================== */
typedef struct {
    /* Residual stream states: x_states[l] = input to block l,
     * x_states[n_layers] = final residual after all blocks.           */
    Matrix *x_states;   /* [n_layers+1] × [seq × d_model]              */
    int     n_states;

    /* Scratch for final norm, LM head, and top-level grad flow */
    Matrix x_norm_final;  /* [seq × d_model]   */
    Matrix logits;        /* [seq × vocab]      */
    Matrix d_logits;      /* [seq × vocab]      */
    Matrix d_x_norm;      /* [seq × d_model]    */
    Matrix d_x_top;       /* [seq × d_model]    */
    Matrix d_x;           /* [seq × d_model]    — flowing gradient      */
    Matrix d_x_new;       /* [seq × d_model]    — per-layer output      */

    /* Per-layer backward scratch (reused each layer) */
    Matrix d_x_norm_attn; /* [seq × d_model]    */
    Matrix d_x_norm_ffn;  /* [seq × d_model]    */

    /* rms_norm weight gradients (accumulated, then flushed each step) */
    float *d_rms_final;   /* [d_model]          */
    float *d_rms_attn;    /* [d_model]          */
    float *d_rms_ffn;     /* [d_model]          */

    int seq_len;   /* the seq_len this scratch was sized for */
} TrainScratch;

static TrainScratch *train_scratch_alloc(const ModelConfig *cfg, int seq_len) {
    TrainScratch *s = (TrainScratch *)xcalloc(1, sizeof(TrainScratch));
    int d = cfg->d_model;
    int V = cfg->vocab_size;
    int L = cfg->n_layers;

    s->seq_len  = seq_len;
    s->n_states = L + 1;
    s->x_states = (Matrix *)xmalloc((L + 1) * sizeof(Matrix));
    for (int i = 0; i <= L; i++)
        s->x_states[i] = mat_alloc(seq_len, d, 0);

    s->x_norm_final  = mat_alloc(seq_len, d, 0);
    s->logits        = mat_alloc(seq_len, V, 0);
    s->d_logits      = mat_alloc(seq_len, V, 0);
    s->d_x_norm      = mat_alloc(seq_len, d, 0);
    s->d_x_top       = mat_alloc(seq_len, d, 0);
    s->d_x           = mat_alloc(seq_len, d, 0);
    s->d_x_new       = mat_alloc(seq_len, d, 0);
    s->d_x_norm_attn = mat_alloc(seq_len, d, 0);
    s->d_x_norm_ffn  = mat_alloc(seq_len, d, 0);

    s->d_rms_final = (float *)xcalloc(d, sizeof(float));
    s->d_rms_attn  = (float *)xcalloc(d, sizeof(float));
    s->d_rms_ffn   = (float *)xcalloc(d, sizeof(float));
    return s;
}

static void train_scratch_free(TrainScratch *s) {
    if (!s) return;
    for (int i = 0; i < s->n_states; i++) mat_free(&s->x_states[i]);
    free(s->x_states);
    mat_free(&s->x_norm_final);
    mat_free(&s->logits);
    mat_free(&s->d_logits);
    mat_free(&s->d_x_norm);
    mat_free(&s->d_x_top);
    mat_free(&s->d_x);
    mat_free(&s->d_x_new);
    mat_free(&s->d_x_norm_attn);
    mat_free(&s->d_x_norm_ffn);
    free(s->d_rms_final);
    free(s->d_rms_attn);
    free(s->d_rms_ffn);
    free(s);
}

/* =========================================================================
 * FFN weights
 * ====================================================================== */

FFNWeights ffn_weights_alloc(int d_model, int ffn_hidden, int with_grad) {
    FFNWeights w;
    w.W_gate = mat_alloc(d_model,    ffn_hidden, with_grad);
    w.W_up   = mat_alloc(d_model,    ffn_hidden, with_grad);
    w.W_down = mat_alloc(ffn_hidden, d_model,    with_grad);

    /* Small init for gate/up, scaled by depth later in model_init */
    float std_in  = sqrtf(2.0f / d_model);
    float std_out = sqrtf(2.0f / ffn_hidden);
    mat_randn(&w.W_gate, std_in);
    mat_randn(&w.W_up,   std_in);
    mat_randn(&w.W_down, std_out);
    return w;
}

void ffn_weights_free(FFNWeights *w) {
    mat_free(&w->W_gate);
    mat_free(&w->W_up);
    mat_free(&w->W_down);
}

/**
 * ffn_forward — dense SwiGLU FFN.
 *
 *   gate_buf = x · W_gate          [seq × ffn_hidden]
 *   up_buf   = x · W_up            [seq × ffn_hidden]
 *   h        = swiglu(gate, up)    [seq × ffn_hidden]  (silu(gate) ⊙ up)
 *   out      = h · W_down          [seq × d_model]
 *
 * gate_buf and up_buf are scratch matrices owned by ForwardCache.
 */
void ffn_forward(const Matrix *x, const FFNWeights *w,
                 const ModelConfig *cfg,
                 Matrix *gate_buf, Matrix *up_buf,
                 Matrix *out) {
    mat_mul(x, &w->W_gate, gate_buf);   /* gate pre-activation */
    mat_mul(x, &w->W_up,   up_buf);     /* up   pre-activation */
    swiglu(gate_buf, up_buf, gate_buf); /* reuse gate_buf as h  */
    mat_mul(gate_buf, &w->W_down, out); /* project back         */
}

/**
 * ffn_backward — backprop through the dense SwiGLU FFN.
 *
 * Requires the saved gate_buf (post-swiglu) and up_buf (pre-swiglu).
 * We save them in ForwardCache so they're available here.
 *
 * Variable naming:
 *   gate_pre  = x · W_gate   (before SwiGLU, saved in cache as gate_pre)
 *   up_pre    = x · W_up     (before SwiGLU, saved in cache as up_buf)
 *   h         = swiglu(gate_pre, up_pre)
 *   out       = h · W_down
 */
/* ffn_backward is called from model_backward (Phase 2).
 * Declared non-static so the linker can resolve it across TUs. */
void ffn_backward(const Matrix *x,
                         const FFNWeights *w,
                         const Matrix *gate_pre,  /* saved: x · W_gate     */
                         const Matrix *up_pre,    /* saved: x · W_up       */
                         const Matrix *h,         /* saved: swiglu output  */
                         const Matrix *d_out,     /* incoming gradient      */
                         Matrix *d_x) {
    int seq      = x->rows;
    int d_model  = x->cols;
    int ffn_h    = w->W_down.rows;

    /* d_h = d_out · W_downᵀ  [seq × ffn_hidden] */
    Matrix d_h = mat_alloc(seq, ffn_h, 0);
    mat_zero(&d_h);
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < ffn_h; j++) {
            float s = 0.0f;
            for (int k = 0; k < d_model; k++)
                s += d_out->data[i * d_model + k] * w->W_down.data[j * d_model + k];
            d_h.data[i * ffn_h + j] = s;
        }
    }

    /* dW_down += hᵀ · d_out */
    for (int j = 0; j < ffn_h; j++)
        for (int k = 0; k < d_model; k++) {
            float s = 0.0f;
            for (int i = 0; i < seq; i++)
                s += h->data[i * ffn_h + j] * d_out->data[i * d_model + k];
            w->W_down.grad[j * d_model + k] += s;
        }

    /*
     * Backprop through SwiGLU:
     *   h[i] = silu(gate_pre[i]) * up_pre[i]
     *   d_gate_pre[i] = d_h[i] * up_pre[i]  * d_silu(gate_pre[i])
     *   d_up_pre[i]   = d_h[i] * silu(gate_pre[i])
     *
     * d_silu(x) = silu(x) + sigmoid(x) * (1 - silu(x))
     *           = σ(x)(1 + x(1 - σ(x)))
     */
    Matrix d_gate_pre = mat_alloc(seq, ffn_h, 0);
    Matrix d_up_pre   = mat_alloc(seq, ffn_h, 0);
    int n = seq * ffn_h;
    for (int i = 0; i < n; i++) {
        float g   = gate_pre->data[i];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        float d_silu = sig * (1.0f + g * (1.0f - sig));

        d_gate_pre.data[i] = d_h.data[i] * up_pre->data[i] * d_silu;
        d_up_pre.data[i]   = d_h.data[i] * silu_g;
    }

    /* dW_gate += xᵀ · d_gate_pre,  dW_up += xᵀ · d_up_pre */
    for (int j = 0; j < d_model; j++)
        for (int k = 0; k < ffn_h; k++) {
            float sg = 0.0f, su = 0.0f;
            for (int i = 0; i < seq; i++) {
                sg += x->data[i * d_model + j] * d_gate_pre.data[i * ffn_h + k];
                su += x->data[i * d_model + j] * d_up_pre.data[i * ffn_h + k];
            }
            w->W_gate.grad[j * ffn_h + k] += sg;
            w->W_up.grad[j * ffn_h + k]   += su;
        }

    /* d_x += d_gate_pre · W_gateᵀ + d_up_pre · W_upᵀ */
    mat_zero(d_x);
    for (int i = 0; i < seq; i++) {
        float *dx = d_x->data + i * d_model;
        for (int k = 0; k < ffn_h; k++) {
            float dg = d_gate_pre.data[i * ffn_h + k];
            float du = d_up_pre.data[i * ffn_h + k];
            for (int j = 0; j < d_model; j++) {
                dx[j] += dg * w->W_gate.data[j * ffn_h + k];
                dx[j] += du * w->W_up.data[j * ffn_h + k];
            }
        }
    }

    mat_free(&d_h);
    mat_free(&d_gate_pre);
    mat_free(&d_up_pre);
}

/* =========================================================================
 * Mixture of Experts forward
 * ====================================================================== */

/**
 * moe_forward — route each token to top_k experts, combine outputs.
 *
 * Algorithm:
 *   1. router_logits = x · W_router   [seq × n_experts]
 *   2. router_probs  = softmax(router_logits) row-wise
 *   3. For each token, pick top_k experts by probability
 *   4. out[token] = sum over selected experts:
 *                     prob[expert] * expert_ffn(x[token])
 *
 * Load-balancing auxiliary loss (Switch Transformer):
 *   aux_loss = n_experts * sum_e (fraction_tokens_e * mean_prob_e)
 * Stored in cache->router_probs so the training loop can compute it.
 *
 * Note: For now this is a clean readable implementation.
 *       A production version would parallelise expert computation.
 */
void moe_forward(const Matrix *x, const MoEWeights *mw,
                 const ModelConfig *cfg, ForwardCache *cache,
                 Matrix *out) {
    int seq       = x->rows;
    int d_model   = x->cols;
    int n_exp     = mw->n_experts;
    int top_k     = mw->top_k;
    int ffn_h     = cfg->ffn_hidden;

    mat_zero(out);

    /* --- Step 1+2: router logits and probabilities -------------------- */
    Matrix router_logits = mat_alloc(seq, n_exp, 0);
    mat_mul(x, &mw->W_router, &router_logits);
    softmax_rows(&router_logits);   /* now router_probs */

    /* Save for backward / aux loss */
    memcpy(cache->router_probs, router_logits.data,
           (size_t)seq * n_exp * sizeof(float));

    /* --- Step 3+4: per-token expert dispatch -------------------------- */
    Matrix expert_out  = mat_alloc(1, d_model, 0);  /* single-token scratch */
    Matrix gate_scratch= mat_alloc(1, ffn_h,   0);
    Matrix up_scratch  = mat_alloc(1, ffn_h,   0);

    for (int t = 0; t < seq; t++) {
        /* Find top_k expert indices for this token */
        float *probs = router_logits.data + t * n_exp;
        int   *ids   = cache->expert_ids  + t * top_k;

        /* Simple selection sort for top_k (k is tiny, usually 1 or 2) */
        for (int ki = 0; ki < top_k; ki++) {
            float best = -1.0f;
            int   best_e = 0;
            for (int e = 0; e < n_exp; e++) {
                /* Skip already-selected experts */
                int already = 0;
                for (int kj = 0; kj < ki; kj++)
                    if (ids[kj] == e) { already = 1; break; }
                if (!already && probs[e] > best) {
                    best = probs[e];
                    best_e = e;
                }
            }
            ids[ki] = best_e;
        }

        /* Normalise selected probabilities so they sum to 1.0 */
        float prob_sum = 0.0f;
        for (int ki = 0; ki < top_k; ki++) prob_sum += probs[ids[ki]];
        float inv_sum = (prob_sum > 1e-9f) ? 1.0f / prob_sum : 0.0f;

        /* Wrap single token row as a 1×d_model matrix for ffn_forward */
        Matrix x_tok = {
            .data = x->data + t * d_model,
            .grad = NULL,
            .rows = 1,
            .cols = d_model,
        };

        /* Accumulate expert outputs */
        for (int ki = 0; ki < top_k; ki++) {
            int e = ids[ki];
            float w = probs[e] * inv_sum;

            ffn_forward(&x_tok, &mw->experts[e], cfg,
                        &gate_scratch, &up_scratch, &expert_out);

            /* out[t] += weight * expert_out */
            float *out_row = out->data + t * d_model;
            for (int j = 0; j < d_model; j++)
                out_row[j] += w * expert_out.data[j];
        }
    }

    mat_free(&router_logits);
    mat_free(&expert_out);
    mat_free(&gate_scratch);
    mat_free(&up_scratch);
}

/* =========================================================================
 * ForwardCache allocation helpers
 * ====================================================================== */

static ForwardCache cache_alloc(const ModelConfig *cfg, int seq_len) {
    int d    = cfg->d_model;
    int nh   = cfg->n_heads;
    int nkv  = cfg->n_kv_heads;
    int hd   = cfg->head_dim;
    int ffnh = cfg->ffn_hidden;
    int ne   = cfg->n_experts;
    int tk   = cfg->top_k_experts;

    ForwardCache c;
    c.x_norm_attn   = mat_alloc(seq_len, d,         0);
    c.x_norm_ffn    = mat_alloc(seq_len, d,         0);
    c.attn_out      = mat_alloc(seq_len, d,         0);
    c.ffn_out       = mat_alloc(seq_len, d,         0);

    c.Q             = mat_alloc(seq_len, nh  * hd,  0);
    c.K             = mat_alloc(seq_len, nkv * hd,  0);
    c.V             = mat_alloc(seq_len, nkv * hd,  0);
    /* scores: one [seq×seq] block per query head, stored flat */
    c.attn_scores   = mat_alloc(nh * seq_len, seq_len, 0);
    c.attn_weighted = mat_alloc(seq_len, nh * hd,   0);
    c.gate_buf      = mat_alloc(seq_len, ffnh,       0);
    c.up_buf        = mat_alloc(seq_len, ffnh,       0);

    c.router_probs  = (float *)xcalloc((size_t)seq_len * ne, sizeof(float));
    c.expert_ids    = (int   *)xcalloc((size_t)seq_len * tk, sizeof(int));

    return c;
}

static void cache_free(ForwardCache *c) {
    mat_free(&c->x_norm_attn);
    mat_free(&c->x_norm_ffn);
    mat_free(&c->attn_out);
    mat_free(&c->ffn_out);
    mat_free(&c->Q);
    mat_free(&c->K);
    mat_free(&c->V);
    mat_free(&c->attn_scores);
    mat_free(&c->attn_weighted);
    mat_free(&c->gate_buf);
    mat_free(&c->up_buf);
    free(c->router_probs);  c->router_probs = NULL;
    free(c->expert_ids);    c->expert_ids   = NULL;
}

/* =========================================================================
 * BlockWeights allocation
 * ====================================================================== */

static BlockWeights block_alloc(const ModelConfig *cfg) {
    BlockWeights bw;
    int d = cfg->d_model;

    bw.rms_attn = (float *)xcalloc(d, sizeof(float));
    bw.rms_ffn  = (float *)xcalloc(d, sizeof(float));

    /* Initialise RMSNorm scales to 1.0 (identity at start of training) */
    for (int i = 0; i < d; i++) {
        bw.rms_attn[i] = 1.0f;
        bw.rms_ffn[i]  = 1.0f;
    }

    bw.attn = attn_weights_alloc(cfg);

    bw.use_moe = (cfg->n_experts > 1);

    if (bw.use_moe) {
        bw.moe.n_experts = cfg->n_experts;
        bw.moe.top_k     = cfg->top_k_experts;
        bw.moe.experts   = (FFNWeights *)xmalloc(
                               cfg->n_experts * sizeof(FFNWeights));
        for (int e = 0; e < cfg->n_experts; e++)
            bw.moe.experts[e] = ffn_weights_alloc(d, cfg->ffn_hidden, 1);
        bw.moe.W_router = mat_alloc(d, cfg->n_experts, 1);
        mat_randn(&bw.moe.W_router, sqrtf(2.0f / d));
    } else {
        bw.ffn = ffn_weights_alloc(d, cfg->ffn_hidden, 1);
    }

    return bw;
}

static void block_free(BlockWeights *bw) {
    free(bw->rms_attn);
    free(bw->rms_ffn);
    bw->rms_attn = NULL;
    bw->rms_ffn  = NULL;
    attn_weights_free(&bw->attn);
    if (bw->use_moe) {
        for (int e = 0; e < bw->moe.n_experts; e++)
            ffn_weights_free(&bw->moe.experts[e]);
        free(bw->moe.experts);
        mat_free(&bw->moe.W_router);
    } else {
        ffn_weights_free(&bw->ffn);
    }
}

/* =========================================================================
 * Model lifecycle
 * ====================================================================== */

Model *model_init(const ModelConfig *cfg) {
    Model *m = (Model *)xcalloc(1, sizeof(Model));
    m->cfg = *cfg;

    /* Embedding table */
    m->weights.embed = mat_alloc(cfg->vocab_size, cfg->d_model, 1);
    mat_randn(&m->weights.embed, sqrtf(1.0f / cfg->d_model));

    /* Transformer blocks */
    m->weights.n_layers = cfg->n_layers;
    m->weights.blocks = (BlockWeights *)xmalloc(
                            cfg->n_layers * sizeof(BlockWeights));
    for (int l = 0; l < cfg->n_layers; l++)
        m->weights.blocks[l] = block_alloc(cfg);

    /* Final RMSNorm */
    m->weights.rms_final = (float *)xmalloc(cfg->d_model * sizeof(float));
    for (int i = 0; i < cfg->d_model; i++)
        m->weights.rms_final[i] = 1.0f;

    /* LM head — tied to embedding weights (weight tying) */
    /* We point lm_head.data at embed.data (transposed logically).
     * In practice for weight-tying we reuse the embed matrix directly
     * in the forward pass. We allocate lm_head as a view here. */
    m->weights.lm_head = mat_alloc(cfg->d_model, cfg->vocab_size, 1);
    /* TODO: actual weight tying copies embed.data on each step — or we
     * manually share the pointer. For now it's a separate matrix. */
    mat_randn(&m->weights.lm_head, sqrtf(1.0f / cfg->d_model));

    /* RoPE tables */
    m->rope = rope_buffer_init(cfg->max_seq_len, cfg->head_dim,
                               cfg->rope_theta);

    /* Activation caches (sized to max_seq_len) */
    m->caches = (ForwardCache *)xmalloc(cfg->n_layers * sizeof(ForwardCache));
    for (int l = 0; l < cfg->n_layers; l++)
        m->caches[l] = cache_alloc(cfg, cfg->max_seq_len);

    /* Pre-allocate backward scratch to eliminate per-step malloc */
    m->train_scratch = train_scratch_alloc(cfg, cfg->max_seq_len);

    return m;
}

void model_free(Model *m) {
    if (!m) return;

    mat_free(&m->weights.embed);

    for (int l = 0; l < m->weights.n_layers; l++) {
        block_free(&m->weights.blocks[l]);
        cache_free(&m->caches[l]);
    }
    free(m->weights.blocks);
    free(m->caches);
    free(m->weights.rms_final);
    mat_free(&m->weights.lm_head);
    rope_buffer_free(&m->rope);
    train_scratch_free((TrainScratch *)m->train_scratch);
    free(m);
}

long model_param_count(const Model *m) {
    long total = 0;
    const ModelConfig *cfg = &m->cfg;
    int d  = cfg->d_model;
    int nh = cfg->n_heads, nkv = cfg->n_kv_heads, hd = cfg->head_dim;

    /* Embedding */
    total += (long)cfg->vocab_size * d;

    for (int l = 0; l < cfg->n_layers; l++) {
        /* RMSNorm scales */
        total += 2 * d;
        /* Attention projections */
        total += (long)d * nh  * hd;   /* W_Q */
        total += (long)d * nkv * hd;   /* W_K */
        total += (long)d * nkv * hd;   /* W_V */
        total += (long)nh * hd * d;    /* W_O */
        /* FFN (dense) */
        if (!m->weights.blocks[l].use_moe)
            total += 3L * d * cfg->ffn_hidden;
        else
            total += (long)cfg->n_experts * 3L * d * cfg->ffn_hidden
                   + (long)d * cfg->n_experts;
    }

    /* Final norm + LM head */
    total += d;
    total += (long)d * cfg->vocab_size;

    return total;
}

/* =========================================================================
 * Save / Load  (simple binary format)
 * ====================================================================== */

int model_save(const Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    /* Header: magic number + config */
    uint32_t magic = 0x4C4C4D43;  /* "CLLM" */
    fwrite(&magic,  sizeof(magic), 1, f);
    fwrite(&m->cfg, sizeof(ModelConfig), 1, f);

    /* Helper macro: write a matrix's data buffer */
    #define WRITE_MAT(mat) \
        fwrite((mat).data, sizeof(float), \
               (size_t)(mat).rows * (mat).cols, f)

    WRITE_MAT(m->weights.embed);

    for (int l = 0; l < m->weights.n_layers; l++) {
        BlockWeights *bw = &m->weights.blocks[l];
        fwrite(bw->rms_attn, sizeof(float), m->cfg.d_model, f);
        WRITE_MAT(bw->attn.W_Q);
        WRITE_MAT(bw->attn.W_K);
        WRITE_MAT(bw->attn.W_V);
        WRITE_MAT(bw->attn.W_O);
        fwrite(bw->rms_ffn, sizeof(float), m->cfg.d_model, f);
        if (!bw->use_moe) {
            WRITE_MAT(bw->ffn.W_gate);
            WRITE_MAT(bw->ffn.W_up);
            WRITE_MAT(bw->ffn.W_down);
        } else {
            WRITE_MAT(bw->moe.W_router);
            for (int e = 0; e < bw->moe.n_experts; e++) {
                WRITE_MAT(bw->moe.experts[e].W_gate);
                WRITE_MAT(bw->moe.experts[e].W_up);
                WRITE_MAT(bw->moe.experts[e].W_down);
            }
        }
    }

    fwrite(m->weights.rms_final, sizeof(float), m->cfg.d_model, f);
    WRITE_MAT(m->weights.lm_head);

    #undef WRITE_MAT
    fclose(f);
    return 0;
}

/* Checked fread helper — aborts cleanly on short reads. */
static int fread_checked(void *ptr, size_t sz, size_t n, FILE *f) {
    return fread(ptr, sz, n, f) == n ? 0 : -1;
}

Model *model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    uint32_t magic = 0;
    if (fread_checked(&magic, sizeof(magic), 1, f) < 0 ||
        magic != 0x4C4C4D43) {
        fprintf(stderr, "[model_load] bad magic or read error\n");
        fclose(f); return NULL;
    }

    ModelConfig cfg;
    if (fread_checked(&cfg, sizeof(ModelConfig), 1, f) < 0) {
        fprintf(stderr, "[model_load] failed to read config\n");
        fclose(f); return NULL;
    }

    Model *m = model_init(&cfg);

    /* Macro: read a matrix data buffer; jumps to 'fail' label on error. */
    #define READ_MAT(mat) do { \
        size_t _n = (size_t)(mat).rows * (mat).cols; \
        if (fread_checked((mat).data, sizeof(float), _n, f) < 0) { \
            fprintf(stderr, "[model_load] read error\n"); \
            goto fail; \
        } \
    } while(0)

    #define READ_VEC(ptr, n) do { \
        if (fread_checked((ptr), sizeof(float), (n), f) < 0) { \
            fprintf(stderr, "[model_load] read error\n"); \
            goto fail; \
        } \
    } while(0)

    READ_MAT(m->weights.embed);

    for (int l = 0; l < cfg.n_layers; l++) {
        BlockWeights *bw = &m->weights.blocks[l];
        READ_VEC(bw->rms_attn, cfg.d_model);
        READ_MAT(bw->attn.W_Q);
        READ_MAT(bw->attn.W_K);
        READ_MAT(bw->attn.W_V);
        READ_MAT(bw->attn.W_O);
        READ_VEC(bw->rms_ffn, cfg.d_model);
        if (!bw->use_moe) {
            READ_MAT(bw->ffn.W_gate);
            READ_MAT(bw->ffn.W_up);
            READ_MAT(bw->ffn.W_down);
        } else {
            READ_MAT(bw->moe.W_router);
            for (int e = 0; e < bw->moe.n_experts; e++) {
                READ_MAT(bw->moe.experts[e].W_gate);
                READ_MAT(bw->moe.experts[e].W_up);
                READ_MAT(bw->moe.experts[e].W_down);
            }
        }
    }

    READ_VEC(m->weights.rms_final, cfg.d_model);
    READ_MAT(m->weights.lm_head);

    #undef READ_MAT
    #undef READ_VEC
    fclose(f);
    return m;

fail:
    #undef READ_MAT
    #undef READ_VEC
    model_free(m);
    fclose(f);
    return NULL;
}

/* =========================================================================
 * Block forward
 * ====================================================================== */

/**
 * block_forward — single Transformer block (pre-norm, residual).
 *
 *   x_norm = RMSNorm(x, rms_attn)
 *   x = x + Attention(x_norm)          <- residual 1
 *   x_norm = RMSNorm(x, rms_ffn)
 *   x = x + FFN(x_norm)                <- residual 2
 */
void block_forward(Matrix *x,
                   const BlockWeights *bw,
                   ForwardCache *cache,
                   const RoPEBuffer *rope,
                   const ModelConfig *cfg,
                   int layer) {
    (void)layer;   /* available for per-layer logging */

    /*
     * Cache matrices were allocated at max_seq_len rows.
     * Update .rows to the actual runtime sequence length so all
     * shape assertions and loop bounds are correct.
     * The underlying data buffers are large enough — we are just
     * narrowing the view.
     */
    int seq = x->rows;
    int nh  = cfg->n_heads;
    cache->x_norm_attn.rows   = seq;
    cache->x_norm_ffn.rows    = seq;
    cache->attn_out.rows      = seq;
    cache->ffn_out.rows       = seq;
    cache->Q.rows             = seq;
    cache->K.rows             = seq;
    cache->V.rows             = seq;
    cache->attn_scores.rows   = nh * seq;   /* [nh*seq × seq] flat layout */
    cache->attn_scores.cols   = seq;
    cache->attn_weighted.rows = seq;

    /* --- Pre-attention RMSNorm ---------------------------------------- */
    rms_norm(x, bw->rms_attn, cfg->rms_norm_eps, &cache->x_norm_attn);

    /* --- Attention sublayer ------------------------------------------- */
    attention_forward(&cache->x_norm_attn,
                      &bw->attn, rope, cfg,
                      &cache->Q, &cache->K, &cache->V,
                      &cache->attn_scores, &cache->attn_weighted,
                      &cache->attn_out);

    /* --- Residual 1 --------------------------------------------------- */
    mat_add_inplace(x, &cache->attn_out);

    /* --- Pre-FFN RMSNorm ---------------------------------------------- */
    rms_norm(x, bw->rms_ffn, cfg->rms_norm_eps, &cache->x_norm_ffn);

    /* --- FFN sublayer (dense or MoE) ---------------------------------- */
    if (!bw->use_moe) {
        ffn_forward(&cache->x_norm_ffn, &bw->ffn, cfg,
                    &cache->gate_buf,
                    &cache->up_buf,
                    &cache->ffn_out);
    } else {
        moe_forward(&cache->x_norm_ffn, &bw->moe, cfg, cache,
                    &cache->ffn_out);
    }

    /* --- Residual 2 --------------------------------------------------- */
    mat_add_inplace(x, &cache->ffn_out);
}

/* =========================================================================
 * Full model forward pass
 * ====================================================================== */

void model_forward(Model *m, const int *tokens, int seq_len, Matrix *logits) {
    const ModelConfig *cfg = &m->cfg;
    assert(seq_len <= cfg->max_seq_len);
    assert(logits->rows == seq_len && logits->cols == cfg->vocab_size);

    /* --- Embed tokens ------------------------------------------------- */
    Matrix x = mat_alloc(seq_len, cfg->d_model, 0);
    embed_lookup(&m->weights.embed, tokens, seq_len, &x);

    /* --- Run through all blocks --------------------------------------- */
    for (int l = 0; l < cfg->n_layers; l++) {
        block_forward(&x,
                      &m->weights.blocks[l],
                      &m->caches[l],
                      &m->rope,
                      cfg,
                      l);
    }

    /* --- Final RMSNorm ------------------------------------------------ */
    Matrix x_norm = mat_alloc(seq_len, cfg->d_model, 0);
    rms_norm(&x, m->weights.rms_final, cfg->rms_norm_eps, &x_norm);

    /* --- LM head: logits = x_norm · W_lm_head ----------------------- */
    mat_mul(&x_norm, &m->weights.lm_head, logits);

    mat_free(&x);
    mat_free(&x_norm);
}

/* =========================================================================
 * Loss + backward
 * ====================================================================== */

/**
 * cross_entropy_and_grad — compute CE loss and gradient w.r.t. logits.
 *
 * For each position t with target token y_t:
 *   loss_t = -log( softmax(logits_t)[y_t] )
 *
 * Gradient w.r.t. logits_t[j]:
 *   d_loss / d_logits_t[j] = softmax(logits_t)[j] - 1{j == y_t}
 *
 * With label smoothing (ls > 0):
 *   target distribution: (1-ls) * one_hot + ls / vocab_size
 */
static float cross_entropy_and_grad(Matrix *logits,
                                    const int *targets,
                                    int seq_len,
                                    float label_smoothing,
                                    Matrix *d_logits) {
    int V = logits->cols;
    float loss = 0.0f;
    float ls   = label_smoothing;

    for (int t = 0; t < seq_len; t++) {
        float *lg  = logits->data  + t * V;
        float *dlg = d_logits->data + t * V;

        /* Numerically stable softmax */
        float max_v = lg[0];
        for (int j = 1; j < V; j++) if (lg[j] > max_v) max_v = lg[j];

        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            dlg[j] = expf(lg[j] - max_v);
            sum += dlg[j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < V; j++) dlg[j] *= inv_sum;  /* now probs */

        /* CE loss for this position */
        int y = targets[t];
        loss += -logf(dlg[y] + 1e-10f);

        /* Gradient: prob - target */
        if (ls > 0.0f) {
            float smooth_target = ls / (float)V;
            for (int j = 0; j < V; j++)
                dlg[j] -= (j == y) ? (1.0f - ls + smooth_target) : smooth_target;
        } else {
            dlg[y] -= 1.0f;
        }

        /* Scale gradient by 1/seq_len for mean loss */
        for (int j = 0; j < V; j++) dlg[j] /= (float)seq_len;
    }

    return loss / (float)seq_len;
}

float model_backward(Model *m,
                     const int *tokens,
                     const int *targets,
                     int seq_len) {
    const ModelConfig *cfg = &m->cfg;
    int d = cfg->d_model;
    int V = cfg->vocab_size;
    int L = cfg->n_layers;

    /* Use pre-allocated scratch — ZERO malloc calls per backward step. */
    TrainScratch *s = (TrainScratch *)m->train_scratch;

    /* ------------------------------------------------------------------ */
    /* Forward pass: save residual stream at every layer boundary          */
    /* ------------------------------------------------------------------ */
    /* Create seq_len-sized views into the (max_seq_len-sized) scratch.   */
    /* The underlying data buffer is large enough; we just adjust .rows.  */
    for (int i = 0; i <= L; i++) s->x_states[i].rows = seq_len;
    s->x_norm_final.rows  = seq_len;
    s->logits.rows        = seq_len;
    s->d_logits.rows      = seq_len;
    s->d_x_norm.rows      = seq_len;
    s->d_x_top.rows       = seq_len;
    s->d_x.rows           = seq_len;
    s->d_x_new.rows       = seq_len;
    s->d_x_norm_attn.rows = seq_len;
    s->d_x_norm_ffn.rows  = seq_len;

    embed_lookup(&m->weights.embed, tokens, seq_len, &s->x_states[0]);

    for (int l = 0; l < L; l++) {
        /* Resize ForwardCache matrices to actual seq_len (they were
         * allocated at max_seq_len; data buffer is large enough). */
        ForwardCache *fc = &m->caches[l];
        fc->x_norm_attn.rows   = seq_len;
        fc->x_norm_ffn.rows    = seq_len;
        fc->attn_out.rows      = seq_len;
        fc->ffn_out.rows       = seq_len;
        fc->Q.rows             = seq_len;
        fc->K.rows             = seq_len;
        fc->V.rows             = seq_len;
        fc->attn_weighted.rows = seq_len;
        /* attn_scores is [n_heads*seq × seq] — resize both dimensions */
        fc->attn_scores.rows   = cfg->n_heads * seq_len;
        fc->attn_scores.cols   = seq_len;
        fc->gate_buf.rows      = seq_len;
        fc->up_buf.rows        = seq_len;

        mat_copy(&s->x_states[l + 1], &s->x_states[l]);
        block_forward(&s->x_states[l + 1],
                      &m->weights.blocks[l],
                      &m->caches[l],
                      &m->rope, cfg, l);
    }

    /* Final RMSNorm */
    rms_norm(&s->x_states[L], m->weights.rms_final,
             cfg->rms_norm_eps, &s->x_norm_final);

    /* LM head logits */
    mat_mul(&s->x_norm_final, &m->weights.lm_head, &s->logits);

    /* ------------------------------------------------------------------ */
    /* Cross-entropy loss + d_logits                                        */
    /* ------------------------------------------------------------------ */
    float loss = cross_entropy_and_grad(&s->logits, targets, seq_len,
                                        0.1f, &s->d_logits);

    /* ------------------------------------------------------------------ */
    /* Backprop: LM head                                                    */
    /* dW_lm  += x_norm_finalᵀ · d_logits   — mat_mul, not scalar loops  */
    /* d_x_norm = d_logits · W_lm_headᵀ    — mat_mul_T                   */
    /* ------------------------------------------------------------------ */

    /* Wrap grad buffer as a Matrix view for mat_mul accumulation */
    Matrix lm_grad_view = {
        .data = m->weights.lm_head.grad,
        .grad = NULL,
        .rows = d, .cols = V
    };
    /* Temp: compute xᵀ·d and add into lm_grad_view */
    /* mat_mul(xᵀ, d_logits) where x is [seq×d] → xᵀ is [d×seq] */
    /* We implement this as: for each d-col of x, dot with each V-col of d_logits */
    /* Using mat_mul_T: lm_grad += (x_norm_final)ᵀ · d_logits */
    /* (x_norm_final)[seq×d]ᵀ × d_logits[seq×V] → [d×V] */
    /* mat_mul_T(A,B) = A×Bᵀ which is wrong direction; do it manually with SIMD */
    for (int i = 0; i < seq_len; i++) {
        const float *xrow = s->x_norm_final.data + i * d;
        const float *drow = s->d_logits.data     + i * V;
        for (int j = 0; j < d; j++) {
            float xj = xrow[j];
            if (xj == 0.0f) continue;
            float *grow = m->weights.lm_head.grad + j * V;
            /* axpy: grow += xj * drow */
            for (int k = 0; k < V; k++) grow[k] += xj * drow[k];
        }
    }

    /* d_x_norm = d_logits · W_lm_headᵀ  →  mat_mul_T(d_logits, W_lm_head) */
    mat_mul_T(&s->d_logits, &m->weights.lm_head, &s->d_x_norm);

    /* ------------------------------------------------------------------ */
    /* Backprop: final RMSNorm                                              */
    /* ------------------------------------------------------------------ */
    memset(s->d_rms_final, 0, d * sizeof(float));
    mat_zero(&s->d_x_top);
    rms_norm_backward(&s->x_states[L], m->weights.rms_final,
                      cfg->rms_norm_eps,
                      &s->d_x_norm, &s->d_x_top, s->d_rms_final);
    /* TODO: accumulate d_rms_final into a proper rms_final gradient buffer */

    /* ------------------------------------------------------------------ */
    /* Backprop: blocks in reverse order — all scratch reused each layer   */
    /* ------------------------------------------------------------------ */
    mat_copy(&s->d_x, &s->d_x_top);

    for (int l = L - 1; l >= 0; l--) {
        BlockWeights *bw    = &m->weights.blocks[l];
        ForwardCache *cache = &m->caches[l];
        Matrix       *x_in  = &s->x_states[l];

        mat_zero(&s->d_x_new);

        /* Backprop FFN RMSNorm */
        memset(s->d_rms_ffn, 0, d * sizeof(float));
        mat_zero(&s->d_x_norm_ffn);
        rms_norm_backward(x_in, bw->rms_ffn, cfg->rms_norm_eps,
                          &s->d_x, &s->d_x_norm_ffn, s->d_rms_ffn);

        /* Backprop attention RMSNorm */
        memset(s->d_rms_attn, 0, d * sizeof(float));
        mat_zero(&s->d_x_norm_attn);
        rms_norm_backward(x_in, bw->rms_attn, cfg->rms_norm_eps,
                          &s->d_x_norm_ffn, &s->d_x_norm_attn, s->d_rms_attn);

        /* Backprop attention weights */
        attention_backward(x_in, &bw->attn, cfg,
                           &cache->Q, &cache->K, &cache->V,
                           &cache->attn_scores, &cache->attn_weighted,
                           &s->d_x_norm_attn, &s->d_x_new);

        /* Residual: d_x_new += d_x (gradient passes through both paths) */
        mat_add_inplace(&s->d_x_new, &s->d_x);

        /* Swap: d_x = d_x_new for next layer */
        float *tmp = s->d_x.data;
        s->d_x.data     = s->d_x_new.data;
        s->d_x_new.data = tmp;
    }

    /* Backprop into embedding table */
    embed_backward(&m->weights.embed, tokens, seq_len, &s->d_x);

    (void)lm_grad_view;
    return loss;
}

/* =========================================================================
 * Gradient zeroing
 * ====================================================================== */

void model_zero_grads(Model *m) {
    mat_zero_grad(&m->weights.embed);

    for (int l = 0; l < m->weights.n_layers; l++) {
        BlockWeights *bw = &m->weights.blocks[l];
        mat_zero_grad(&bw->attn.W_Q);
        mat_zero_grad(&bw->attn.W_K);
        mat_zero_grad(&bw->attn.W_V);
        mat_zero_grad(&bw->attn.W_O);
        if (!bw->use_moe) {
            mat_zero_grad(&bw->ffn.W_gate);
            mat_zero_grad(&bw->ffn.W_up);
            mat_zero_grad(&bw->ffn.W_down);
        } else {
            mat_zero_grad(&bw->moe.W_router);
            for (int e = 0; e < bw->moe.n_experts; e++) {
                mat_zero_grad(&bw->moe.experts[e].W_gate);
                mat_zero_grad(&bw->moe.experts[e].W_up);
                mat_zero_grad(&bw->moe.experts[e].W_down);
            }
        }
    }

    mat_zero_grad(&m->weights.lm_head);
}
