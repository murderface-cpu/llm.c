/**
 * attention.c
 *
 * Implementation of:
 *   - RoPE table init and application
 *   - Grouped Query Attention (GQA) forward pass
 *   - Sliding window causal mask
 *   - Attention backward pass
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>

#include "../include/attention.h"

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

AttentionWeights attn_weights_alloc(const ModelConfig *cfg) {
    AttentionWeights w;
    int d  = cfg->d_model;
    int qd = cfg->n_heads    * cfg->head_dim;  /* total Q dimension */
    int kd = cfg->n_kv_heads * cfg->head_dim;  /* total K/V dimension */

    w.W_Q = mat_alloc(d,  qd, /*with_grad=*/1);
    w.W_K = mat_alloc(d,  kd, /*with_grad=*/1);
    w.W_V = mat_alloc(d,  kd, /*with_grad=*/1);
    w.W_O = mat_alloc(qd, d,  /*with_grad=*/1);

    /* He initialisation: std = sqrt(2 / fan_in) */
    mat_randn(&w.W_Q, sqrtf(2.0f / d));
    mat_randn(&w.W_K, sqrtf(2.0f / d));
    mat_randn(&w.W_V, sqrtf(2.0f / d));
    mat_randn(&w.W_O, sqrtf(2.0f / qd));

    return w;
}

void attn_weights_free(AttentionWeights *w) {
    mat_free(&w->W_Q);
    mat_free(&w->W_K);
    mat_free(&w->W_V);
    mat_free(&w->W_O);
}

RoPEBuffer rope_buffer_init(int max_seq_len, int head_dim, float theta) {
    assert(head_dim % 2 == 0);
    int half = head_dim / 2;

    RoPEBuffer rb;
    rb.max_seq_len = max_seq_len;
    rb.head_dim    = head_dim;
    rb.cos_table   = (float *)malloc((size_t)max_seq_len * half * sizeof(float));
    rb.sin_table   = (float *)malloc((size_t)max_seq_len * half * sizeof(float));

    if (!rb.cos_table || !rb.sin_table) {
        fprintf(stderr, "[attention] rope_buffer_init: OOM\n");
        abort();
    }

    /*
     * angle(pos, i) = pos / theta^(2i / head_dim)
     * cos_table[pos * half + i] = cos(angle)
     * sin_table[pos * half + i] = sin(angle)
     */
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float angle = (float)pos
                        / powf(theta, 2.0f * (float)i / (float)head_dim);
            rb.cos_table[pos * half + i] = cosf(angle);
            rb.sin_table[pos * half + i] = sinf(angle);
        }
    }
    return rb;
}

void rope_buffer_free(RoPEBuffer *rb) {
    free(rb->cos_table);
    free(rb->sin_table);
    rb->cos_table = NULL;
    rb->sin_table = NULL;
}

AttentionCache attn_cache_alloc(int max_seq_len, int kv_dim) {
    AttentionCache c;
    c.K_cache = mat_alloc(max_seq_len, kv_dim, /*with_grad=*/0);
    c.V_cache = mat_alloc(max_seq_len, kv_dim, /*with_grad=*/0);
    c.seq_pos = 0;
    c.head    = 0;
    return c;
}

void attn_cache_free(AttentionCache *c) {
    mat_free(&c->K_cache);
    mat_free(&c->V_cache);
    c->seq_pos = 0;
}

void attn_cache_reset(AttentionCache *c) {
    mat_zero(&c->K_cache);
    mat_zero(&c->V_cache);
    c->seq_pos = 0;
}

/* =========================================================================
 * RoPE application
 * ====================================================================== */

/**
 * rope_apply — rotate each head's Q or K vectors in-place.
 *
 * For head h and position pos, the vector slice:
 *   v = qk[pos][h * head_dim : (h+1) * head_dim]
 * is rotated pair-wise:
 *   (v[2i], v[2i+1]) → (v[2i]*cos - v[2i+1]*sin,
 *                        v[2i]*sin + v[2i+1]*cos)
 */
void rope_apply(Matrix *qk, const RoPEBuffer *rb,
                int n_heads, int head_dim, int offset) {
    int half     = head_dim / 2;
    int seq_len  = qk->rows;
    int total_d  = qk->cols;   /* n_heads * head_dim */

    assert(total_d == n_heads * head_dim);

    for (int pos = 0; pos < seq_len; pos++) {
        int abs_pos = pos + offset;  /* actual sequence position */
        float *row  = qk->data + pos * total_d;

        for (int h = 0; h < n_heads; h++) {
            float *v = row + h * head_dim;
            const float *cos_row = rb->cos_table + abs_pos * half;
            const float *sin_row = rb->sin_table + abs_pos * half;

            for (int i = 0; i < half; i++) {
                float v0 = v[2 * i];
                float v1 = v[2 * i + 1];
                v[2 * i]     = v0 * cos_row[i] - v1 * sin_row[i];
                v[2 * i + 1] = v0 * sin_row[i] + v1 * cos_row[i];
            }
        }
    }
}

/* =========================================================================
 * Causal + sliding window mask
 * ====================================================================== */

/**
 * apply_causal_mask — mask future tokens and tokens outside the window.
 *
 * scores is [seq_len × seq_len] for one attention head.
 * We set score[i][j] = -inf when:
 *   j > i                    (future: causal mask)
 *   i - j >= window_size     (too far in the past: sliding window)
 */
void apply_causal_mask(Matrix *scores, int window_size) {
    int seq_len = scores->rows;
    assert(scores->cols == seq_len);

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            int future  = (j > i);
            int too_far = (window_size > 0) && (i - j >= window_size);
            if (future || too_far)
                scores->data[i * seq_len + j] = -FLT_MAX;
        }
    }
}

/* =========================================================================
 * Attention forward pass
 * ====================================================================== */

/**
 * attention_forward — full GQA forward pass.
 *
 * Steps (matching the transformer paper + GQA paper):
 *   1. Project: Q = x·W_Q,  K = x·W_K,  V = x·W_V
 *   2. Apply RoPE to Q and K
 *   3. For each query head q_h:
 *        kv_h = q_h / (n_heads / n_kv_heads)   <- which KV group to use
 *        scores[q_h] = Q[q_h] · K[kv_h]ᵀ / sqrt(head_dim)
 *        apply causal mask
 *        softmax(scores[q_h])
 *        out[q_h] = scores[q_h] · V[kv_h]
 *   4. Concatenate all head outputs
 *   5. out = concat · W_O
 */
void attention_forward(const Matrix *x,
                       const AttentionWeights *w,
                       const RoPEBuffer *rb,
                       const ModelConfig *cfg,
                       Matrix *Q, Matrix *K, Matrix *V,
                       Matrix *scores, Matrix *attn,
                       Matrix *out) {
    int seq  = x->rows;
    int nh   = cfg->n_heads;
    int nkv  = cfg->n_kv_heads;
    int hd   = cfg->head_dim;
    int win  = cfg->window_size;
    int gqa_ratio = nh / nkv;   /* query heads per KV head */
    float scale = 1.0f / sqrtf((float)hd);

    /* ------------------------------------------------------------------ */
    /* Step 1: Project to Q, K, V                                          */
    mat_mul(x, &w->W_Q, Q);   /* [seq × (nh  * hd)] */
    mat_mul(x, &w->W_K, K);   /* [seq × (nkv * hd)] */
    mat_mul(x, &w->W_V, V);   /* [seq × (nkv * hd)] */

    /* ------------------------------------------------------------------ */
    /* Step 2: Apply RoPE to Q and K                                       */
    rope_apply(Q, rb, nh,  hd, /*offset=*/0);
    rope_apply(K, rb, nkv, hd, /*offset=*/0);

    /* ------------------------------------------------------------------ */
    /* Steps 3–4: Per-head attention with GQA                             */

    /*
     * We process each query head independently.
     * Slice views into Q, K, V, scores, attn — no extra allocation.
     *
     * Q row layout: [h0_d0..h0_dhd | h1_d0..h1_dhd | ... ]
     * We index: Q[pos][head * hd + dim_within_head]
     */

    mat_zero(attn);  /* zero the output concat buffer */

    /* Temporary single-head score matrix [seq × seq] */
    /* We reuse scores->data slice (scores is [nh*seq × seq] flat) */

    for (int qh = 0; qh < nh; qh++) {
        int kv_h = qh / gqa_ratio;  /* which KV head this query head uses */

        /* Pointers to head slice in the flat score buffer */
        float *head_scores = scores->data + qh * seq * seq;

        /* Compute dot products: scores[i][j] = Q[qh,i] · K[kv_h,j] * scale */
        for (int i = 0; i < seq; i++) {
            const float *qi = Q->data + i * nh * hd + qh * hd;
            for (int j = 0; j < seq; j++) {
                const float *kj = K->data + j * nkv * hd + kv_h * hd;
                float dot = 0.0f;
                for (int k = 0; k < hd; k++)
                    dot += qi[k] * kj[k];
                head_scores[i * seq + j] = dot * scale;
            }
        }

        /* Apply causal + sliding window mask */
        /* Wrap head_scores in a stack Matrix for apply_causal_mask */
        Matrix head_score_mat = {
            .data = head_scores,
            .grad = NULL,
            .rows = seq,
            .cols = seq,
        };
        apply_causal_mask(&head_score_mat, win);

        /* Softmax each row */
        softmax_rows(&head_score_mat);

        /* Weighted sum of V: attn[i][qh*hd:(qh+1)*hd] = scores[i] · V[kv_h] */
        for (int i = 0; i < seq; i++) {
            float *out_i = attn->data + i * nh * hd + qh * hd;
            for (int k = 0; k < hd; k++) out_i[k] = 0.0f;

            for (int j = 0; j < seq; j++) {
                float s = head_scores[i * seq + j];
                const float *vj = V->data + j * nkv * hd + kv_h * hd;
                for (int k = 0; k < hd; k++)
                    out_i[k] += s * vj[k];
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Step 5: Output projection: out = attn · W_O                        */
    mat_mul(attn, &w->W_O, out);
}

/* =========================================================================
 * Attention backward pass
 * ====================================================================== */

/**
 * attention_backward
 *
 * Backprop through the attention layer. This closely mirrors forward_pass
 * in reverse. We need:
 *   - Saved Q, K, V projections
 *   - Saved softmax scores
 *   - Saved post-softmax attn (weighted V concat)
 *   - The incoming gradient d_out [seq × d_model]
 *
 * We accumulate (not overwrite) into weight gradients so multiple backward
 * calls within a batch can be summed before the optimiser step.
 */
void attention_backward(const Matrix *x,
                        const AttentionWeights *w,
                        const ModelConfig *cfg,
                        const Matrix *Q, const Matrix *K, const Matrix *V,
                        const Matrix *scores, const Matrix *attn,
                        const Matrix *d_out,
                        Matrix *d_x) {
    int seq      = x->rows;
    int d        = cfg->d_model;
    int nh       = cfg->n_heads;
    int nkv      = cfg->n_kv_heads;
    int hd       = cfg->head_dim;
    int gqa_ratio= nh / nkv;
    float scale  = 1.0f / sqrtf((float)hd);

    /* --- Gradient through W_O: d_attn = d_out · W_Oᵀ ------------------- */
    /* attn: [seq × nh*hd],  W_O: [nh*hd × d],  d_out: [seq × d]           */
    Matrix d_attn = mat_alloc(seq, nh * hd, /*with_grad=*/0);
    mat_zero(&d_attn);
    /* d_attn = d_out · W_Oᵀ  (W_O is [nh*hd × d], so transpose is [d × nh*hd]) */
    /* Use mat_mul_T with arguments swapped: d_attn = d_out · W_O^T */
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < nh * hd; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++)
                s += d_out->data[i * d + k] * w->W_O.data[j * d + k];
            d_attn.data[i * nh * hd + j] = s;
        }
    }

    /* Accumulate gradient for W_O: dW_O += attnᵀ · d_out */
    for (int j = 0; j < nh * hd; j++) {
        for (int k = 0; k < d; k++) {
            float s = 0.0f;
            for (int i = 0; i < seq; i++)
                s += attn->data[i * nh * hd + j] * d_out->data[i * d + k];
            w->W_O.grad[j * d + k] += s;
        }
    }

    /* --- Per-head backprop --------------------------------------------- */
    Matrix d_Q = mat_alloc(seq, nh  * hd, /*with_grad=*/0);
    Matrix d_K = mat_alloc(seq, nkv * hd, /*with_grad=*/0);
    Matrix d_V = mat_alloc(seq, nkv * hd, /*with_grad=*/0);
    mat_zero(&d_Q);
    mat_zero(&d_K);
    mat_zero(&d_V);

    for (int qh = 0; qh < nh; qh++) {
        int kv_h = qh / gqa_ratio;
        const float *head_scores = scores->data + qh * seq * seq;

        /* d_V[kv_h] += softmax_scoresᵀ · d_attn[qh] */
        for (int j = 0; j < seq; j++) {
            float *dv = d_V.data + j * nkv * hd + kv_h * hd;
            for (int i = 0; i < seq; i++) {
                float s = head_scores[i * seq + j];
                const float *da = d_attn.data + i * nh * hd + qh * hd;
                for (int k = 0; k < hd; k++)
                    dv[k] += s * da[k];
            }
        }

        /* d_scores[i][j] = d_attn[qh,i] · V[kv_h,j] */
        float *d_scores_h = (float *)malloc((size_t)seq * seq * sizeof(float));
        for (int i = 0; i < seq; i++) {
            const float *da  = d_attn.data + i * nh * hd + qh * hd;
            for (int j = 0; j < seq; j++) {
                const float *vj = V->data + j * nkv * hd + kv_h * hd;
                float dot = 0.0f;
                for (int k = 0; k < hd; k++) dot += da[k] * vj[k];
                d_scores_h[i * seq + j] = dot;
            }
        }

        /* Backprop through softmax: d_raw[i][j] = p[i][j] * (d[i][j] - sum_k p[i][k]*d[i][k]) */
        for (int i = 0; i < seq; i++) {
            const float *p = head_scores + i * seq;
            float *ds      = d_scores_h  + i * seq;
            float dot = 0.0f;
            for (int j = 0; j < seq; j++) dot += p[j] * ds[j];
            for (int j = 0; j < seq; j++) ds[j] = p[j] * (ds[j] - dot);
        }

        /* Scale by 1/sqrt(hd) */
        for (int i = 0; i < seq * seq; i++) d_scores_h[i] *= scale;

        /* d_Q[qh] += d_scores · K[kv_h] */
        for (int i = 0; i < seq; i++) {
            float *dq = d_Q.data + i * nh * hd + qh * hd;
            for (int j = 0; j < seq; j++) {
                const float *kj = K->data + j * nkv * hd + kv_h * hd;
                float ds = d_scores_h[i * seq + j];
                for (int k = 0; k < hd; k++) dq[k] += ds * kj[k];
            }
        }

        /* d_K[kv_h] += d_scoresᵀ · Q[qh] */
        for (int j = 0; j < seq; j++) {
            float *dk = d_K.data + j * nkv * hd + kv_h * hd;
            for (int i = 0; i < seq; i++) {
                const float *qi = Q->data + i * nh * hd + qh * hd;
                float ds = d_scores_h[i * seq + j];
                for (int k = 0; k < hd; k++) dk[k] += ds * qi[k];
            }
        }

        free(d_scores_h);
    }

    /* --- Gradient through projection matrices and into x ---------------- */
    /* dW_Q += xᵀ · d_Q,  dW_K += xᵀ · d_K,  dW_V += xᵀ · d_V          */
    for (int j = 0; j < d; j++) {
        for (int k = 0; k < nh * hd; k++) {
            float s = 0.0f;
            for (int i = 0; i < seq; i++)
                s += x->data[i * d + j] * d_Q.data[i * nh * hd + k];
            w->W_Q.grad[j * nh * hd + k] += s;
        }
        for (int k = 0; k < nkv * hd; k++) {
            float sq = 0.0f, sv = 0.0f;
            for (int i = 0; i < seq; i++) {
                sq += x->data[i * d + j] * d_K.data[i * nkv * hd + k];
                sv += x->data[i * d + j] * d_V.data[i * nkv * hd + k];
            }
            w->W_K.grad[j * nkv * hd + k] += sq;
            w->W_V.grad[j * nkv * hd + k] += sv;
        }
    }

    /* d_x += d_Q · W_Qᵀ + d_K · W_Kᵀ + d_V · W_Vᵀ */
    mat_zero(d_x);
    for (int i = 0; i < seq; i++) {
        float *dx = d_x->data + i * d;
        for (int k = 0; k < nh * hd; k++) {
            float dq = d_Q.data[i * nh * hd + k];
            for (int j = 0; j < d; j++)
                dx[j] += dq * w->W_Q.data[j * nh * hd + k];
        }
        for (int k = 0; k < nkv * hd; k++) {
            float dk_v = d_K.data[i * nkv * hd + k];
            float dv_v = d_V.data[i * nkv * hd + k];
            for (int j = 0; j < d; j++) {
                dx[j] += dk_v * w->W_K.data[j * nkv * hd + k];
                dx[j] += dv_v * w->W_V.data[j * nkv * hd + k];
            }
        }
    }

    mat_free(&d_attn);
    mat_free(&d_Q);
    mat_free(&d_K);
    mat_free(&d_V);
}
