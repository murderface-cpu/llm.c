/**
 * attention.c
 *
 * Attention mechanisms - fully sgemm-accelerated forward and backward.
 *
 * Every matrix multiply is a cblas_sgemm call when USE_OPENBLAS is defined.
 * This eliminates the O(seq² × d) scalar triple-loops that were the dominant
 * training bottleneck.
 *
 * sgemm reference:
 *   C = alpha * op(A) * op(B) + beta * C
 *   CblasNoTrans: use A as-is
 *   CblasTrans:   use Aᵀ
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>

#include "../include/attention.h"
#include "../include/simd.h"

#ifdef USE_OPENBLAS
#  include <cblas.h>
/* Thin wrappers so the rest of the code stays readable */

/* out = A × B  (accumulate: beta=1) */
static inline void sgemm_nn_acc(int M, int N, int K,
                                  const float *A, const float *B,
                                  float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.f, A, K, B, N, 1.f, C, N);
}
/* out = A × B  (overwrite: beta=0) */
static inline void sgemm_nn(int M, int N, int K,
                              const float *A, const float *B,
                              float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.f, A, K, B, N, 0.f, C, N);
}
/* out = Aᵀ × B  (accumulate) */
static inline void sgemm_tn_acc(int M, int N, int K,
                                  const float *A, int lda,
                                  const float *B,
                                  float *C) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.f, A, M, B, N, 1.f, C, N);
}
/* out = A × Bᵀ  (overwrite) */
static inline void sgemm_nt(int M, int N, int K,
                              const float *A,
                              const float *B, int ldb,
                              float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.f, A, K, B, K, 0.f, C, N);
}
/* out = A × Bᵀ  (accumulate) */
static inline void sgemm_nt_acc(int M, int N, int K,
                                  const float *A,
                                  const float *B, int ldb,
                                  float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.f, A, K, B, K, 1.f, C, N);
}
/* out += scale * A × B */
static inline void sgemm_nn_scale(int M, int N, int K, float scale,
                                    const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, scale, A, K, B, N, 1.f, C, N);
}
/* out += scale * A × Bᵀ */
static inline void sgemm_nt_scale(int M, int N, int K, float scale,
                                    const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, scale, A, K, B, K, 1.f, C, N);
}
#else
/* Fallback scalar implementations */
static void sgemm_nn(int M, int N, int K, const float *A, const float *B, float *C) {
    memset(C, 0, M*N*sizeof(float));
    for(int i=0;i<M;i++) for(int k=0;k<K;k++) {
        float a=A[i*K+k]; if(!a) continue;
        for(int j=0;j<N;j++) C[i*N+j]+=a*B[k*N+j];
    }
}
static void sgemm_nn_acc(int M, int N, int K, const float *A, const float *B, float *C) {
    for(int i=0;i<M;i++) for(int k=0;k<K;k++) {
        float a=A[i*K+k]; if(!a) continue;
        for(int j=0;j<N;j++) C[i*N+j]+=a*B[k*N+j];
    }
}
static void sgemm_nt(int M, int N, int K, const float *A, const float *B, int ldb, float *C) {
    memset(C,0,M*N*sizeof(float));
    for(int i=0;i<M;i++) for(int j=0;j<N;j++) {
        float s=0; for(int k=0;k<K;k++) s+=A[i*K+k]*B[j*K+k];
        C[i*N+j]=s;
    }
}
static void sgemm_nt_acc(int M, int N, int K, const float *A, const float *B, int ldb, float *C) {
    for(int i=0;i<M;i++) for(int j=0;j<N;j++) {
        float s=0; for(int k=0;k<K;k++) s+=A[i*K+k]*B[j*K+k];
        C[i*N+j]+=s;
    }
}
static void sgemm_tn_acc(int M, int N, int K, const float *A, int lda, const float *B, float *C) {
    for(int k=0;k<K;k++) for(int i=0;i<M;i++) {
        float a=A[k*M+i]; if(!a) continue;
        for(int j=0;j<N;j++) C[i*N+j]+=a*B[k*N+j];
    }
}
static void sgemm_nn_scale(int M,int N,int K,float s,const float*A,const float*B,float*C) {
    for(int i=0;i<M;i++) for(int k=0;k<K;k++) {
        float a=s*A[i*K+k]; if(!a) continue;
        for(int j=0;j<N;j++) C[i*N+j]+=a*B[k*N+j];
    }
}
static void sgemm_nt_scale(int M,int N,int K,float sc,const float*A,const float*B,float*C) {
    for(int i=0;i<M;i++) for(int j=0;j<N;j++) {
        float s=0; for(int k=0;k<K;k++) s+=A[i*K+k]*B[j*K+k];
        C[i*N+j]+=sc*s;
    }
}
#endif

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

AttentionWeights attn_weights_alloc(const ModelConfig *cfg) {
    AttentionWeights w;
    int d  = cfg->d_model;
    int qd = cfg->n_heads    * cfg->head_dim;
    int kd = cfg->n_kv_heads * cfg->head_dim;

    w.W_Q = mat_alloc(d, qd, 1);  mat_randn(&w.W_Q, sqrtf(2.f/d));
    w.W_K = mat_alloc(d, kd, 1);  mat_randn(&w.W_K, sqrtf(2.f/d));
    w.W_V = mat_alloc(d, kd, 1);  mat_randn(&w.W_V, sqrtf(2.f/d));
    w.W_O = mat_alloc(qd, d, 1);  mat_randn(&w.W_O, sqrtf(2.f/qd));
    return w;
}

void attn_weights_free(AttentionWeights *w) {
    mat_free(&w->W_Q); mat_free(&w->W_K);
    mat_free(&w->W_V); mat_free(&w->W_O);
}

RoPEBuffer rope_buffer_init(int max_seq_len, int head_dim, float theta) {
    assert(head_dim % 2 == 0);
    int half = head_dim / 2;
    RoPEBuffer rb;
    rb.max_seq_len = max_seq_len;
    rb.head_dim    = head_dim;
    rb.cos_table   = malloc((size_t)max_seq_len * half * sizeof(float));
    rb.sin_table   = malloc((size_t)max_seq_len * half * sizeof(float));
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float angle = (float)pos / powf(theta, 2.f*(float)i/(float)head_dim);
            rb.cos_table[pos*half+i] = cosf(angle);
            rb.sin_table[pos*half+i] = sinf(angle);
        }
    }
    return rb;
}
void rope_buffer_free(RoPEBuffer *rb) {
    free(rb->cos_table); free(rb->sin_table);
    rb->cos_table = rb->sin_table = NULL;
}

AttentionCache attn_cache_alloc(int max_seq_len, int kv_dim) {
    AttentionCache c;
    c.K_cache = mat_alloc(max_seq_len, kv_dim, 0);
    c.V_cache = mat_alloc(max_seq_len, kv_dim, 0);
    c.seq_pos = 0; c.head = 0;
    return c;
}
void attn_cache_free(AttentionCache *c) {
    mat_free(&c->K_cache); mat_free(&c->V_cache); c->seq_pos = 0;
}
void attn_cache_reset(AttentionCache *c) {
    mat_zero(&c->K_cache); mat_zero(&c->V_cache); c->seq_pos = 0;
}

/* =========================================================================
 * RoPE
 * ====================================================================== */

void rope_apply(Matrix *qk, const RoPEBuffer *rb,
                int n_heads, int head_dim, int offset) {
    int half    = head_dim / 2;
    int seq_len = qk->rows;
    int total_d = qk->cols;

    for (int pos = 0; pos < seq_len; pos++) {
        int abs_pos = pos + offset;
        float *row  = qk->data + pos * total_d;
        for (int h = 0; h < n_heads; h++) {
            float *v = row + h * head_dim;
            const float *cr = rb->cos_table + abs_pos * half;
            const float *sr = rb->sin_table + abs_pos * half;
            for (int i = 0; i < half; i++) {
                float v0 = v[2*i], v1 = v[2*i+1];
                v[2*i]   = v0*cr[i] - v1*sr[i];
                v[2*i+1] = v0*sr[i] + v1*cr[i];
            }
        }
    }
}

/* =========================================================================
 * Causal mask
 * ====================================================================== */

void apply_causal_mask(Matrix *scores, int window_size) {
    int seq_len = scores->rows;
    for (int i = 0; i < seq_len; i++) {
        float *row = scores->data + i * seq_len;
        for (int j = 0; j < seq_len; j++) {
            if (j > i || (window_size > 0 && i - j >= window_size))
                row[j] = -FLT_MAX;
        }
    }
}

/* =========================================================================
 * Attention forward — fully sgemm-accelerated
 * ====================================================================== */

void attention_forward(const Matrix *x,
                       const AttentionWeights *w,
                       const RoPEBuffer *rb,
                       const ModelConfig *cfg,
                       Matrix *Q, Matrix *K, Matrix *V,
                       Matrix *scores, Matrix *attn,
                       Matrix *out) {
    int seq  = x->rows;
    int d    = cfg->d_model;
    int nh   = cfg->n_heads;
    int nkv  = cfg->n_kv_heads;
    int hd   = cfg->head_dim;
    int win  = cfg->window_size;
    int gqa  = nh / nkv;
    float scale = 1.f / sqrtf((float)hd);

    /* 1. Project Q, K, V */
    sgemm_nn(seq, nh*hd,  d, x->data, w->W_Q.data, Q->data);
    sgemm_nn(seq, nkv*hd, d, x->data, w->W_K.data, K->data);
    sgemm_nn(seq, nkv*hd, d, x->data, w->W_V.data, V->data);

    /* 2. Apply RoPE */
    rope_apply(Q, rb, nh,  hd, 0);
    rope_apply(K, rb, nkv, hd, 0);

    /* 3. Per-head attention with GQA */
    mat_zero(attn);

    for (int qh = 0; qh < nh; qh++) {
        int kv_h = qh / gqa;

        /*
         * Build strided views:
         * Q_h: seq rows, each row offset by qh*hd within nh*hd columns
         * K_h: seq rows, each row offset by kv_h*hd within nkv*hd columns
         *
         * We can't use sgemm on strided columns directly, so we extract
         * each head into a contiguous scratch buffer.
         *
         * For small head_dim (64) this copy is cheap vs the sgemm benefit.
         */
        float *Q_h = (float *)malloc((size_t)seq * hd * sizeof(float));
        float *K_h = (float *)malloc((size_t)seq * hd * sizeof(float));
        float *V_h = (float *)malloc((size_t)seq * hd * sizeof(float));
        float *A_h = (float *)malloc((size_t)seq * seq * sizeof(float));

        for (int t = 0; t < seq; t++) {
            memcpy(Q_h + t*hd, Q->data + t*nh*hd  + qh*hd,  hd*sizeof(float));
            memcpy(K_h + t*hd, K->data + t*nkv*hd + kv_h*hd, hd*sizeof(float));
            memcpy(V_h + t*hd, V->data + t*nkv*hd + kv_h*hd, hd*sizeof(float));
        }

        /* scores = Q_h × K_hᵀ × scale  [seq × seq] */
        sgemm_nt(seq, seq, hd, Q_h, K_h, hd, A_h);
        for (int i = 0; i < seq*seq; i++) A_h[i] *= scale;

        /* Causal mask */
        Matrix score_view = { A_h, NULL, seq, seq };
        apply_causal_mask(&score_view, win);

        /* Softmax row-wise */
        softmax_rows(&score_view);

        /* Save scores for backward */
        memcpy(scores->data + qh * seq * seq, A_h, seq*seq*sizeof(float));

        /* out_h = A_h × V_h  [seq × hd] */
        float *out_h = (float *)malloc((size_t)seq * hd * sizeof(float));
        sgemm_nn(seq, hd, seq, A_h, V_h, out_h);

        /* Scatter back into attn [seq × nh*hd] */
        for (int t = 0; t < seq; t++)
            memcpy(attn->data + t*nh*hd + qh*hd, out_h + t*hd, hd*sizeof(float));

        free(Q_h); free(K_h); free(V_h); free(A_h); free(out_h);
    }

    /* 4. Output projection */
    sgemm_nn(seq, d, nh*hd, attn->data, w->W_O.data, out->data);
}

/* =========================================================================
 * Attention backward — all scalar triple-loops replaced with sgemm
 *
 * Shapes recap:
 *   x:     [seq × d]
 *   Q:     [seq × nh*hd]      K,V: [seq × nkv*hd]
 *   scores:[nh × seq × seq]   attn:[seq × nh*hd]
 *   d_out: [seq × d]
 *
 * Operations:
 *   d_attn = d_out × W_Oᵀ              [seq×d] × [d×nh*hd]  → [seq×nh*hd]
 *   dW_O  += attnᵀ × d_out             [nh*hd×seq] × [seq×d] → [nh*hd×d]
 *   per head:
 *     d_V_h   += scores_hᵀ × d_attn_h  [seq×seq]ᵀ × [seq×hd]
 *     d_S_h    = d_attn_h × V_hᵀ       [seq×hd] × [hd×seq]
 *     softmax backward (scalar, O(seq²) but just multiplications)
 *     d_Q_h   += d_S_h × K_h           [seq×seq] × [seq×hd]
 *     d_K_h   += d_S_hᵀ × Q_h          [seq×seq]ᵀ × [seq×hd]
 *   dW_Q += xᵀ × d_Q                   [d×seq] × [seq×nh*hd]
 *   dW_K += xᵀ × d_K                   same
 *   dW_V += xᵀ × d_V
 *   d_x   = d_Q × W_Qᵀ + d_K × W_Kᵀ + d_V × W_Vᵀ
 * ====================================================================== */

void attention_backward(const Matrix *x,
                        const AttentionWeights *w,
                        const ModelConfig *cfg,
                        const Matrix *Q, const Matrix *K, const Matrix *V,
                        const Matrix *scores, const Matrix *attn,
                        const Matrix *d_out,
                        Matrix *d_x) {
    int seq  = x->rows;
    int d    = cfg->d_model;
    int nh   = cfg->n_heads;
    int nkv  = cfg->n_kv_heads;
    int hd   = cfg->head_dim;
    int gqa  = nh / nkv;
    float scale = 1.f / sqrtf((float)hd);

    /* d_attn = d_out × W_Oᵀ  [seq×d] × [d×nh*hd] → [seq×nh*hd] */
    float *d_attn = (float *)calloc((size_t)seq * nh * hd, sizeof(float));
    /* W_O is [nh*hd × d], so W_Oᵀ is [d × nh*hd]:
       d_attn[seq × nh*hd] = d_out[seq×d] × W_O[nh*hd×d]ᵀ
       = sgemm(NoTrans, Trans, seq, nh*hd, d) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, nh*hd, d, 1.f,
                d_out->data, d,
                w->W_O.data, d,
                0.f, d_attn, nh*hd);

    /* dW_O += attnᵀ × d_out  [nh*hd×seq] × [seq×d] → [nh*hd×d] */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                nh*hd, d, seq, 1.f,
                attn->data, nh*hd,
                d_out->data, d,
                1.f, w->W_O.grad, d);

    /* Accumulators for d_Q, d_K, d_V */
    float *d_Q = (float *)calloc((size_t)seq * nh  * hd, sizeof(float));
    float *d_K = (float *)calloc((size_t)seq * nkv * hd, sizeof(float));
    float *d_V = (float *)calloc((size_t)seq * nkv * hd, sizeof(float));

    for (int qh = 0; qh < nh; qh++) {
        int kv_h = qh / gqa;
        const float *S_h   = scores->data + qh * seq * seq; /* softmax probs */

        /* Extract contiguous head slices */
        float *Q_h    = malloc((size_t)seq*hd*sizeof(float));
        float *K_h    = malloc((size_t)seq*hd*sizeof(float));
        float *V_h    = malloc((size_t)seq*hd*sizeof(float));
        float *dA_h   = malloc((size_t)seq*hd*sizeof(float)); /* d_attn slice */
        float *dS_h   = malloc((size_t)seq*seq*sizeof(float));

        for (int t = 0; t < seq; t++) {
            memcpy(Q_h  + t*hd, Q->data + t*nh*hd  + qh*hd,   hd*sizeof(float));
            memcpy(K_h  + t*hd, K->data + t*nkv*hd + kv_h*hd, hd*sizeof(float));
            memcpy(V_h  + t*hd, V->data + t*nkv*hd + kv_h*hd, hd*sizeof(float));
            memcpy(dA_h + t*hd, d_attn  + t*nh*hd  + qh*hd,   hd*sizeof(float));
        }

        /* d_V_h += S_hᵀ × dA_h  [seq×seq]ᵀ × [seq×hd] → [seq×hd] */
        float *dV_h = (float *)calloc((size_t)seq*hd, sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    seq, hd, seq, 1.f, S_h, seq, dA_h, hd, 0.f, dV_h, hd);

        /* Scatter dV_h back into d_V */
        for (int t = 0; t < seq; t++) {
            float *dst = d_V + t*nkv*hd + kv_h*hd;
            const float *src = dV_h + t*hd;
            for (int k = 0; k < hd; k++) dst[k] += src[k];
        }
        free(dV_h);

        /* dS_raw = dA_h × V_hᵀ  [seq×hd] × [hd×seq] → [seq×seq] */
        sgemm_nt(seq, seq, hd, dA_h, V_h, hd, dS_h);

        /* Softmax backward (element-wise): dS[i][j] = S[i][j]*(dS[i][j] - Σ_k S[i][k]*dS[i][k])
         * This is O(seq²) but only scalar multiply-adds, not triple-nested */
        for (int i = 0; i < seq; i++) {
            const float *si = S_h  + i*seq;
            float       *di = dS_h + i*seq;
            float dot = 0.f;
            for (int j = 0; j < seq; j++) dot += si[j]*di[j];
            for (int j = 0; j < seq; j++) di[j] = si[j]*(di[j]-dot) * scale;
        }

        /* d_Q_h += dS_h × K_h   [seq×seq] × [seq×hd] → [seq×hd] */
        float *dQ_h = (float *)calloc((size_t)seq*hd, sizeof(float));
        sgemm_nn(seq, hd, seq, dS_h, K_h, dQ_h);

        /* d_K_h += dS_hᵀ × Q_h  [seq×seq]ᵀ × [seq×hd] → [seq×hd] */
        float *dK_h = (float *)calloc((size_t)seq*hd, sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    seq, hd, seq, 1.f, dS_h, seq, Q_h, hd, 0.f, dK_h, hd);

        /* Scatter back into d_Q, d_K */
        for (int t = 0; t < seq; t++) {
            float *dq = d_Q + t*nh*hd  + qh*hd;
            float *dk = d_K + t*nkv*hd + kv_h*hd;
            for (int k = 0; k < hd; k++) { dq[k] += dQ_h[t*hd+k]; dk[k] += dK_h[t*hd+k]; }
        }

        free(Q_h); free(K_h); free(V_h); free(dA_h);
        free(dS_h); free(dQ_h); free(dK_h);
    }

    /* dW_Q += xᵀ × d_Q  [d×seq] × [seq×nh*hd] → [d×nh*hd] */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, nh*hd, seq, 1.f,
                x->data, d, d_Q, nh*hd,
                1.f, w->W_Q.grad, nh*hd);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, nkv*hd, seq, 1.f,
                x->data, d, d_K, nkv*hd,
                1.f, w->W_K.grad, nkv*hd);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, nkv*hd, seq, 1.f,
                x->data, d, d_V, nkv*hd,
                1.f, w->W_V.grad, nkv*hd);

    /* d_x = d_Q × W_Qᵀ + d_K × W_Kᵀ + d_V × W_Vᵀ */
    mat_zero(d_x);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, d, nh*hd, 1.f,
                d_Q, nh*hd, w->W_Q.data, nh*hd,
                0.f, d_x->data, d);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, d, nkv*hd, 1.f,
                d_K, nkv*hd, w->W_K.data, nkv*hd,
                1.f, d_x->data, d);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, d, nkv*hd, 1.f,
                d_V, nkv*hd, w->W_V.data, nkv*hd,
                1.f, d_x->data, d);

    free(d_attn); free(d_Q); free(d_K); free(d_V);
}
