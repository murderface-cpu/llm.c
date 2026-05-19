/**
 * tools/train_nested.c
 *
 * Nested Learning training extension.
 *
 * Loads an existing checkpoint, attaches Continuum Memory Blocks (one per
 * transformer layer), then runs a fine-tuning loop using the Multi-Rate
 * AdamW with optional deep optimiser.
 *
 * The standard transformer weights continue to train with the normal schedule.
 * The CMB parameters train at a smaller LR and update frequency — they act
 * as a persistent external memory that learns orthogonal features to the
 * attention layers.
 *
 * Usage:
 *   ./build/train_nested <checkpoint.bin> <tokenizer.vocab> [options]
 *
 * Options:
 *   --data_dir  PATH   where to find train.bin / val.bin  (default: data/)
 *   --steps     INT    fine-tuning steps                  (default: 1000)
 *   --lr        FLOAT  peak learning rate                 (default: 1e-4)
 *   --cmb_slots INT    memory slots per CMB               (default: 64)
 *   --cmb_dim   INT    CMB internal dimension             (default: 128)
 *   --cmb_freq  INT    CMB update frequency in steps      (default: 4)
 *   --no_deep_opt      disable deep optimiser (use standard AdamW)
 *   --save      PATH   where to write the augmented checkpoint
 *   --cmb_init  PATH   load pre-trained CMB from a previous run
 */

#define _GNU_SOURCE
#include <stdio.h>
#ifdef USE_OPENBLAS
#  include <cblas.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#include "../include/transformer.h"
#include "../include/tokenizer.h"
#include "../include/nested.h"
#include "../include/config.h"
#include "../include/inference.h"

/* =========================================================================
 * Config
 * ====================================================================== */
typedef struct {
    const char *checkpoint;
    const char *vocab_path;
    const char *data_dir;
    const char *save_path;
    const char *cmb_init;
    int   steps;
    float lr;
    int   cmb_slots;
    int   cmb_dim;
    int   cmb_freq;
    int   use_deep_opt;
    int   batch_size;
    int   seq_len;
    float label_smoothing;
} NLTrainConfig;

static NLTrainConfig default_cfg(void) {
    return (NLTrainConfig){
        .checkpoint     = NULL,
        .vocab_path     = NULL,
        .data_dir       = "data",
        .save_path      = "checkpoints/nested_ckpt.bin",
        .cmb_init       = NULL,
        .steps          = 1000,
        .lr             = 1e-4f,
        .cmb_slots      = 64,
        .cmb_dim        = 128,
        .cmb_freq       = 4,
        .use_deep_opt   = 1,
        .batch_size     = 4,
        .seq_len        = 128,
        .label_smoothing= 0.0f,
    };
}

static void print_usage(const char *p) {
    fprintf(stderr,
        "Usage: %s <checkpoint.bin> <tokenizer.vocab> [options]\n"
        "  --data_dir PATH   (default: data/)\n"
        "  --steps    INT    (default: 1000)\n"
        "  --lr       FLOAT  (default: 1e-4)\n"
        "  --cmb_slots INT   (default: 64)\n"
        "  --cmb_dim  INT    (default: 128)\n"
        "  --cmb_freq INT    (default: 4)\n"
        "  --no_deep_opt     disable deep optimiser\n"
        "  --save     PATH   output checkpoint\n"
        "  --cmb_init PATH   load existing CMB weights\n",
        p);
}

static int parse_args(int argc, char **argv, NLTrainConfig *cfg) {
    if (argc < 3) { print_usage(argv[0]); return -1; }
    cfg->checkpoint = argv[1];
    cfg->vocab_path = argv[2];
    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i],"--data_dir")  && i+1<argc) cfg->data_dir   = argv[++i];
        else if (!strcmp(argv[i],"--steps")     && i+1<argc) cfg->steps      = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--lr")        && i+1<argc) cfg->lr         = atof(argv[++i]);
        else if (!strcmp(argv[i],"--cmb_slots") && i+1<argc) cfg->cmb_slots  = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--cmb_dim")   && i+1<argc) cfg->cmb_dim    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--cmb_freq")  && i+1<argc) cfg->cmb_freq   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--save")      && i+1<argc) cfg->save_path  = argv[++i];
        else if (!strcmp(argv[i],"--cmb_init")  && i+1<argc) cfg->cmb_init   = argv[++i];
        else if (!strcmp(argv[i],"--no_deep_opt")) cfg->use_deep_opt = 0;
        else { fprintf(stderr,"Unknown arg: %s\n", argv[i]); return -1; }
    }
    return 0;
}

/* =========================================================================
 * Data loader (reuse the binary format from prepare_data)
 * ====================================================================== */
typedef struct { int32_t *tokens; long n; long pos; } DL;

static DL *dl_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    int32_t n = 0;
    if (fread(&n, 4, 1, f) != 1 || n <= 0) { fclose(f); return NULL; }
    DL *dl = calloc(1, sizeof(DL));
    dl->n = n;
    dl->tokens = malloc((size_t)n * sizeof(int32_t));
    if (fread(dl->tokens, 4, n, f) != (size_t)n) { fclose(f); free(dl->tokens); free(dl); return NULL; }
    fclose(f);
    return dl;
}
static void dl_free(DL *dl) { free(dl->tokens); free(dl); }
static void dl_batch(DL *dl, int *in, int *tgt, int seq) {
    if (dl->pos + seq + 1 > dl->n) dl->pos = 0;
    for (int i = 0; i < seq; i++) {
        in[i]  = dl->tokens[dl->pos + i];
        tgt[i] = dl->tokens[dl->pos + i + 1];
    }
    dl->pos += seq;
}

/* =========================================================================
 * Cross-entropy loss (same as main training loop)
 * ====================================================================== */
static float ce_loss_and_grad(const float *logits, const int *targets,
                               float *d_logits, int seq, int V,
                               float ls) {
    float loss = 0.f;
    for (int t = 0; t < seq; t++) {
        const float *lg = logits   + t * V;
        float       *dl = d_logits + t * V;
        float vmax = lg[0];
        for (int j = 1; j < V; j++) if (lg[j] > vmax) vmax = lg[j];
        float sum = 0.f;
        for (int j = 0; j < V; j++) { dl[j] = expf(lg[j]-vmax); sum += dl[j]; }
        float inv = 1.f / sum;
        for (int j = 0; j < V; j++) dl[j] *= inv;
        int y = targets[t];
        loss += -logf(dl[y] + 1e-10f);
        if (ls > 0.f) {
            float st = ls / V;
            for (int j = 0; j < V; j++) dl[j] -= (j==y) ? (1.f-ls+st) : st;
        } else {
            dl[y] -= 1.f;
        }
        for (int j = 0; j < V; j++) dl[j] /= (float)seq;
    }
    return loss / (float)seq;
}

/* =========================================================================
 * Cosine LR schedule with warmup
 * ====================================================================== */
static float cosine_lr(int step, int warmup, int total, float peak, float min_lr) {
    if (step < warmup) return peak * (float)(step+1) / (float)warmup;
    float t = (float)(step - warmup) / (float)(total - warmup + 1);
    return min_lr + (peak - min_lr) * 0.5f * (1.f + cosf((float)M_PI * t));
}

/* =========================================================================
 * Main
 * ====================================================================== */
int main(int argc, char **argv) {
    NLTrainConfig cfg = default_cfg();
    if (parse_args(argc, argv, &cfg) < 0) return 1;

    /* ── Load base model ─────────────────────────────────────────────── */
    printf("[nl] loading base model from %s\n", cfg.checkpoint);
    Model *m = model_load(cfg.checkpoint);
    if (!m) { fprintf(stderr, "Cannot load model\n"); return 1; }
    printf("[nl] base model: %ldM params, d=%d, layers=%d\n",
           model_param_count(m)/1000000,
           m->cfg.d_model, m->cfg.n_layers);

    /* ── Load tokenizer ──────────────────────────────────────────────── */
    Tokenizer *tok = tok_load(cfg.vocab_path);
    if (!tok) { fprintf(stderr, "Cannot load tokenizer\n"); model_free(m); return 1; }

    int d      = m->cfg.d_model;
    int V      = m->cfg.vocab_size;
    int L      = m->cfg.n_layers;
    int seq    = cfg.seq_len;
    int cdim   = cfg.cmb_dim;
    int slots  = cfg.cmb_slots;

    /* ── Allocate CMBs (one per layer) ───────────────────────────────── */
    printf("[nl] allocating %d CMBs (slots=%d dim=%d update_freq=%d)\n",
           L, slots, cdim, cfg.cmb_freq);

    CMBWeights *cmbs       = malloc(L * sizeof(CMBWeights));
    CMBCache   *cmb_caches = malloc(L * sizeof(CMBCache));
    Matrix     *x_cmb_in   = malloc(L * sizeof(Matrix)); /* saved x before CMB */

    for (int l = 0; l < L; l++) {
        cmbs[l]       = cmb_alloc(d, slots, cdim);
        cmb_caches[l] = cmb_cache_alloc(seq, slots, cdim, d);
        x_cmb_in[l]   = mat_alloc(seq, d, 0);

        if (cfg.cmb_init) {
            char path[512];
            snprintf(path, sizeof(path), "%s.layer%d", cfg.cmb_init, l);
            cmb_load(&cmbs[l], path);
        }
    }

    /* ── Multi-rate optimiser setup ──────────────────────────────────── */
    /*
     * Parameter groups:
     *   0:      embedding table
     *   1..L:   transformer blocks (early / late split)
     *   L+1:    LM head
     *   L+2..2L+1: CMB parameters (one group per layer, all 4 matrices)
     *
     * For simplicity we treat each transformer block and each CMB as one
     * flat parameter group by collecting all their data/grad pointers.
     */
    NLConfig nlcfg = nl_config_default();
    nlcfg.cmb_slots       = slots;
    nlcfg.cmb_dim         = cdim;
    nlcfg.cmb_update_freq = cfg.cmb_freq;
    nlcfg.use_deep_optim  = cfg.use_deep_opt;

    /* Collect all parameter groups */
    /* 1 embed + L*7 attention/ffn + 1 LM head + L*4 CMB params */
    int    MAX_GROUPS = 1 + L * 7 + 1 + L * 4 + 16;
    float **pd = malloc(MAX_GROUPS * sizeof(float *));
    float **pg = malloc(MAX_GROUPS * sizeof(float *));
    int    *pn = malloc(MAX_GROUPS * sizeof(int));
    float  *ls = malloc(MAX_GROUPS * sizeof(float));
    int    *uf = malloc(MAX_GROUPS * sizeof(int));
    int     ng = 0;

    /* Embedding */
    pd[ng]=m->weights.embed.data; pg[ng]=m->weights.embed.grad;
    pn[ng]=m->weights.embed.rows*m->weights.embed.cols;
    ls[ng]=nlcfg.embed_lr_scale; uf[ng]=1; ng++;

    /* Transformer blocks */
    for (int l = 0; l < L; l++) {
        BlockWeights *bw = &m->weights.blocks[l];
        float lr_scale = (l < L/2) ? nlcfg.early_lr_scale : nlcfg.late_lr_scale;
        /* Flatten all 4 attention matrices + 3 FFN matrices into one group */
        /* For simplicity use W_O as representative — in production you'd
         * split these into separate groups. Here we just update all
         * matrices with the same scale by linking them. */
        /* We add each matrix as a separate group since MRAdamState needs
         * individual n values. */
#define ADD_MAT(mat) \
        pd[ng]=(mat).data; pg[ng]=(mat).grad; \
        pn[ng]=(mat).rows*(mat).cols; \
        ls[ng]=lr_scale; uf[ng]=1; ng++

        ADD_MAT(bw->attn.W_Q); ADD_MAT(bw->attn.W_K);
        ADD_MAT(bw->attn.W_V); ADD_MAT(bw->attn.W_O);
        if (!bw->use_moe) {
            ADD_MAT(bw->ffn.W_gate); ADD_MAT(bw->ffn.W_up);
            ADD_MAT(bw->ffn.W_down);
        }
#undef ADD_MAT
    }

    /* LM head */
    pd[ng]=m->weights.lm_head.data; pg[ng]=m->weights.lm_head.grad;
    pn[ng]=m->weights.lm_head.rows*m->weights.lm_head.cols;
    ls[ng]=nlcfg.late_lr_scale; uf[ng]=1; ng++;

    /* CMB groups (one per layer, treated as one flat group per CMB) */
    for (int l = 0; l < L; l++) {
        /* We only track keys/values (the persistent memory) separately
         * with the slow update schedule. W_q and W_out train normally. */
        /* keys + values as one group */
        int kv_n = cmbs[l].keys.rows * cmbs[l].keys.cols;
        pd[ng]=cmbs[l].keys.data;   pg[ng]=cmbs[l].keys.grad;
        pn[ng]=kv_n; ls[ng]=nlcfg.cmb_lr_scale; uf[ng]=cfg.cmb_freq; ng++;

        pd[ng]=cmbs[l].values.data; pg[ng]=cmbs[l].values.grad;
        pn[ng]=kv_n; ls[ng]=nlcfg.cmb_lr_scale; uf[ng]=cfg.cmb_freq; ng++;

        pd[ng]=cmbs[l].W_q.data;   pg[ng]=cmbs[l].W_q.grad;
        pn[ng]=cmbs[l].W_q.rows*cmbs[l].W_q.cols;
        ls[ng]=nlcfg.cmb_lr_scale * 2.f; uf[ng]=1; ng++;

        pd[ng]=cmbs[l].W_out.data;  pg[ng]=cmbs[l].W_out.grad;
        pn[ng]=cmbs[l].W_out.rows*cmbs[l].W_out.cols;
        ls[ng]=nlcfg.cmb_lr_scale * 2.f; uf[ng]=1; ng++;
    }

    printf("[nl] %d parameter groups\n", ng);

    MRAdamState *adam = mr_adam_init(ng, ls, uf, pn, cfg.use_deep_opt);

    /* ── Data loaders ────────────────────────────────────────────────── */
    char train_path[512], val_path[512];
    snprintf(train_path, sizeof(train_path), "%s/train.bin", cfg.data_dir);
    snprintf(val_path,   sizeof(val_path),   "%s/val.bin",   cfg.data_dir);
    DL *train_dl = dl_open(train_path);
    DL *val_dl   = dl_open(val_path);
    if (!train_dl) { fprintf(stderr, "Cannot open %s\n", train_path); return 1; }

    /* ── Scratch buffers ─────────────────────────────────────────────── */
    int   *in_buf  = malloc(seq * sizeof(int));
    int   *tgt_buf = malloc(seq * sizeof(int));
    Matrix logits  = mat_alloc(seq, V, 0);
    Matrix d_logits= mat_alloc(seq, V, 0);
    Matrix x_final = mat_alloc(seq, d, 0); /* residual after all blocks + CMBs */

    /* ── Training loop ───────────────────────────────────────────────── */
    printf("\n[nl] training config:\n");
    printf("  steps=%d  lr=%.2e  batch=%d  seq=%d\n",
           cfg.steps, (double)cfg.lr, cfg.batch_size, seq);
    printf("  deep_optim=%s  cmb_freq=%d\n\n",
           cfg.use_deep_opt ? "yes" : "no", cfg.cmb_freq);
    printf("%-6s  %-8s  %-8s  %-8s  %s\n",
           "step","loss","smooth","val","ms/step");
    printf("----------------------------------------------\n");

    double smooth = -1.0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < cfg.steps; step++) {
        struct timespec st0, st1;
        clock_gettime(CLOCK_MONOTONIC, &st0);

        float lr = cosine_lr(step, cfg.steps/20, cfg.steps, cfg.lr, cfg.lr*0.1f);

        /* Zero all grads */
        model_zero_grads(m);
        for (int l = 0; l < L; l++) {
            mat_zero_grad(&cmbs[l].keys);
            mat_zero_grad(&cmbs[l].values);
            mat_zero_grad(&cmbs[l].W_q);
            mat_zero_grad(&cmbs[l].W_out);
        }

        float step_loss = 0.f;

        for (int b = 0; b < cfg.batch_size; b++) {
            dl_batch(train_dl, in_buf, tgt_buf, seq);

            /* ── Forward pass with CMBs ─────────────────────────────── */
            /* 1. Embed */
            Matrix x = mat_alloc(seq, d, 0);
            embed_lookup(&m->weights.embed, in_buf, seq, &x);

            /* 2. Run through blocks, inserting CMB after each block */
            for (int l = 0; l < L; l++) {
                /* Resize caches to seq */
                ForwardCache *fc = &m->caches[l];
                fc->x_norm_attn.rows=seq; fc->x_norm_ffn.rows=seq;
                fc->attn_out.rows=seq;    fc->ffn_out.rows=seq;
                fc->Q.rows=seq;           fc->K.rows=seq;
                fc->V.rows=seq;           fc->attn_weighted.rows=seq;
                fc->attn_scores.rows=m->cfg.n_heads*seq;
                fc->attn_scores.cols=seq;
                fc->gate_buf.rows=seq;    fc->up_buf.rows=seq;

                /* Standard transformer block */
                block_forward(&x, &m->weights.blocks[l],
                              &m->caches[l], &m->rope, &m->cfg, l);

                /* Save x before CMB for backward */
                mat_copy(&x_cmb_in[l], &x);

                /* CMB forward (adds to x in-place) */
                cmb_forward(&x, &cmbs[l], &cmb_caches[l],
                            m->cfg.rms_norm_eps);
            }

            /* 3. Final RMSNorm + LM head */
            rms_norm(&x, m->weights.rms_final, m->cfg.rms_norm_eps, &x_final);
            mat_mul(&x_final, &m->weights.lm_head, &logits);

            /* ── Loss ───────────────────────────────────────────────── */
            step_loss += ce_loss_and_grad(logits.data, tgt_buf,
                                          d_logits.data, seq, V,
                                          cfg.label_smoothing);

            /* ── Backward through LM head and final norm ────────────── */
            /* dW_lm += x_finalᵀ × d_logits */
#ifdef USE_OPENBLAS
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        d, V, seq, 1.f,
                        x_final.data, d, d_logits.data, V,
                        1.f, m->weights.lm_head.grad, V);
            /* dx_final = d_logits × W_lmᵀ */
            Matrix dx_final = mat_alloc(seq, d, 0);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq, d, V, 1.f,
                        d_logits.data, V, m->weights.lm_head.data, V,
                        0.f, dx_final.data, d);
#else
            Matrix dx_final = mat_alloc(seq, d, 0);
            for (int i=0;i<seq;i++) for(int j=0;j<d;j++) {
                float xj=x_final.data[i*d+j]; if(!xj) continue;
                float *grow=m->weights.lm_head.grad+j*V;
                for(int k=0;k<V;k++) grow[k]+=xj*d_logits.data[i*V+k];
            }
            mat_mul_T(&d_logits, &m->weights.lm_head, &dx_final);
#endif

            /* Backprop through final rms_norm into d_x */
            float *d_rms_final = calloc(d, sizeof(float));
            Matrix d_x = mat_alloc(seq, d, 0);
            rms_norm_backward(&x, m->weights.rms_final,
                              m->cfg.rms_norm_eps, &dx_final, &d_x,
                              d_rms_final);
            free(d_rms_final);
            mat_free(&dx_final);

            /* ── Backward through blocks + CMBs in reverse ──────────── */
            for (int l = L-1; l >= 0; l--) {
                /* CMB backward first (it sits after the block) */
                cmb_backward(&x_cmb_in[l], &cmbs[l], &cmb_caches[l],
                             &d_x, m->cfg.rms_norm_eps);

                /* Then standard block backward */
                BlockWeights *bw = &m->weights.blocks[l];
                ForwardCache *cache = &m->caches[l];

                Matrix d_x_new = mat_alloc(seq, d, 0);
                float *d_rms_ffn  = calloc(d, sizeof(float));
                float *d_rms_attn = calloc(d, sizeof(float));
                Matrix d_xn_ffn   = mat_alloc(seq, d, 0);
                Matrix d_xn_attn  = mat_alloc(seq, d, 0);

                rms_norm_backward(&x_cmb_in[l], bw->rms_ffn,
                                  m->cfg.rms_norm_eps, &d_x,
                                  &d_xn_ffn, d_rms_ffn);

                rms_norm_backward(&x_cmb_in[l], bw->rms_attn,
                                  m->cfg.rms_norm_eps, &d_xn_ffn,
                                  &d_xn_attn, d_rms_attn);

                attention_backward(&x_cmb_in[l], &bw->attn, &m->cfg,
                                   &cache->Q, &cache->K, &cache->V,
                                   &cache->attn_scores, &cache->attn_weighted,
                                   &d_xn_attn, &d_x_new);

                mat_add_inplace(&d_x_new, &d_x);
                free(d_x.data); d_x.data = d_x_new.data;
                d_x.rows = d_x_new.rows; d_x.cols = d_x_new.cols;
                d_x_new.data = NULL;

                free(d_rms_ffn); free(d_rms_attn);
                mat_free(&d_xn_ffn); mat_free(&d_xn_attn);
            }

            embed_backward(&m->weights.embed, in_buf, seq, &d_x);
            mat_free(&d_x);
            mat_free(&x);
        }
        step_loss /= (float)cfg.batch_size;

        /* Gradient clipping across all groups */
        float gnorm_sq = 0.f;
        for (int i = 0; i < ng; i++)
            for (int j = 0; j < pn[i]; j++)
                gnorm_sq += pg[i][j] * pg[i][j];
        float gnorm = sqrtf(gnorm_sq);
        float clip  = 1.0f;
        if (gnorm > clip) {
            float s = clip / (gnorm + 1e-6f);
            for (int i = 0; i < ng; i++)
                for (int j = 0; j < pn[i]; j++)
                    pg[i][j] *= s;
        }

        /* Multi-rate optimiser step */
        mr_adam_step(adam, lr, 0.9f, 0.95f, 1e-8f, 0.1f,
                     pd, pg, ng);

        /* EMA smooth loss */
        if (smooth < 0) smooth = step_loss;
        else smooth = 0.05*step_loss + 0.95*smooth;

        clock_gettime(CLOCK_MONOTONIC, &st1);
        double ms = (st1.tv_sec-st0.tv_sec)*1e3 + (st1.tv_nsec-st0.tv_nsec)*1e-6;

        /* ── Logging ─────────────────────────────────────────────────── */
        if (step % 10 == 0) {
            printf("%-6d  %-8.4f  %-8.4f  %-8s  %.0f  gnorm=%.1f\n",
                   step, step_loss, smooth, "---", ms, (double)gnorm);
            fflush(stdout);
        }

        /* Validation */
        if (val_dl && step > 0 && step % 100 == 0) {
            float vl = 0.f;
            for (int vs = 0; vs < 20; vs++) {
                dl_batch(val_dl, in_buf, tgt_buf, seq);
                /* quick forward-only pass */
                Matrix xv = mat_alloc(seq, d, 0);
                embed_lookup(&m->weights.embed, in_buf, seq, &xv);
                for (int l = 0; l < L; l++) {
                    ForwardCache *fc = &m->caches[l];
                    fc->x_norm_attn.rows=seq; fc->x_norm_ffn.rows=seq;
                    fc->attn_out.rows=seq;    fc->ffn_out.rows=seq;
                    fc->Q.rows=seq; fc->K.rows=seq; fc->V.rows=seq;
                    fc->attn_weighted.rows=seq;
                    fc->attn_scores.rows=m->cfg.n_heads*seq;
                    fc->attn_scores.cols=seq;
                    fc->gate_buf.rows=seq; fc->up_buf.rows=seq;
                    block_forward(&xv, &m->weights.blocks[l],
                                  &m->caches[l], &m->rope, &m->cfg, l);
                    /* CMB forward-only (cache is scratch, not saved) */
                    CMBCache tmpc = cmb_cache_alloc(seq, slots, cdim, d);
                    cmb_forward(&xv, &cmbs[l], &tmpc, m->cfg.rms_norm_eps);
                    cmb_cache_free(&tmpc);
                }
                rms_norm(&xv, m->weights.rms_final, m->cfg.rms_norm_eps, &x_final);
                mat_mul(&x_final, &m->weights.lm_head, &logits);
                for (int t = 0; t < seq; t++) {
                    float *row = logits.data + t*V;
                    int y = tgt_buf[t];
                    float mx = row[0]; for(int j=1;j<V;j++) if(row[j]>mx) mx=row[j];
                    float s = 0.f; for(int j=0;j<V;j++) s+=expf(row[j]-mx);
                    vl += -(row[y]-mx) + logf(s);
                }
                vl /= seq;
                mat_free(&xv);
            }
            vl /= 20.f;
            printf("  [val] step=%-4d  val=%.4f  smooth=%.4f  gap=%.4f\n",
                   step, vl, (float)smooth, vl-(float)smooth);
        }

        /* Checkpoint */
        if (step > 0 && step % 500 == 0) {
            model_save(m, cfg.save_path);
            printf("  [ckpt] saved %s\n", cfg.save_path);
            for (int l = 0; l < L; l++) {
                char cp[512];
                snprintf(cp, sizeof(cp), "%s.cmb.layer%d", cfg.save_path, l);
                cmb_save(&cmbs[l], cp);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("\n[nl] done in %.1fs\n", total);

    /* ── Final save ──────────────────────────────────────────────────── */
    mkdir("checkpoints", 0755);
    model_save(m, cfg.save_path);
    printf("[nl] model saved: %s\n", cfg.save_path);
    for (int l = 0; l < L; l++) {
        char cp[512];
        snprintf(cp, sizeof(cp), "%s.cmb.layer%d", cfg.save_path, l);
        cmb_save(&cmbs[l], cp);
        printf("[nl] CMB layer %d saved: %s\n", l, cp);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    for (int l = 0; l < L; l++) {
        cmb_free(&cmbs[l]); cmb_cache_free(&cmb_caches[l]);
        mat_free(&x_cmb_in[l]);
    }
    free(cmbs); free(cmb_caches); free(x_cmb_in);
    mr_adam_free(adam);
    free(pd); free(pg); free(pn); free(ls); free(uf);
    mat_free(&logits); mat_free(&d_logits);
    mat_free(&x_final);
    free(in_buf); free(tgt_buf);
    dl_free(train_dl); if (val_dl) dl_free(val_dl);
    tok_free(tok);
    model_free(m);
    return 0;
}
