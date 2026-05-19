/**
 * train.c
 *
 * Training loop, AdamW optimiser, learning rate schedule, and
 * knowledge distillation loss support.
 *
 * AdamW (Adam + decoupled weight decay):
 *   m_t = β₁·m_{t-1} + (1-β₁)·g_t          <- first moment  (mean)
 *   v_t = β₂·v_{t-1} + (1-β₂)·g_t²         <- second moment (variance)
 *   m̂ = m_t / (1-β₁ᵗ)                       <- bias correction
 *   v̂ = v_t / (1-β₂ᵗ)                       <- bias correction
 *   θ_t = θ_{t-1} * (1 - lr·wd) - lr·m̂/(√v̂+ε)  <- AdamW update
 *
 * Decoupling weight decay from the gradient update (the "W" in AdamW)
 * improves regularisation vs vanilla Adam+L2.
 *
 * LR Schedule:
 *   1. Linear warmup for `warmup_steps` steps.
 *   2. Cosine decay from peak lr to lr_min over remaining steps.
 */

#define _GNU_SOURCE  /* expose M_PI on Linux */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "../include/transformer.h"
#include "../include/tokenizer.h"
#include "../include/config.h"

/* =========================================================================
 * AdamW optimiser state
 * ====================================================================== */

/**
 * AdamState — one pair of moment buffers per parameter matrix.
 * We store them in a flat array matching model_collect_params().
 */
typedef struct {
    float *m;   /* first  moment (mean of gradients)       */
    float *v;   /* second moment (variance of gradients)   */
    int    n;   /* number of scalar parameters in this slot */
} AdamSlot;

typedef struct {
    AdamSlot *slots;
    int       n_slots;
    int       step;     /* current optimiser step (for bias correction) */
} AdamState;

/* -------------------------------------------------------------------------
 * Parameter iterator — yields every (data, grad, n) tuple in the model.
 * This is the single source of truth for "all optimisable parameters."
 *
 * We use a callback rather than a flat copy so we never need extra memory
 * proportional to model size just to list parameters.
 * ---------------------------------------------------------------------- */
typedef void (*ParamCallback)(float *data, float *grad, int n, void *ctx);

static void model_iterate_params(Model *m, ParamCallback cb, void *ctx) {
    ModelWeights *w = &m->weights;

    /* Embedding */
    cb(w->embed.data, w->embed.grad,
       w->embed.rows * w->embed.cols, ctx);

    for (int l = 0; l < w->n_layers; l++) {
        BlockWeights *bw = &w->blocks[l];

        /* Attention */
        #define YIELD_MAT(mat) \
            cb((mat).data, (mat).grad, (mat).rows * (mat).cols, ctx)

        YIELD_MAT(bw->attn.W_Q);
        YIELD_MAT(bw->attn.W_K);
        YIELD_MAT(bw->attn.W_V);
        YIELD_MAT(bw->attn.W_O);

        /* FFN / MoE */
        if (!bw->use_moe) {
            YIELD_MAT(bw->ffn.W_gate);
            YIELD_MAT(bw->ffn.W_up);
            YIELD_MAT(bw->ffn.W_down);
        } else {
            YIELD_MAT(bw->moe.W_router);
            for (int e = 0; e < bw->moe.n_experts; e++) {
                YIELD_MAT(bw->moe.experts[e].W_gate);
                YIELD_MAT(bw->moe.experts[e].W_up);
                YIELD_MAT(bw->moe.experts[e].W_down);
            }
        }
        #undef YIELD_MAT
    }

    /* LM head */
    cb(w->lm_head.data, w->lm_head.grad,
       w->lm_head.rows * w->lm_head.cols, ctx);
}

/* Count total slots for adam_init */
static void count_cb(float *d, float *g, int n, void *ctx) {
    (void)d; (void)g; (void)n;
    (*(int *)ctx)++;
}

/* Fill adam slots */
typedef struct { AdamSlot *slots; int idx; } FillCtx;
static void fill_cb(float *d, float *g, int n, void *ctx) {
    (void)d; (void)g;
    FillCtx *fc = (FillCtx *)ctx;
    AdamSlot *s = &fc->slots[fc->idx++];
    s->n = n;
    s->m = (float *)calloc(n, sizeof(float));
    s->v = (float *)calloc(n, sizeof(float));
    if (!s->m || !s->v) { fprintf(stderr, "[adam] OOM\n"); abort(); }
}

AdamState *adam_init(Model *m) {
    AdamState *adam = (AdamState *)calloc(1, sizeof(AdamState));
    adam->step = 0;

    int n_slots = 0;
    model_iterate_params(m, count_cb, &n_slots);

    adam->n_slots = n_slots;
    adam->slots   = (AdamSlot *)calloc(n_slots, sizeof(AdamSlot));

    FillCtx fc = { adam->slots, 0 };
    model_iterate_params(m, fill_cb, &fc);

    return adam;
}

void adam_free(AdamState *adam) {
    for (int i = 0; i < adam->n_slots; i++) {
        free(adam->slots[i].m);
        free(adam->slots[i].v);
    }
    free(adam->slots);
    free(adam);
}

/* =========================================================================
 * Learning rate schedule
 * ====================================================================== */

/**
 * lr_schedule — cosine decay with linear warmup.
 *
 * @step:         current training step (0-indexed)
 * @warmup_steps: number of linear warmup steps
 * @max_steps:    total training steps
 * @lr_max:       peak learning rate
 * @lr_min:       minimum learning rate (floor of cosine)
 */
float lr_schedule(int step, int warmup_steps, int max_steps,
                  float lr_max, float lr_min) {
    if (step < warmup_steps) {
        /* Linear warmup */
        return lr_max * (float)(step + 1) / (float)warmup_steps;
    }

    /* Cosine decay from lr_max to lr_min */
    int   decay_steps = max_steps - warmup_steps;
    int   t           = step - warmup_steps;
    float progress    = (float)t / (float)(decay_steps > 0 ? decay_steps : 1);
    float cosine      = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return lr_min + (lr_max - lr_min) * cosine;
}

/* =========================================================================
 * AdamW update step
 * ====================================================================== */

typedef struct {
    AdamSlot *slots;
    int       idx;
    float     lr;
    float     beta1, beta2, epsilon, weight_decay;
    float     bc1, bc2;   /* bias correction denominators */
} AdamCtx;

static void adam_update_cb(float *data, float *grad, int n, void *ctx) {
    AdamCtx  *ac = (AdamCtx *)ctx;
    AdamSlot *s  = &ac->slots[ac->idx++];

    for (int i = 0; i < n; i++) {
        float g = grad[i];

        /* Moment updates */
        s->m[i] = ac->beta1 * s->m[i] + (1.0f - ac->beta1) * g;
        s->v[i] = ac->beta2 * s->v[i] + (1.0f - ac->beta2) * g * g;

        /* Bias-corrected moments */
        float m_hat = s->m[i] / ac->bc1;
        float v_hat = s->v[i] / ac->bc2;

        /* AdamW: decouple weight decay from gradient */
        data[i] *= (1.0f - ac->lr * ac->weight_decay);
        data[i] -= ac->lr * m_hat / (sqrtf(v_hat) + ac->epsilon);
    }
}

/**
 * adam_step — perform one AdamW parameter update.
 * Call after model_backward() has filled all .grad buffers.
 */
void adam_step(AdamState *adam, Model *m, const TrainConfig *tcfg, float lr) {
    adam->step++;
    float t   = (float)adam->step;
    float bc1 = 1.0f - powf(tcfg->beta1, t);
    float bc2 = 1.0f - powf(tcfg->beta2, t);

    AdamCtx ac = {
        .slots        = adam->slots,
        .idx          = 0,
        .lr           = lr,
        .beta1        = tcfg->beta1,
        .beta2        = tcfg->beta2,
        .epsilon      = tcfg->epsilon,
        .weight_decay = tcfg->weight_decay,
        .bc1          = bc1,
        .bc2          = bc2,
    };
    model_iterate_params(m, adam_update_cb, &ac);
}

/* =========================================================================
 * Gradient clipping
 * ====================================================================== */

typedef struct { float sum_sq; } GradNormCtx;
static void grad_norm_cb(float *d, float *g, int n, void *ctx) {
    (void)d;
    GradNormCtx *gc = (GradNormCtx *)ctx;
    for (int i = 0; i < n; i++) gc->sum_sq += g[i] * g[i];
}

typedef struct { float scale; } GradScaleCtx;
static void grad_scale_cb(float *d, float *g, int n, void *ctx) {
    (void)d;
    float s = ((GradScaleCtx *)ctx)->scale;
    for (int i = 0; i < n; i++) g[i] *= s;
}

/**
 * clip_grad_norm — clip all parameter gradients so the global L2 norm
 * does not exceed `max_norm`. Modifies grad buffers in-place.
 * Returns the pre-clip global gradient norm.
 */
float clip_grad_norm(Model *m, float max_norm) {
    GradNormCtx gc = { 0.0f };
    model_iterate_params(m, grad_norm_cb, &gc);

    float norm = sqrtf(gc.sum_sq);
    if (norm > max_norm) {
        GradScaleCtx sc = { max_norm / (norm + 1e-6f) };
        model_iterate_params(m, grad_scale_cb, &sc);
    }
    return norm;
}

/* =========================================================================
 * Data loading (simple binary format)
 *
 * Token data file format:
 *   [int32: n_tokens_total]
 *   [int32 × n_tokens_total: token ids]
 *
 * During training we slide a window of length (seq_len+1) across the data.
 * Input  = tokens[offset .. offset+seq_len-1]
 * Target = tokens[offset+1 .. offset+seq_len]   (next-token prediction)
 * ====================================================================== */

typedef struct {
    int  *tokens;
    long  n_tokens;
    long  pos;       /* current read position */
} DataLoader;

DataLoader *dataloader_init(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    DataLoader *dl = (DataLoader *)calloc(1, sizeof(DataLoader));

    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) {
        fprintf(stderr, "[dataloader] bad header in %s\n", path);
        fclose(f); free(dl); return NULL;
    }
    dl->n_tokens = n;
    dl->tokens   = (int *)malloc((size_t)n * sizeof(int));
    if (!dl->tokens) { fclose(f); free(dl); return NULL; }
    if (fread(dl->tokens, sizeof(int), (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "[dataloader] short read in %s\n", path);
        free(dl->tokens); fclose(f); free(dl); return NULL;
    }
    fclose(f);
    return dl;
}

void dataloader_free(DataLoader *dl) {
    if (!dl) return;
    free(dl->tokens);
    free(dl);
}

/**
 * dataloader_next_batch — fill input/target arrays for one training step.
 * Wraps around to the beginning when the file is exhausted.
 *
 * @in:      pre-allocated int array [batch_size × seq_len]
 * @targets: pre-allocated int array [batch_size × seq_len]
 */
void dataloader_next_batch(DataLoader *dl,
                           int *in, int *targets,
                           int batch_size, int seq_len) {
    int stride = seq_len + 1;  /* +1 for target offset */
    for (int b = 0; b < batch_size; b++) {
        /* Wrap around */
        if (dl->pos + stride > dl->n_tokens)
            dl->pos = 0;

        memcpy(in      + b * seq_len, dl->tokens + dl->pos,     seq_len * sizeof(int));
        memcpy(targets + b * seq_len, dl->tokens + dl->pos + 1, seq_len * sizeof(int));
        dl->pos += seq_len;   /* stride by seq_len, not stride, for overlapping windows */
    }
}

/* =========================================================================
 * Distillation loss
 *
 * KL divergence between teacher soft labels and student logits:
 *   L_KD = KL(teacher_probs || student_probs)
 *        = sum_j teacher_probs[j] * log(teacher_probs[j] / student_probs[j])
 *
 * Temperature T softens distributions:
 *   softmax_T(logits)[j] = exp(logits[j]/T) / sum_k exp(logits[k]/T)
 *
 * Total loss = (1-α)·CE + α·L_KD
 * ====================================================================== */

/**
 * distill_loss_and_grad — compute KD loss and add its gradient to d_logits.
 *
 * @student_logits:  [seq_len × vocab]  (already computed in forward pass)
 * @teacher_logits:  [seq_len × vocab]  (loaded from pre-computed file)
 * @temperature:     softening temperature (2.0 is a common default)
 * @alpha:           weight of distillation loss (0.0 to 1.0)
 * @d_logits:        accumulated gradient [seq_len × vocab]  (in/out)
 *
 * Returns the KD loss value.
 */
float distill_loss_and_grad(const Matrix *student_logits,
                             const float  *teacher_logits,
                             float temperature,
                             float alpha,
                             int   seq_len,
                             int   vocab_size,
                             Matrix *d_logits) {
    float T    = temperature;
    float T2   = T * T;
    float loss = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        const float *s_lg = student_logits->data + t * vocab_size;
        const float *t_lg = teacher_logits        + t * vocab_size;
        float       *d_lg = d_logits->data         + t * vocab_size;

        /* Compute soft student probs at temperature T */
        float s_max = s_lg[0];
        for (int j = 1; j < vocab_size; j++)
            if (s_lg[j] > s_max) s_max = s_lg[j];
        float s_sum = 0.0f;
        float *s_probs = (float *)malloc(vocab_size * sizeof(float));
        for (int j = 0; j < vocab_size; j++) {
            s_probs[j] = expf((s_lg[j] - s_max) / T);
            s_sum += s_probs[j];
        }
        for (int j = 0; j < vocab_size; j++) s_probs[j] /= s_sum;

        /* Compute soft teacher probs at temperature T */
        float t_max = t_lg[0];
        for (int j = 1; j < vocab_size; j++)
            if (t_lg[j] > t_max) t_max = t_lg[j];
        float t_sum = 0.0f;
        float *t_probs = (float *)malloc(vocab_size * sizeof(float));
        for (int j = 0; j < vocab_size; j++) {
            t_probs[j] = expf((t_lg[j] - t_max) / T);
            t_sum += t_probs[j];
        }
        for (int j = 0; j < vocab_size; j++) t_probs[j] /= t_sum;

        /* KL divergence contribution */
        for (int j = 0; j < vocab_size; j++)
            if (t_probs[j] > 1e-10f)
                loss += t_probs[j] * logf(t_probs[j] / (s_probs[j] + 1e-10f));

        /* Gradient: d_KL/d_s_logits[j] = T² * alpha * (s_probs[j] - t_probs[j])
         * The T² factor corrects for the temperature scaling in the loss. */
        for (int j = 0; j < vocab_size; j++)
            d_lg[j] += T2 * alpha * (s_probs[j] - t_probs[j])
                       / (float)seq_len;

        free(s_probs);
        free(t_probs);
    }
    return loss / (float)seq_len;
}

/* =========================================================================
 * Main training loop
 * ====================================================================== */

/**
 * train — run the full training loop.
 *
 * Prints loss every 10 steps, saves checkpoint every tcfg->save_every steps.
 */
void train(Model *m, const TrainConfig *tcfg) {
    DataLoader *train_dl = dataloader_init(tcfg->train_data_path);
    DataLoader *val_dl   = dataloader_init(tcfg->val_data_path);
    if (!train_dl) { fprintf(stderr, "Cannot open train data\n"); return; }
    if (!val_dl)   { fprintf(stderr, "Cannot open val data\n");   return; }

    AdamState *adam = adam_init(m);

    int bs      = tcfg->batch_size;
    int seq_len = tcfg->seq_len;

    int   *batch_in  = (int *)malloc((size_t)bs * seq_len * sizeof(int));
    int   *batch_tgt = (int *)malloc((size_t)bs * seq_len * sizeof(int));

    long tokens_per_step = (long)bs * seq_len;
    printf("[train] %d steps | batch=%d | seq=%d | %.1fM tok/epoch\n",
           tcfg->max_steps, bs, seq_len,
           (double)tokens_per_step * tcfg->max_steps / 1e6);
    printf("[train] model params: %ldM\n", model_param_count(m) / 1000000);
    printf("\n%-6s  %-8s  %-8s  %-8s  %-8s  %-8s\n",
           "step", "loss", "smooth", "val", "lr", "ms/step");
    printf("%s\n", "------------------------------------------------------");

    double total_loss   = 0.0;
    double smooth_loss  = -1.0;   /* EMA of loss, init on first step */
    double smooth_alpha = 0.05;   /* EMA coefficient: smaller = smoother */
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    struct timespec step_t0, step_t1;

    /* Pre-allocate validation logits scratch to avoid alloc in hot path */
    Matrix val_logits = mat_alloc(seq_len, m->cfg.vocab_size, 0);

    for (int step = 0; step < tcfg->max_steps; step++) {
        clock_gettime(CLOCK_MONOTONIC, &step_t0);

        float lr = lr_schedule(step,
                               tcfg->warmup_steps, tcfg->max_steps,
                               tcfg->learning_rate, tcfg->lr_min);

        model_zero_grads(m);

        float step_loss = 0.0f;
        for (int b = 0; b < bs; b++) {
            dataloader_next_batch(train_dl,
                                  batch_in  + b * seq_len,
                                  batch_tgt + b * seq_len,
                                  1, seq_len);
            step_loss += model_backward(m,
                                        batch_in  + b * seq_len,
                                        batch_tgt + b * seq_len,
                                        seq_len);
        }
        step_loss /= (float)bs;

        float grad_norm = clip_grad_norm(m, tcfg->grad_clip);
        adam_step(adam, m, tcfg, lr);

        total_loss += step_loss;

        /* Exponential moving average of loss — much less noisy than raw */
        if (smooth_loss < 0.0) smooth_loss = step_loss;
        else smooth_loss = smooth_alpha * step_loss + (1.0 - smooth_alpha) * smooth_loss;

        clock_gettime(CLOCK_MONOTONIC, &step_t1);
        double ms_step = (step_t1.tv_sec - step_t0.tv_sec) * 1000.0
                       + (step_t1.tv_nsec - step_t0.tv_nsec) / 1e6;

        /* Log every 10 steps */
        if (step % 10 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            double elapsed = (ts1.tv_sec - ts0.tv_sec)
                           + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
            (void)elapsed;
            printf("%-6d  %-8.4f  %-8.4f  %-8s  %-8.2e  %-6.0f  gnorm=%.1f\n",
                   step,
                   step_loss,
                   smooth_loss,
                   "---",
                   (double)lr,
                   ms_step,
                   (double)grad_norm);
            fflush(stdout);
        }

        /* Validation — actually compute cross-entropy loss */
        if (tcfg->eval_every > 0 && step % tcfg->eval_every == 0 && step > 0) {
            float val_loss = 0.0f;
            int   val_steps = 20;
            for (int vs = 0; vs < val_steps; vs++) {
                dataloader_next_batch(val_dl, batch_in, batch_tgt, 1, seq_len);
                model_forward(m, batch_in, seq_len, &val_logits);
                /* Compute cross-entropy from logits + targets */
                float vl = 0.0f;
                int V = m->cfg.vocab_size;
                for (int t = 0; t < seq_len; t++) {
                    float *row = val_logits.data + t * V;
                    int    tgt = batch_tgt[t];
                    float  vmax = row[0];
                    for (int j = 1; j < V; j++) if (row[j] > vmax) vmax = row[j];
                    float  s = 0.0f;
                    for (int j = 0; j < V; j++) s += expf(row[j] - vmax);
                    vl += -(row[tgt] - vmax) + logf(s);
                }
                val_loss += vl / (float)seq_len;
            }
            val_loss /= (float)val_steps;
            printf("  [val] step=%-6d  val_loss=%.4f  train_smooth=%.4f  gap=%.4f\n",
                   step, val_loss, (float)smooth_loss,
                   val_loss - (float)smooth_loss);
            fflush(stdout);
        }

        /* Checkpoint */
        if (tcfg->save_every > 0 && step % tcfg->save_every == 0 && step > 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/ckpt_%05d.bin",
                     tcfg->checkpoint_dir, step);
            if (model_save(m, path) == 0)
                printf("  [ckpt] saved: %s\n", path);
            else
                fprintf(stderr, "  [ckpt] FAILED: %s\n", path);
        }
    }

    mat_free(&val_logits);

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double total_sec = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec)*1e-9;
    long total_tokens = (long)tcfg->max_steps * tokens_per_step;
    printf("\n[train] done in %.1fs | avg_loss=%.4f | %.0f tok/s\n",
           total_sec,
           total_loss / (double)tcfg->max_steps,
           (double)total_tokens / total_sec);

    free(batch_in);
    free(batch_tgt);
    adam_free(adam);
    dataloader_free(train_dl);
    dataloader_free(val_dl);
}
