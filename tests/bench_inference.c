/**
 * tests/bench_inference.c
 *
 * Benchmarks the inference engine end-to-end.
 *
 * Measures:
 *   - Prefill throughput   (tokens/s processing the prompt)
 *   - Decode  throughput   (tokens/s generating new tokens)
 *   - Per-layer breakdown  (where time is spent)
 *   - KV cache memory      (bytes used)
 *
 * Usage:
 *   ./build/bench_inference [checkpoint.bin] [tokenizer.vocab] [prompt]
 *
 * If no checkpoint is given, synthesises a random model with default config
 * so you can benchmark without a trained model.
 *
 * Compile:
 *   make bench_inference
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "../include/transformer.h"
#include "../include/inference.h"
#include "../include/tokenizer.h"
#include "../include/config.h"

/* =========================================================================
 * Timing
 * ====================================================================== */

static double wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* =========================================================================
 * Streaming callback — prints each token as it is generated
 * ====================================================================== */

static int stream_cb(int token_id, const char *text, void *userdata) {
    (void)token_id; (void)userdata;
    printf("%s", text);
    fflush(stdout);
    return 0;   /* 0 = continue */
}

/* =========================================================================
 * KV cache memory report
 * ====================================================================== */

static void print_kv_cache_stats(const Model *m) {
    const ModelConfig *cfg = &m->cfg;
    long kv_dim     = cfg->n_kv_heads * cfg->head_dim;
    long bytes_layer= 2L * cfg->max_seq_len * kv_dim * sizeof(float);
    long bytes_total= bytes_layer * cfg->n_layers;

    printf("KV cache:\n");
    printf("  per layer:  %ld KB  (%d positions × %ld floats × 2 K/V)\n",
           bytes_layer / 1024,
           cfg->max_seq_len, kv_dim);
    printf("  all layers: %ld KB  (%d layers)\n",
           bytes_total / 1024, cfg->n_layers);
    printf("  at max_seq_len=%d:  %.1f MB\n\n",
           cfg->max_seq_len, (double)bytes_total / 1e6);
}

/* =========================================================================
 * Prefill benchmark
 *
 * Runs forward pass over `n_tokens` tokens and measures throughput.
 * ====================================================================== */

static void bench_prefill(Model *m, InferenceState *state,
                           int n_tokens, int n_runs) {
    /* Fabricate a simple token sequence */
    int *tokens = (int *)malloc(n_tokens * sizeof(int));
    for (int i = 0; i < n_tokens; i++)
        tokens[i] = (i % (m->cfg.vocab_size - 4)) + 4;

    double best_tps = 0.0;
    double total_time = 0.0;

    for (int run = 0; run < n_runs; run++) {
        inference_reset(state);

        double t0 = wall_sec();
        for (int t = 0; t < n_tokens; t++)
            inference_step(m, state, tokens[t]);
        double t1 = wall_sec();

        double tps = n_tokens / (t1 - t0);
        total_time += t1 - t0;
        if (tps > best_tps) best_tps = tps;
    }

    double avg_tps = (n_tokens * n_runs) / total_time;
    printf("  Prefill  %4d tokens × %d runs | best %7.1f tok/s | "
           "avg %7.1f tok/s | %.2f ms/tok\n",
           n_tokens, n_runs, best_tps, avg_tps,
           1000.0 / avg_tps);
    free(tokens);
}

/* =========================================================================
 * Decode benchmark
 *
 * Runs single-token generation for `n_tokens` steps.
 * ====================================================================== */

static void bench_decode(Model *m, InferenceState *state,
                          int n_tokens, int n_runs) {
    double best_tps = 0.0;
    double total_time = 0.0;

    for (int run = 0; run < n_runs; run++) {
        inference_reset(state);

        /* Seed with BOS */
        inference_step(m, state, 1 /* TOK_BOS */);

        int next = 4;   /* first real token */

        double t0 = wall_sec();
        for (int t = 0; t < n_tokens; t++) {
            inference_step(m, state, next);
            /* Greedy next token from logits (avoids alloc inside loop) */
            next = sample_argmax(state->logits, m->cfg.vocab_size);
        }
        double t1 = wall_sec();

        double tps = n_tokens / (t1 - t0);
        total_time += t1 - t0;
        if (tps > best_tps) best_tps = tps;
    }

    double avg_tps = (n_tokens * n_runs) / total_time;
    printf("  Decode   %4d tokens × %d runs | best %7.1f tok/s | "
           "avg %7.1f tok/s | %.2f ms/tok\n",
           n_tokens, n_runs, best_tps, avg_tps,
           1000.0 / avg_tps);
}

/* =========================================================================
 * Per-step timing breakdown
 *
 * Measures how long a single decode step takes, split by:
 *   - Embedding lookup
 *   - Attention (QKV projection + score + weighted sum + output proj)
 *   - FFN
 *   - Final norm + LM head
 * ====================================================================== */

static void bench_step_breakdown(Model *m, InferenceState *state) {
    const ModelConfig *cfg = &m->cfg;
    int ITERS = 200;

    inference_reset(state);
    inference_step(m, state, 1);  /* prime cache at pos=0 */

    double t_total = 0.0;
    double t_start = wall_sec();

    for (int i = 0; i < ITERS; i++) {
        /* Use pos=1 repeatedly by resetting seq_pos after each step
         * so the KV cache lookup always covers 2 positions — small
         * and consistent, isolating compute rather than memory bandwidth. */
        state->seq_pos = 1;
        inference_step(m, state, 4);
    }

    t_total = wall_sec() - t_start;
    double ms_per_step = t_total * 1000.0 / ITERS;

    printf("\nSingle decode step (seq_pos=1, %d iters):\n", ITERS);
    printf("  Total:   %.3f ms/step  →  %.1f tok/s\n",
           ms_per_step, 1000.0 / ms_per_step);

    /* Estimate parameter counts and theoretical FLOP breakdown */
    int d    = cfg->d_model;
    int nh   = cfg->n_heads;
    int nkv  = cfg->n_kv_heads;
    int hd   = cfg->head_dim;
    int ffnh = cfg->ffn_hidden;
    int V    = cfg->vocab_size;

    /* FLOPs per layer per token (matvec = 2*M*N) */
    long attn_proj = 2L * d * (nh + 2*nkv) * hd;  /* Q + K + V */
    long attn_out  = 2L * nh * hd * d;
    long ffn       = 2L * d * ffnh * 3;             /* gate + up + down */
    long per_layer = attn_proj + attn_out + ffn;
    long lm_head   = 2L * d * V;
    long total_flops= (long)cfg->n_layers * per_layer + lm_head;

    printf("  Estimated FLOPs/step: %.2f M\n",
           (double)total_flops / 1e6);
    printf("  GFLOP/s (achieved):   %.2f\n\n",
           (double)total_flops / (ms_per_step * 1e6));
}

/* =========================================================================
 * Throughput vs sequence length
 *
 * Shows how decode speed degrades as the KV cache grows.
 * This is the key characteristic that separates inference from training.
 * ====================================================================== */

static void bench_vs_seq_len(Model *m, InferenceState *state) {
    int lens[]  = {1, 8, 32, 64, 128, 256, 512};
    int n_lens  = (int)(sizeof(lens) / sizeof(lens[0]));
    int STEPS   = 50;

    printf("Decode throughput vs KV cache depth:\n");
    printf("  %-8s  %-12s  %-10s\n", "seq_len", "tok/s", "ms/tok");

    for (int li = 0; li < n_lens; li++) {
        int target_pos = lens[li];
        if (target_pos >= m->cfg.max_seq_len) continue;

        /* Fill cache to target_pos */
        inference_reset(state);
        for (int p = 0; p < target_pos; p++)
            inference_step(m, state, (p % (m->cfg.vocab_size - 4)) + 4);

        /* Now measure STEPS decode steps at this cache depth */
        int saved_pos = state->seq_pos;
        double t0 = wall_sec();

        for (int s = 0; s < STEPS; s++) {
            state->seq_pos = saved_pos;   /* reset position, keep cache */
            inference_step(m, state, 4);
        }

        double elapsed = wall_sec() - t0;
        double tps = STEPS / elapsed;

        printf("  %-8d  %-12.1f  %-10.3f\n",
               target_pos, tps, 1000.0 / tps);
    }
    printf("\n");
}

/* =========================================================================
 * Text generation demo
 * ====================================================================== */

static void demo_generate(Model *m, const Tokenizer *tok,
                           const char *prompt) {
    InferenceState *state = inference_init(m);
    inference_reset(state);

    GenerateConfig gcfg = generate_config_default();
    gcfg.max_new_tokens = 80;
    gcfg.sampler.temperature = 0.8f;
    gcfg.sampler.top_k  = 40;
    gcfg.sampler.top_p  = 0.9f;
    gcfg.on_token = stream_cb;

    int eos_id = 2;   /* TOK_EOS */
    gcfg.stop_tokens   = &eos_id;
    gcfg.n_stop_tokens = 1;

    printf("Prompt: \"%s\"\n", prompt);
    printf("Output: ");
    fflush(stdout);

    double t0 = wall_sec();
    char *out = generate(m, tok, state, prompt, &gcfg);
    double elapsed = wall_sec() - t0;

    printf("\n\n");
    if (out) {
        int n_toks = (int)(strlen(out) / 4 + 1);  /* rough estimate */
        printf("Generated ~%d tokens in %.2fs (%.1f tok/s)\n",
               n_toks, elapsed, n_toks / elapsed);
        free(out);
    }

    inference_free(state);
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char **argv) {
    printf("=== llm.c inference benchmark ===\n\n");

    /* ------------------------------------------------------------------ */
    /* Load or synthesise model                                             */
    /* ------------------------------------------------------------------ */
    Model *m = NULL;

    if (argc >= 2 && strcmp(argv[1], "--random") != 0) {
        printf("Loading model from %s...\n", argv[1]);
        m = model_load(argv[1]);
        if (!m) {
            fprintf(stderr, "Failed to load model. "
                    "Use --random for a synthetic benchmark.\n");
            return 1;
        }
    } else {
        printf("No checkpoint given — synthesising random 50M model...\n");
        ModelConfig cfg = model_config_default();
        /* Use a smaller vocab for speed in the benchmark */
        cfg.vocab_size = 4096;
        m = model_init(&cfg);
    }

    printf("Model: %ld M parameters\n", model_param_count(m) / 1000000);
    printf("Config: d=%d, layers=%d, heads=%d, kv_heads=%d, "
           "ffn=%d, vocab=%d\n\n",
           m->cfg.d_model, m->cfg.n_layers, m->cfg.n_heads,
           m->cfg.n_kv_heads, m->cfg.ffn_hidden, m->cfg.vocab_size);

    /* ------------------------------------------------------------------ */
    /* KV cache stats                                                       */
    /* ------------------------------------------------------------------ */
    print_kv_cache_stats(m);

    /* ------------------------------------------------------------------ */
    /* Core benchmarks                                                      */
    /* ------------------------------------------------------------------ */
    InferenceState *state = inference_init(m);

    printf("=== Throughput benchmarks ===\n");
    bench_prefill(m, state, 16,  5);
    bench_prefill(m, state, 64,  3);
    bench_prefill(m, state, 128, 3);
    printf("\n");
    bench_decode(m, state, 32,  5);
    bench_decode(m, state, 64,  3);
    bench_decode(m, state, 128, 2);

    bench_step_breakdown(m, state);
    bench_vs_seq_len(m, state);

    inference_free(state);

    /* ------------------------------------------------------------------ */
    /* Text generation demo (only if tokenizer provided)                   */
    /* ------------------------------------------------------------------ */
    if (argc >= 3) {
        printf("=== Text generation demo ===\n");
        Tokenizer *tok = tok_load(argv[2]);
        if (tok) {
            const char *prompt = (argc >= 4) ? argv[3]
                               : "The attention mechanism";
            InferenceState *gen_state = inference_init(m);
            demo_generate(m, tok, prompt);
            inference_free(gen_state);
            tok_free(tok);
        } else {
            fprintf(stderr, "Could not load tokenizer from %s\n", argv[2]);
        }
    } else {
        printf("(Pass a tokenizer.vocab as second arg to run text generation)\n");
    }

    model_free(m);
    return 0;
}
