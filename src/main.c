/**
 * main.c
 *
 * Entry point for llm.c training.
 *
 * Reads data/meta.txt (written by prepare_data) to auto-configure the
 * model and training schedule based on actual corpus size.
 *
 * Usage:
 *   ./build/train                          — auto-configure from data/meta.txt
 *   ./build/train --tiny                   — force tiny config  (~1M params)
 *   ./build/train --small                  — force small config (~7M params)
 *   ./build/train --steps 5000             — override max_steps
 *   ./build/train --lr 5e-4                — override learning rate
 *   ./build/train --checkpoint ckpt.bin    — resume from checkpoint
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config.h"
#include "../include/transformer.h"
#include "../include/tokenizer.h"

void train(Model *m, const TrainConfig *tcfg);

/* ---------------------------------------------------------------------------
 * Read meta.txt — returns 0 on success
 * ------------------------------------------------------------------------- */
static int read_meta(const char *path, long *n_train, long *n_val,
                     int *vocab_size) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    *n_train = 0; *n_val = 0; *vocab_size = 0;
    while (fgets(line, sizeof(line), f)) {
        if      (sscanf(line, "train_tokens: %ld",  n_train)   == 1) {}
        else if (sscanf(line, "val_tokens:   %ld",  n_val)     == 1) {}
        else if (sscanf(line, "vocab_size:   %d",   vocab_size)== 1) {}
    }
    fclose(f);
    return (*n_train > 0 && *vocab_size > 0) ? 0 : -1;
}

/* ---------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    printf("llm.c — minimal LLM in pure C\n");
    printf("================================\n\n");

    /* Parse args */
    int         force_tiny  = 0;
    int         force_small = 0;
    int         override_steps = 0;
    float       override_lr    = 0.0f;
    const char *resume_path    = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--tiny"))  force_tiny  = 1;
        else if (!strcmp(argv[i], "--small")) force_small = 1;
        else if (!strcmp(argv[i], "--steps") && i+1 < argc)
            override_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc)
            override_lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--checkpoint") && i+1 < argc)
            resume_path = argv[++i];
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--tiny|--small] [--steps N] "
                    "[--lr F] [--checkpoint FILE]\n", argv[0]);
            return 1;
        }
    }

    /* ------------------------------------------------------------------ */
    /* Determine config                                                     */
    /* ------------------------------------------------------------------ */
    ModelConfig mcfg;
    TrainConfig tcfg;

    if (resume_path) {
        /* Resume: load model config from checkpoint */
        printf("[init] resuming from %s\n", resume_path);
        Model *tmp = model_load(resume_path);
        if (!tmp) {
            fprintf(stderr, "[error] could not load checkpoint\n");
            return 1;
        }
        mcfg = tmp->cfg;
        model_free(tmp);
        tcfg = train_config_default();
    } else if (force_tiny) {
        mcfg = model_config_tiny();
        tcfg = train_config_tiny();
        printf("[init] forced tiny config\n");
    } else if (force_small) {
        mcfg = model_config_small();
        tcfg = train_config_default();
        printf("[init] forced small config\n");
    } else {
        /* Auto-configure from meta.txt */
        long n_train = 0, n_val = 0;
        int  vocab_size = 0;
        if (read_meta("data/meta.txt", &n_train, &n_val, &vocab_size) == 0) {
            printf("[init] read data/meta.txt: %ld train tokens, "
                   "vocab=%d\n", n_train, vocab_size);
            config_auto(n_train, vocab_size, &mcfg, &tcfg);
        } else {
            printf("[init] no data/meta.txt — using tiny defaults\n"
                   "       Run prepare_data first for auto-configuration.\n\n");
            mcfg = model_config_tiny();
            tcfg = train_config_tiny();
        }
    }

    /* Apply CLI overrides */
    if (override_steps > 0) { tcfg.max_steps = override_steps;
                               printf("[init] override max_steps=%d\n", override_steps); }
    if (override_lr   > 0)  { tcfg.learning_rate = override_lr;
                               printf("[init] override lr=%.2e\n", (double)override_lr); }

    /* ------------------------------------------------------------------ */
    /* Print config summary                                                 */
    /* ------------------------------------------------------------------ */
    printf("\nModel config:\n");
    printf("  vocab     = %d\n",   mcfg.vocab_size);
    printf("  d_model   = %d\n",   mcfg.d_model);
    printf("  layers    = %d\n",   mcfg.n_layers);
    printf("  heads     = %d  (kv=%d, GQA ratio=%d:1)\n",
           mcfg.n_heads, mcfg.n_kv_heads,
           mcfg.n_kv_heads > 0 ? mcfg.n_heads / mcfg.n_kv_heads : 1);
    printf("  ffn       = %d\n",   mcfg.ffn_hidden);
    printf("  seq_len   = %d\n",   mcfg.max_seq_len);

    printf("\nTrain config:\n");
    printf("  steps     = %d\n",   tcfg.max_steps);
    printf("  batch     = %d\n",   tcfg.batch_size);
    printf("  seq_len   = %d\n",   tcfg.seq_len);
    printf("  lr        = %.2e → %.2e\n",
           (double)tcfg.learning_rate, (double)tcfg.lr_min);
    printf("  warmup    = %d steps\n", tcfg.warmup_steps);

    /* ------------------------------------------------------------------ */
    /* Check data exists                                                    */
    /* ------------------------------------------------------------------ */
    FILE *check = fopen(tcfg.train_data_path, "rb");
    if (!check) {
        printf("\n[error] Training data not found at '%s'.\n"
               "  Run the full pipeline first:\n"
               "    ./build/build_vocab  corpus.txt 2048 data/tokenizer.vocab\n"
               "    ./build/prepare_data corpus.txt data/tokenizer.vocab\n"
               "    ./build/train\n",
               tcfg.train_data_path);
        return 0;
    }
    fclose(check);

    /* ------------------------------------------------------------------ */
    /* Build model                                                          */
    /* ------------------------------------------------------------------ */
    printf("\n[init] allocating model...\n");
    Model *m;
    if (resume_path) {
        m = model_load(resume_path);
        if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    } else {
        m = model_init(&mcfg);
    }

    long params = model_param_count(m);
    printf("[init] parameters: %.2fM\n\n", (double)params / 1e6);

    /* Sanity: warn if corpus is very small relative to param count */
    {
        long n_train = 0, n_val = 0; int vs = 0;
        if (read_meta("data/meta.txt", &n_train, &n_val, &vs) == 0) {
            float ratio = (float)n_train / (float)params;
            if (ratio < 5.0f) {
                printf("[warning] only %.1f tokens per parameter "
                       "(corpus likely too small for generalisation).\n"
                       "          The model will mostly memorise the training text.\n"
                       "          Consider: --tiny flag, larger corpus, or smaller vocab.\n\n",
                       (double)ratio);
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Create checkpoint directory                                          */
    /* ------------------------------------------------------------------ */
    #include <sys/stat.h>
    mkdir(tcfg.checkpoint_dir, 0755);

    /* ------------------------------------------------------------------ */
    /* Train                                                                */
    /* ------------------------------------------------------------------ */
    train(m, &tcfg);

    model_free(m);
    return 0;
}
