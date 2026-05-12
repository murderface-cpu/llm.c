/**
 * generate.c
 *
 * Command-line interface for prompting a trained llm.c model.
 *
 * Usage:
 *   ./build/generate <checkpoint.bin> <tokenizer.vocab> [options]
 *
 * Options:
 *   --prompt    TEXT    Input prompt (default: interactive readline mode)
 *   --max_new   INT     Max tokens to generate      (default: 256)
 *   --temp      FLOAT   Sampling temperature         (default: 0.8)
 *   --top_k     INT     Top-k sampling cutoff        (default: 40)
 *   --top_p     FLOAT   Nucleus probability mass     (default: 0.9)
 *   --greedy            Use greedy (argmax) decoding (no randomness)
 *   --seed      INT     RNG seed for reproducibility (default: time-based)
 *   --bench             Print tokens/sec after generation
 *   --chat              Multi-turn chat loop (keeps KV cache between turns)
 *
 * Examples:
 *   # Single prompt
 *   ./build/generate checkpoints/ckpt_05000.bin data/tokenizer.vocab \
 *       --prompt "The attention mechanism works by"
 *
 *   # Greedy, reproducible
 *   ./build/generate checkpoints/ckpt_05000.bin data/tokenizer.vocab \
 *       --prompt "Once upon a time" --greedy --seed 42
 *
 *   # Interactive chat loop
 *   ./build/generate checkpoints/ckpt_05000.bin data/tokenizer.vocab --chat
 *
 *   # Test on random weights (no checkpoint needed)
 *   ./build/generate --random data/tokenizer.vocab \
 *       --prompt "Hello" --max_new 32
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../include/transformer.h"
#include "../include/inference.h"
#include "../include/tokenizer.h"
#include "../include/config.h"

/* =========================================================================
 * CLI config
 * ====================================================================== */

typedef struct {
    /* Required */
    const char *checkpoint;   /* path to .bin checkpoint, or "--random"     */
    const char *vocab;        /* path to .vocab file                        */

    /* Generation */
    const char *prompt;       /* NULL = interactive / chat mode             */
    int         max_new;
    float       temperature;
    int         top_k;
    float       top_p;
    int         greedy;
    unsigned    seed;

    /* Modes */
    int         bench;        /* print tok/s after generation               */
    int         chat;         /* multi-turn: keep KV cache between turns    */
} CLIConfig;

static CLIConfig default_cli(void) {
    return (CLIConfig){
        .checkpoint  = NULL,
        .vocab       = NULL,
        .prompt      = NULL,
        .max_new     = 256,
        .temperature = 0.8f,
        .top_k       = 40,
        .top_p       = 0.9f,
        .greedy      = 0,
        .seed        = 0,
        .bench       = 0,
        .chat        = 0,
    };
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s <checkpoint.bin | --random> <tokenizer.vocab> [options]\n"
        "\n"
        "Options:\n"
        "  --prompt TEXT    Input prompt text\n"
        "  --max_new INT    Max tokens to generate        (default: 256)\n"
        "  --temp    FLOAT  Sampling temperature           (default: 0.8)\n"
        "  --top_k   INT    Top-k cutoff; 0=disabled       (default: 40)\n"
        "  --top_p   FLOAT  Nucleus probability; 0=disabled(default: 0.9)\n"
        "  --greedy         Greedy (argmax) decoding\n"
        "  --seed    INT    RNG seed (0=time-based)         (default: 0)\n"
        "  --bench          Print tokens/sec after output\n"
        "  --chat           Interactive multi-turn chat loop\n"
        "\n"
        "Examples:\n"
        "  %s ckpt_05000.bin tokenizer.vocab "
        "--prompt \"The transformer\"\n"
        "  %s ckpt_05000.bin tokenizer.vocab --chat\n"
        "  %s --random       tokenizer.vocab "
        "--prompt \"test\" --max_new 20\n",
        argv0, argv0, argv0, argv0);
}

static int parse_args(int argc, char **argv, CLIConfig *cfg) {
    if (argc < 3) { print_usage(argv[0]); return -1; }

    cfg->checkpoint = argv[1];
    cfg->vocab      = argv[2];

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i], "--prompt")  && i+1 < argc) cfg->prompt      = argv[++i];
        else if (!strcmp(argv[i], "--max_new") && i+1 < argc) cfg->max_new     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp")    && i+1 < argc) cfg->temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top_k")   && i+1 < argc) cfg->top_k       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top_p")   && i+1 < argc) cfg->top_p       = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed")    && i+1 < argc) cfg->seed        = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--greedy"))  cfg->greedy = 1;
        else if (!strcmp(argv[i], "--bench"))   cfg->bench  = 1;
        else if (!strcmp(argv[i], "--chat"))    cfg->chat   = 1;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

/* =========================================================================
 * Streaming callback — writes each token to stdout as it is produced
 * ====================================================================== */

typedef struct {
    int   n_tokens;
    double t_start;
} StreamCtx;

static int stream_cb(int token_id, const char *text, void *userdata) {
    (void)token_id;
    StreamCtx *ctx = (StreamCtx *)userdata;
    ctx->n_tokens++;
    fputs(text, stdout);
    fflush(stdout);
    return 0;   /* 0 = keep going */
}

/* =========================================================================
 * Single generation run
 * ====================================================================== */

static void run_generation(const Model *m, const Tokenizer *tok,
                            InferenceState *state,
                            const CLIConfig *cli,
                            const char *prompt,
                            int reset_state) {
    if (reset_state) inference_reset(state);

    /* Build GenerateConfig from CLI flags */
    GenerateConfig gcfg = generate_config_default();
    gcfg.max_new_tokens = cli->max_new;

    if (cli->greedy) {
        gcfg.sampler.temperature = 0.0f;   /* 0 = greedy in sample_token() */
        gcfg.sampler.top_k       = 0;
        gcfg.sampler.top_p       = 0.0f;
    } else {
        gcfg.sampler.temperature = cli->temperature;
        gcfg.sampler.top_k       = cli->top_k;
        gcfg.sampler.top_p       = cli->top_p;
    }
    gcfg.sampler.seed = cli->seed != 0 ? cli->seed : (unsigned)time(NULL);

    /* Stop at EOS token */
    int eos = 2;   /* TOK_EOS */
    gcfg.stop_tokens   = &eos;
    gcfg.n_stop_tokens = 1;

    /* Streaming */
    StreamCtx ctx = { .n_tokens = 0, .t_start = 0.0 };
    gcfg.on_token  = stream_cb;
    gcfg.user_data = &ctx;

    /* Print prompt, then stream the model's continuation */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ctx.t_start = ts.tv_sec + ts.tv_nsec * 1e-9;

    char *output = generate(m, tok, state, prompt, &gcfg);

    clock_gettime(CLOCK_MONOTONIC, &ts);
    double elapsed = ts.tv_sec + ts.tv_nsec * 1e-9 - ctx.t_start;

    printf("\n");

    if (cli->bench && elapsed > 0.0 && ctx.n_tokens > 0) {
        fprintf(stderr,
            "[bench] %d tokens in %.2fs = %.1f tok/s\n",
            ctx.n_tokens, elapsed, ctx.n_tokens / elapsed);
    }

    free(output);
}

/* =========================================================================
 * Interactive single-prompt mode
 * ====================================================================== */

static void mode_single(const Model *m, const Tokenizer *tok,
                         InferenceState *state, const CLIConfig *cli) {
    printf("%s", cli->prompt);
    fflush(stdout);
    run_generation(m, tok, state, cli, cli->prompt, /*reset=*/1);
}

/* =========================================================================
 * Interactive chat loop
 *
 * Keeps the KV cache across turns so the model can reference earlier
 * turns in its context window.  Prints a ">" prompt, reads a line,
 * feeds it through the model, streams the reply.
 *
 * Type "quit" or "exit" or press Ctrl-D to end the session.
 * Type "/reset" to clear the KV cache and start fresh.
 * Type "/info"  to print current context position.
 * ====================================================================== */

static void mode_chat(const Model *m, const Tokenizer *tok,
                       InferenceState *state, const CLIConfig *cli) {
    char input_buf[4096];
    int  turn = 0;

    printf("llm.c interactive chat  "
           "(type 'exit' to quit, '/reset' to clear context)\n");
    printf("Model: %ld M params | context: %d tokens\n\n",
           model_param_count(m) / 1000000, m->cfg.max_seq_len);

    inference_reset(state);

    while (1) {
        /* Print turn indicator and context usage */
        int remaining = m->cfg.max_seq_len - state->seq_pos;
        printf("[%d | %d tokens left] > ", turn + 1, remaining);
        fflush(stdout);

        if (!fgets(input_buf, sizeof(input_buf), stdin)) {
            printf("\n");
            break;   /* EOF (Ctrl-D) */
        }

        /* Strip trailing newline */
        int len = (int)strlen(input_buf);
        while (len > 0 && (input_buf[len-1] == '\n' || input_buf[len-1] == '\r'))
            input_buf[--len] = '\0';

        if (len == 0) continue;

        /* Built-in commands */
        if (!strcmp(input_buf, "exit") || !strcmp(input_buf, "quit")) break;

        if (!strcmp(input_buf, "/reset")) {
            inference_reset(state);
            turn = 0;
            printf("[context cleared]\n\n");
            continue;
        }

        if (!strcmp(input_buf, "/info")) {
            printf("[seq_pos=%d / %d  |  %.1f%% full]\n\n",
                   state->seq_pos, m->cfg.max_seq_len,
                   100.0 * state->seq_pos / m->cfg.max_seq_len);
            continue;
        }

        /* Context overflow guard */
        if (remaining < 32) {
            printf("[context nearly full — type /reset to clear]\n");
        }

        /* Generate reply — DO NOT reset state between turns */
        printf("Assistant: ");
        fflush(stdout);
        run_generation(m, tok, state, cli, input_buf, /*reset=*/0);
        printf("\n");
        turn++;
    }

    printf("Session ended. %d turns, %d total tokens used.\n",
           turn, state->seq_pos);
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char **argv) {
    CLIConfig cli = default_cli();
    if (parse_args(argc, argv, &cli) < 0) return 1;

    /* ------------------------------------------------------------------ */
    /* Load model                                                           */
    /* ------------------------------------------------------------------ */
    Model *m = NULL;

    if (!strcmp(cli.checkpoint, "--random")) {
        /* Synthesise a random model — useful for testing the pipeline
         * without a trained checkpoint.                                    */
        fprintf(stderr, "[info] using random (untrained) model weights\n");
        ModelConfig mcfg = model_config_default();
        /* Keep vocab small so it might match the real tokenizer size      */
        mcfg.vocab_size = 4096;
        m = model_init(&mcfg);
    } else {
        m = model_load(cli.checkpoint);
        if (!m) {
            fprintf(stderr,
                "[error] could not load checkpoint: %s\n"
                "        (use --random to test without a checkpoint)\n",
                cli.checkpoint);
            return 1;
        }
        fprintf(stderr, "[info] loaded %s  (%ld M params)\n",
                cli.checkpoint, model_param_count(m) / 1000000);
    }

    /* ------------------------------------------------------------------ */
    /* Load tokenizer                                                       */
    /* ------------------------------------------------------------------ */
    Tokenizer *tok = tok_load(cli.vocab);
    if (!tok) {
        fprintf(stderr, "[error] could not load tokenizer: %s\n", cli.vocab);
        model_free(m);
        return 1;
    }
    fprintf(stderr, "[info] tokenizer vocab size: %d\n\n",
            tok_vocab_size(tok));

    /* Warn if vocab sizes mismatch */
    if (tok_vocab_size(tok) != m->cfg.vocab_size) {
        fprintf(stderr,
            "[warning] tokenizer vocab (%d) != model vocab (%d)\n"
            "          Output may be garbage — make sure the tokenizer\n"
            "          matches the checkpoint.\n\n",
            tok_vocab_size(tok), m->cfg.vocab_size);
    }

    /* ------------------------------------------------------------------ */
    /* Initialise inference state                                           */
    /* ------------------------------------------------------------------ */
    InferenceState *state = inference_init(m);

    /* ------------------------------------------------------------------ */
    /* Run the requested mode                                               */
    /* ------------------------------------------------------------------ */
    if (cli.chat) {
        mode_chat(m, tok, state, &cli);
    } else if (cli.prompt) {
        mode_single(m, tok, state, &cli);
    } else {
        /* No prompt and no --chat: print usage hint */
        fprintf(stderr,
            "No --prompt given.  Options:\n"
            "  --prompt \"text\"   single generation\n"
            "  --chat            interactive multi-turn mode\n");
        inference_free(state);
        tok_free(tok);
        model_free(m);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Cleanup                                                              */
    /* ------------------------------------------------------------------ */
    inference_free(state);
    tok_free(tok);
    model_free(m);
    return 0;
}
