/**
 * tools/prepare_data.c
 *
 * Tokenise a raw UTF-8 text corpus into binary token files for training.
 *
 * Usage:
 *   ./build/prepare_data <corpus.txt> <tokenizer.vocab> [options]
 *
 * Options:
 *   --train_split  0.9       fraction of data for training (default 0.9)
 *   --out_dir      data/     output directory (default data/)
 *   --chunk_size   2048      tokens per output chunk (default 2048)
 *   --add_bos      1         prepend <bos> to each document (default 1)
 *   --add_eos      1         append  <eos> to each document (default 1)
 *
 * Output files:
 *   data/train.bin   — binary file of int32 token ids for training
 *   data/val.bin     — binary file of int32 token ids for validation
 *   data/meta.txt    — human-readable summary (token counts, vocab size)
 *
 * Binary file format (matches DataLoader in train.c):
 *   [int32: n_tokens]
 *   [int32 × n_tokens: token id sequence]
 *
 * Document handling:
 *   The corpus is treated as a sequence of documents separated by blank
 *   lines.  Each document is encoded independently (BOS+EOS added per
 *   document) then concatenated.  This is the standard "packed" format
 *   for causal language model training.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>  /* mkdir */

/* We link against tokenizer.c directly so we can call tok_load/tok_encode */
#include "../include/tokenizer.h"

/* =========================================================================
 * Config
 * ====================================================================== */

typedef struct {
    const char *corpus_path;
    const char *vocab_path;
    const char *out_dir;
    float       train_split;
    int         chunk_size;
    int         add_bos;
    int         add_eos;
} PrepConfig;

static PrepConfig default_config(void) {
    return (PrepConfig){
        .corpus_path = NULL,
        .vocab_path  = NULL,
        .out_dir     = "data",
        .train_split = 0.9f,
        .chunk_size  = 2048,
        .add_bos     = 1,
        .add_eos     = 1,
    };
}


/* =========================================================================
 * Argument parser
 * ====================================================================== */

static int parse_args(int argc, char **argv, PrepConfig *cfg) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <corpus.txt> <tokenizer.vocab> [options]\n"
            "\n"
            "Options:\n"
            "  --train_split FLOAT   fraction for training  (default 0.9)\n"
            "  --out_dir     PATH    output directory       (default data/)\n"
            "  --chunk_size  INT     tokens per chunk       (default 2048)\n"
            "  --add_bos     0|1     prepend <bos>          (default 1)\n"
            "  --add_eos     0|1     append  <eos>          (default 1)\n",
            argv[0]);
        return -1;
    }

    cfg->corpus_path = argv[1];
    cfg->vocab_path  = argv[2];

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--train_split") == 0 && i + 1 < argc)
            cfg->train_split = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--out_dir") == 0 && i + 1 < argc)
            cfg->out_dir = argv[++i];
        else if (strcmp(argv[i], "--chunk_size") == 0 && i + 1 < argc)
            cfg->chunk_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--add_bos") == 0 && i + 1 < argc)
            cfg->add_bos = atoi(argv[++i]);
        else if (strcmp(argv[i], "--add_eos") == 0 && i + 1 < argc)
            cfg->add_eos = atoi(argv[++i]);
        else {
            fprintf(stderr, "[error] unknown argument: %s\n", argv[i]);
            return -1;
        }
    }

    if (cfg->train_split <= 0.0f || cfg->train_split >= 1.0f) {
        fprintf(stderr, "[error] train_split must be in (0, 1)\n");
        return -1;
    }
    return 0;
}


/* =========================================================================
 * Token buffer — a growable array of int32 token ids
 * ====================================================================== */

typedef struct {
    int32_t *data;
    long     size;
    long     capacity;
} TokenBuf;

static TokenBuf *tokbuf_alloc(long cap) {
    TokenBuf *b = (TokenBuf *)malloc(sizeof(TokenBuf));
    b->size     = 0;
    b->capacity = cap;
    b->data     = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    return b;
}

static void tokbuf_free(TokenBuf *b) {
    free(b->data);
    free(b);
}

static void tokbuf_append(TokenBuf *b, int id) {
    if (b->size >= b->capacity) {
        b->capacity = b->capacity * 2 + 1;
        b->data = (int32_t *)realloc(b->data,
                      (size_t)b->capacity * sizeof(int32_t));
    }
    b->data[b->size++] = (int32_t)id;
}

static void tokbuf_append_array(TokenBuf *b, const int *ids, int n) {
    for (int i = 0; i < n; i++) tokbuf_append(b, ids[i]);
}


/* =========================================================================
 * Write a TokenBuf to a binary file
 *
 * Format:
 *   [int32: n_tokens]
 *   [int32 × n_tokens: token ids]
 * ====================================================================== */

static int write_bin(const char *path, const TokenBuf *buf) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    int32_t n = (int32_t)buf->size;
    fwrite(&n, sizeof(int32_t), 1, f);
    fwrite(buf->data, sizeof(int32_t), (size_t)buf->size, f);
    fclose(f);

    printf("[write] %s: %ld tokens (%.1f MB)\n",
           path, buf->size,
           (double)buf->size * sizeof(int32_t) / 1e6);
    return 0;
}


/* =========================================================================
 * Document splitter
 *
 * Splits the corpus into documents at blank lines (double newline).
 * Returns a heap-allocated array of heap-allocated strings.
 * Sets *n_docs to the document count.
 *
 * Each document string is null-terminated and owned by the caller;
 * free with:
 *   for (int i = 0; i < n_docs; i++) free(docs[i]);
 *   free(docs);
 * ====================================================================== */

static char **split_documents(const char *corpus, long corpus_len,
                               long *n_docs) {
    /* Count documents: each blank line starts a new one */
    long count = 1;
    for (long i = 0; i + 1 < corpus_len; i++) {
        if (corpus[i] == '\n' && corpus[i + 1] == '\n')
            count++;
    }

    char **docs = (char **)malloc((size_t)count * sizeof(char *));
    long   doc_idx = 0;
    long   start   = 0;

    for (long i = 0; i <= corpus_len; i++) {
        /* Document boundary: double newline, or end of file */
        int boundary = (i == corpus_len) ||
                       (i + 1 < corpus_len &&
                        corpus[i] == '\n' && corpus[i + 1] == '\n');

        if (boundary) {
            long doc_len = i - start;
            /* Skip empty documents */
            if (doc_len > 0) {
                char *doc = (char *)malloc((size_t)doc_len + 1);
                memcpy(doc, corpus + start, doc_len);
                doc[doc_len] = '\0';
                docs[doc_idx++] = doc;
            }
            start = i + 2;  /* skip the two newlines */
            i++;             /* outer loop will increment again */
        }
    }

    *n_docs = doc_idx;
    return docs;
}


/* =========================================================================
 * Main encode loop
 * ====================================================================== */

/**
 * encode_corpus — tokenise all documents and split into train/val buffers.
 *
 * The split happens at the document level (not token level) to prevent
 * a document appearing partially in both splits.
 *
 * Progress is printed every 5% of documents.
 */
static void encode_corpus(const char **docs, long n_docs,
                           const Tokenizer *tok,
                           const PrepConfig *cfg,
                           TokenBuf *train_buf, TokenBuf *val_buf) {
    /* Determine the train/val boundary document index */
    long train_cutoff = (long)(n_docs * cfg->train_split);

    /* Scratch buffer: worst case every byte is its own token + BOS + EOS */
    /* We'll use a large fixed buffer and resize if needed */
    int   scratch_cap = cfg->chunk_size * 4 + 4;
    int  *scratch     = (int *)malloc((size_t)scratch_cap * sizeof(int));

    long total_docs_encoded = 0;
    long total_tokens       = 0;
    long report_every       = n_docs / 20 + 1;  /* report every ~5% */

    clock_t t0 = clock();

    for (long d = 0; d < n_docs; d++) {
        const char *doc = docs[d];
        int doc_len     = (int)strlen(doc);

        /* Grow scratch if needed — safe upper bound: one token per byte */
        if (doc_len + 4 > scratch_cap) {
            scratch_cap = doc_len * 2 + 4;
            scratch = (int *)realloc(scratch, (size_t)scratch_cap * sizeof(int));
        }

        /* Encode the document */
        int n_toks = 0;
        if (tok_encode(tok, doc, cfg->add_bos, cfg->add_eos,
                       scratch, &n_toks) < 0) {
            fprintf(stderr, "[encode] failed on document %ld; skipping\n", d);
            continue;
        }

        /* Append to the appropriate buffer */
        TokenBuf *dest = (d < train_cutoff) ? train_buf : val_buf;
        tokbuf_append_array(dest, scratch, n_toks);

        total_tokens += n_toks;
        total_docs_encoded++;

        /* Progress */
        if (d % report_every == 0) {
            double pct     = 100.0 * (double)d / (double)n_docs;
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("[encode] %.0f%%  doc %ld/%ld  |  %ld tokens  |  %.1fs\r",
                   pct, d, n_docs, total_tokens, elapsed);
            fflush(stdout);
        }
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\n[encode] done: %ld docs, %ld tokens in %.1fs  "
           "(%.0f tok/s)\n",
           total_docs_encoded, total_tokens, elapsed,
           (double)total_tokens / (elapsed + 1e-9));

    free(scratch);
}


/* =========================================================================
 * Write meta.txt — human-readable summary
 * ====================================================================== */

static void write_meta(const char *out_dir,
                       const PrepConfig *cfg,
                       const Tokenizer *tok,
                       long n_train, long n_val,
                       long n_docs) {
    char path[512];
    snprintf(path, sizeof(path), "%s/meta.txt", out_dir);
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }

    fprintf(f, "# llm.c data preparation summary\n");
    fprintf(f, "corpus:       %s\n", cfg->corpus_path);
    fprintf(f, "vocab_file:   %s\n", cfg->vocab_path);
    fprintf(f, "vocab_size:   %d\n", tok_vocab_size(tok));
    fprintf(f, "n_documents:  %ld\n", n_docs);
    fprintf(f, "train_split:  %.2f\n", (double)cfg->train_split);
    fprintf(f, "add_bos:      %d\n", cfg->add_bos);
    fprintf(f, "add_eos:      %d\n", cfg->add_eos);
    fprintf(f, "\n");
    fprintf(f, "train_tokens: %ld  (%.2f M)\n",
            n_train, (double)n_train / 1e6);
    fprintf(f, "val_tokens:   %ld  (%.2f M)\n",
            n_val, (double)n_val / 1e6);
    fprintf(f, "total_tokens: %ld  (%.2f M)\n",
            n_train + n_val, (double)(n_train + n_val) / 1e6);
    fprintf(f, "\n");
    fprintf(f, "# Approx training epochs before data exhaustion\n");
    fprintf(f, "# (batch_size=8, seq_len=512 → 4096 tokens/step)\n");
    long tokens_per_step = 8 * 512;
    fprintf(f, "steps_per_epoch: %ld\n", n_train / tokens_per_step + 1);
    fclose(f);

    printf("[meta] wrote %s\n", path);
}


/* =========================================================================
 * Validate output — quick sanity check on the written files
 * ====================================================================== */

static void validate_bin(const char *path, const Tokenizer *tok) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[validate] cannot open %s\n", path); return; }

    int32_t n_tokens = 0;
    if (fread(&n_tokens, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "[validate] bad header in %s\n", path);
        fclose(f); return;
    }

    /* Read and check the first 32 tokens */
    int32_t sample[32];
    int     n_sample = n_tokens < 32 ? (int)n_tokens : 32;
    if (fread(sample, sizeof(int32_t), n_sample, f) != (size_t)n_sample) {
        fprintf(stderr, "[validate] short read in %s\n", path);
        fclose(f); return;
    }
    fclose(f);

    int vocab_sz = tok_vocab_size(tok);
    int bad = 0;
    for (int i = 0; i < n_sample; i++) {
        if (sample[i] < 0 || sample[i] >= vocab_sz) {
            fprintf(stderr, "[validate] out-of-range token %d at pos %d\n",
                    sample[i], i);
            bad++;
        }
    }

    /* Decode and print first few tokens for a visual sanity check */
    printf("[validate] %s: %d tokens — first 16: ", path, n_tokens);
    for (int i = 0; i < (n_sample < 16 ? n_sample : 16); i++) {
        const char *txt = tok_decode_token(tok, sample[i]);
        /* Escape newlines in the printout */
        if (strcmp(txt, "\n") == 0) printf("\\n ");
        else                        printf("'%s' ", txt);
    }
    printf("\n");

    if (bad == 0)
        printf("[validate] %s: OK\n", path);
    else
        printf("[validate] %s: %d BAD TOKENS\n", path, bad);
}


/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char **argv) {
    PrepConfig cfg = default_config();
    if (parse_args(argc, argv, &cfg) < 0)
        return 1;

    /* --- Create output directory --------------------------------------- */
#ifdef _WIN32
    _mkdir(cfg.out_dir);
#else
    mkdir(cfg.out_dir, 0755);
#endif

    /* --- Load tokenizer ------------------------------------------------ */
    printf("[init] loading tokenizer from %s...\n", cfg.vocab_path);
    Tokenizer *tok = tok_load(cfg.vocab_path);
    if (!tok) {
        fprintf(stderr, "[error] failed to load tokenizer\n");
        return 1;
    }
    printf("[init] vocab size: %d\n", tok_vocab_size(tok));

    /* --- Load corpus --------------------------------------------------- */
    printf("[init] loading corpus from %s...\n", cfg.corpus_path);
    FILE *f = fopen(cfg.corpus_path, "rb");
    if (!f) { perror(cfg.corpus_path); tok_free(tok); return 1; }

    fseek(f, 0, SEEK_END);
    long corpus_len = ftell(f);
    rewind(f);

    char *corpus_raw = (char *)malloc((size_t)corpus_len + 1);
    if (!corpus_raw || (long)fread(corpus_raw, 1, corpus_len, f) != corpus_len) {
        fprintf(stderr, "[error] failed to read corpus\n");
        fclose(f); tok_free(tok); return 1;
    }
    corpus_raw[corpus_len] = '\0';
    fclose(f);

    printf("[init] corpus: %.2f MB\n", (double)corpus_len / 1e6);

    /* --- Split into documents ----------------------------------------- */
    long   n_docs = 0;
    char **docs   = split_documents(corpus_raw, corpus_len, &n_docs);
    free(corpus_raw);   /* done with raw text */
    printf("[init] found %ld documents\n", n_docs);

    if (n_docs == 0) {
        fprintf(stderr, "[error] no documents found in corpus\n");
        tok_free(tok); return 1;
    }

    /* --- Encode -------------------------------------------------------- */
    TokenBuf *train_buf = tokbuf_alloc(1 << 24);   /* 16M initial capacity */
    TokenBuf *val_buf   = tokbuf_alloc(1 << 22);   /*  4M initial capacity */

    encode_corpus((const char **)docs, n_docs, tok, &cfg,
                  train_buf, val_buf);

    /* Free documents */
    for (long i = 0; i < n_docs; i++) free(docs[i]);
    free(docs);

    /* --- Guard: require at least one full sequence in each split ------- */
    if (train_buf->size < cfg.chunk_size) {
        fprintf(stderr,
            "[error] only %ld training tokens — need at least %d.\n"
            "        Use a larger corpus or smaller --chunk_size.\n",
            train_buf->size, cfg.chunk_size);
        tokbuf_free(train_buf); tokbuf_free(val_buf);
        tok_free(tok);
        return 1;
    }
    if (val_buf->size == 0) {
        fprintf(stderr,
            "[warning] no validation tokens — adjusting split.\n");
        /* Move the last 10%% of training tokens to validation */
        long move = train_buf->size / 10;
        long new_train_size = train_buf->size - move;
        tokbuf_append_array(val_buf,
                            (const int *)(train_buf->data + new_train_size),
                            (int)move);
        train_buf->size = new_train_size;
    }

    printf("[split] train: %ld tokens | val: %ld tokens\n",
           train_buf->size, val_buf->size);

    /* --- Write output -------------------------------------------------- */
    char train_path[512], val_path[512];
    snprintf(train_path, sizeof(train_path), "%s/train.bin", cfg.out_dir);
    snprintf(val_path,   sizeof(val_path),   "%s/val.bin",   cfg.out_dir);

    int ret = 0;
    ret |= write_bin(train_path, train_buf);
    ret |= write_bin(val_path,   val_buf);

    /* --- Write metadata ----------------------------------------------- */
    write_meta(cfg.out_dir, &cfg, tok,
               train_buf->size, val_buf->size, n_docs);

    /* --- Validate ------------------------------------------------------ */
    printf("\n--- Validation ---\n");
    validate_bin(train_path, tok);
    validate_bin(val_path,   tok);

    /* --- Cleanup ------------------------------------------------------- */
    tokbuf_free(train_buf);
    tokbuf_free(val_buf);
    tok_free(tok);

    if (ret == 0)
        printf("\nDone. Files written to %s/\n", cfg.out_dir);
    else
        fprintf(stderr, "\nErrors occurred during writing.\n");

    return ret;
}
