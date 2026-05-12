/**
 * tools/build_vocab.c
 *
 * Train a Byte-Pair Encoding vocabulary from a raw UTF-8 text corpus
 * and write it to a .vocab binary file consumed by tok_load().
 *
 * Usage:
 *   ./build/build_vocab <corpus.txt> <vocab_size> <output.vocab>
 *
 * Example:
 *   ./build/build_vocab data/corpus.txt 4096 data/tokenizer.vocab
 *
 * Algorithm:
 *   1. Read the entire corpus into memory.
 *   2. Split into bytes — each byte is its own initial token (256 base tokens).
 *      Special tokens <UNK>, <BOS>, <EOS>, <PAD> occupy ids 0-3.
 *      Bytes 0x00-0xFF get ids 4-259.
 *   3. Count every adjacent pair of token ids across the whole corpus.
 *   4. Find the most frequent pair; assign it a new token id; record the merge.
 *   5. Replace every occurrence of that pair in the corpus with the new id.
 *   6. Repeat steps 3-5 until we have reached vocab_size tokens.
 *   7. Write the .vocab binary file.
 *
 * Complexity:
 *   O(vocab_size × corpus_length) — acceptable for corpora up to ~500MB
 *   and vocab sizes up to ~32K on a modern CPU.
 *   For larger corpora use the Python sentencepiece library instead and
 *   convert with tools/convert_spm_vocab.py (future tool).
 *
 * Memory:
 *   The corpus is kept in memory as a flat int array (4 bytes per token).
 *   A 100MB text corpus uses ~400MB RAM during training.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>

/* =========================================================================
 * Constants
 * ====================================================================== */

/* First 4 ids are reserved for special tokens (matches tokenizer.h) */
#define TOK_UNK  0
#define TOK_BOS  1
#define TOK_EOS  2
#define TOK_PAD  3
#define TOK_BYTE_OFFSET 4   /* byte 0x00 gets id 4, byte 0xFF gets id 259  */
#define BASE_VOCAB_SIZE 260  /* 4 special + 256 bytes                       */

/* Maximum length of a merged token string (bytes) */
#define MAX_TOKEN_LEN 256

/* =========================================================================
 * Pair frequency hash map
 *
 * Maps (left_id, right_id) → count.
 * Open addressing, power-of-two capacity.
 * ====================================================================== */

#define PAIR_MAP_EMPTY UINT64_MAX

typedef struct {
    uint64_t key;    /* packed (left<<32|right), EMPTY if free */
    int64_t  count;  /* frequency of this pair                 */
} PairEntry;

typedef struct {
    PairEntry *buckets;
    int        capacity;
    int        size;
} PairMap;

static PairMap *pairmap_alloc(int capacity) {
    /* Round up to next power of two */
    int cap = 1;
    while (cap < capacity) cap <<= 1;
    PairMap *m = (PairMap *)malloc(sizeof(PairMap));
    m->capacity = cap;
    m->size     = 0;
    m->buckets  = (PairEntry *)malloc((size_t)cap * sizeof(PairEntry));
    for (int i = 0; i < cap; i++) m->buckets[i].key = PAIR_MAP_EMPTY;
    return m;
}

static void pairmap_free(PairMap *m) {
    free(m->buckets);
    free(m);
}

static void pairmap_clear(PairMap *m) {
    for (int i = 0; i < m->capacity; i++) m->buckets[i].key = PAIR_MAP_EMPTY;
    m->size = 0;
}

static inline uint64_t pair_key(int left, int right) {
    return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
}

/* Add `delta` to the count for (left, right), inserting if absent. */
static void pairmap_add(PairMap *m, int left, int right, int64_t delta) {
    /* Resize if load factor exceeds 0.7 */
    if (m->size > (int)(m->capacity * 0.7)) {
        int old_cap = m->capacity;
        PairEntry *old = m->buckets;
        m->capacity <<= 1;
        m->buckets = (PairEntry *)malloc((size_t)m->capacity * sizeof(PairEntry));
        for (int i = 0; i < m->capacity; i++) m->buckets[i].key = PAIR_MAP_EMPTY;
        m->size = 0;
        for (int i = 0; i < old_cap; i++) {
            if (old[i].key != PAIR_MAP_EMPTY) {
                int l = (int)(old[i].key >> 32);
                int r = (int)(old[i].key & 0xFFFFFFFFULL);
                pairmap_add(m, l, r, old[i].count);
            }
        }
        free(old);
    }

    uint64_t key  = pair_key(left, right);
    int      mask = m->capacity - 1;
    int      idx  = (int)(key & mask);
    while (m->buckets[idx].key != PAIR_MAP_EMPTY &&
           m->buckets[idx].key != key)
        idx = (idx + 1) & mask;

    if (m->buckets[idx].key == PAIR_MAP_EMPTY) {
        m->buckets[idx].key   = key;
        m->buckets[idx].count = delta;
        m->size++;
    } else {
        m->buckets[idx].count += delta;
    }
}

/* Find the entry with the highest count. Returns key or PAIR_MAP_EMPTY. */
static uint64_t pairmap_argmax(const PairMap *m, int64_t *out_count) {
    uint64_t best_key   = PAIR_MAP_EMPTY;
    int64_t  best_count = -1;
    for (int i = 0; i < m->capacity; i++) {
        if (m->buckets[i].key != PAIR_MAP_EMPTY &&
            m->buckets[i].count > best_count) {
            best_count = m->buckets[i].count;
            best_key   = m->buckets[i].key;
        }
    }
    if (out_count) *out_count = best_count;
    return best_key;
}


/* =========================================================================
 * Vocabulary table
 * ====================================================================== */

typedef struct {
    char  *text;     /* UTF-8 string for this token (heap-allocated) */
    float  score;    /* BPE score: log(frequency) of the merge       */
    int    id;
} VocabEntry;

typedef struct {
    VocabEntry *entries;
    int         size;
    int         capacity;
} Vocab;

static Vocab *vocab_alloc(int initial_cap) {
    Vocab *v    = (Vocab *)malloc(sizeof(Vocab));
    v->size     = 0;
    v->capacity = initial_cap;
    v->entries  = (VocabEntry *)malloc(initial_cap * sizeof(VocabEntry));
    return v;
}

static void vocab_free(Vocab *v) {
    for (int i = 0; i < v->size; i++) free(v->entries[i].text);
    free(v->entries);
    free(v);
}

static int vocab_add(Vocab *v, const char *text, float score, int id) {
    if (v->size >= v->capacity) {
        v->capacity *= 2;
        v->entries = (VocabEntry *)realloc(v->entries,
                         v->capacity * sizeof(VocabEntry));
    }
    v->entries[v->size].text  = strdup(text);
    v->entries[v->size].score = score;
    v->entries[v->size].id    = id;
    return v->size++;
}


/* =========================================================================
 * Merge table
 * ====================================================================== */

typedef struct {
    int left, right, result;
} MergeRule;

typedef struct {
    MergeRule *rules;
    int        size, capacity;
} MergeTable;

static MergeTable *mergetable_alloc(int cap) {
    MergeTable *mt = (MergeTable *)malloc(sizeof(MergeTable));
    mt->size     = 0;
    mt->capacity = cap;
    mt->rules    = (MergeRule *)malloc(cap * sizeof(MergeRule));
    return mt;
}

static void mergetable_free(MergeTable *mt) {
    free(mt->rules);
    free(mt);
}

static void mergetable_add(MergeTable *mt, int left, int right, int result) {
    if (mt->size >= mt->capacity) {
        mt->capacity *= 2;
        mt->rules = (MergeRule *)realloc(mt->rules,
                        mt->capacity * sizeof(MergeRule));
    }
    mt->rules[mt->size].left   = left;
    mt->rules[mt->size].right  = right;
    mt->rules[mt->size].result = result;
    mt->size++;
}


/* =========================================================================
 * Corpus representation
 *
 * We store the corpus as a flat int array of token ids.
 * A value of -1 marks a word boundary (space between words).
 * This lets us avoid merging across word boundaries (standard BPE practice).
 * ====================================================================== */

typedef struct {
    int  *tokens;    /* flat array of token ids; -1 = boundary */
    long  length;    /* number of elements (including boundaries) */
    long  capacity;
} Corpus;

static Corpus *corpus_alloc(long initial_cap) {
    Corpus *c   = (Corpus *)malloc(sizeof(Corpus));
    c->length   = 0;
    c->capacity = initial_cap;
    c->tokens   = (int *)malloc(initial_cap * sizeof(int));
    return c;
}

static void corpus_free(Corpus *c) {
    free(c->tokens);
    free(c);
}

static void corpus_append(Corpus *c, int tok) {
    if (c->length >= c->capacity) {
        c->capacity = c->capacity * 2 + 1;
        c->tokens   = (int *)realloc(c->tokens, c->capacity * sizeof(int));
    }
    c->tokens[c->length++] = tok;
}


/* =========================================================================
 * Load corpus from UTF-8 text file
 * ====================================================================== */

static Corpus *load_corpus(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    printf("[corpus] loading %s (%.1f MB)...\n",
           path, (double)fsize / 1e6);

    char *raw = (char *)malloc(fsize + 1);
    if (!raw || (long)fread(raw, 1, fsize, f) != fsize) {
        fprintf(stderr, "[corpus] read failed\n");
        fclose(f); free(raw); return NULL;
    }
    raw[fsize] = '\0';
    fclose(f);

    /*
     * Convert raw bytes to token ids.
     * Each byte → id = byte_value + TOK_BYTE_OFFSET (4..259).
     * Insert boundary (-1) at whitespace so merges don't cross words.
     *
     * We insert boundaries at: space, newline, tab.
     * The boundary itself is NOT emitted as a token — it just prevents
     * the adjacent real tokens from being merged together.
     */
    Corpus *corpus = corpus_alloc(fsize + 1);
    int in_space = 0;

    for (long i = 0; i < fsize; i++) {
        unsigned char b = (unsigned char)raw[i];
        if (b == ' ' || b == '\n' || b == '\t' || b == '\r') {
            if (!in_space) {
                corpus_append(corpus, -1);  /* boundary */
                in_space = 1;
            }
        } else {
            in_space = 0;
            corpus_append(corpus, (int)b + TOK_BYTE_OFFSET);
        }
    }

    free(raw);
    printf("[corpus] %ld tokens (boundaries included)\n", corpus->length);
    return corpus;
}


/* =========================================================================
 * BPE training
 * ====================================================================== */

/**
 * count_pairs — populate `pairs` with (left,right)→count for all adjacent
 * token pairs in the corpus (skipping boundaries).
 */
static void count_pairs(const Corpus *c, PairMap *pairs) {
    pairmap_clear(pairs);
    for (long i = 0; i + 1 < c->length; i++) {
        int a = c->tokens[i];
        int b = c->tokens[i + 1];
        if (a < 0 || b < 0) continue;   /* skip boundaries */
        pairmap_add(pairs, a, b, 1);
    }
}

/**
 * apply_merge — replace every occurrence of (left, right) in the corpus
 * with `result`, in-place.
 *
 * Returns the number of replacements made.
 */
static long apply_merge(Corpus *c, int left, int right, int result) {
    long replacements = 0;
    long write = 0;

    for (long read = 0; read < c->length; read++) {
        int tok = c->tokens[read];

        /* Check if this position starts a merge pair */
        if (tok == left &&
            read + 1 < c->length &&
            c->tokens[read + 1] == right) {
            /* Look behind: don't merge if previous token was a boundary
             * (the left token is already at a word start — it's fine to
             * merge it; boundaries only block cross-boundary merges).
             * The corpus already has -1 entries so we can just check
             * that tok != -1 and c->tokens[read+1] != -1, which we did. */
            c->tokens[write++] = result;
            read++;   /* skip the right token */
            replacements++;
        } else {
            c->tokens[write++] = tok;
        }
    }
    c->length = write;
    return replacements;
}

/**
 * train_bpe — run the full BPE training loop.
 *
 * @corpus:      the byte-tokenised corpus (modified in-place)
 * @target_size: desired final vocabulary size
 * @vocab:       output vocabulary table (caller-allocated, pre-filled with base tokens)
 * @merges:      output merge table (caller-allocated, empty)
 */
static void train_bpe(Corpus *corpus, int target_size,
                      Vocab *vocab, MergeTable *merges) {
    int n_merges_needed = target_size - vocab->size;
    printf("[bpe] training %d merges (base vocab = %d, target = %d)...\n",
           n_merges_needed, vocab->size, target_size);

    /* Allocate pair frequency map — sized generously */
    PairMap *pairs = pairmap_alloc(1 << 20);

    clock_t t0 = clock();

    for (int step = 0; step < n_merges_needed; step++) {
        /* Count all pairs */
        count_pairs(corpus, pairs);

        if (pairs->size == 0) {
            printf("[bpe] no more pairs at step %d; stopping early\n", step);
            break;
        }

        /* Find best pair */
        int64_t  best_count = 0;
        uint64_t best_key   = pairmap_argmax(pairs, &best_count);

        if (best_count < 2) {
            printf("[bpe] best pair frequency = %lld; stopping early\n",
                   (long long)best_count);
            break;
        }

        int left   = (int)(best_key >> 32);
        int right  = (int)(best_key & 0xFFFFFFFFULL);
        int new_id = vocab->size;

        /* Build the merged token string */
        const char *ls = vocab->entries[left].text;
        const char *rs = vocab->entries[right].text;
        char merged[MAX_TOKEN_LEN * 2 + 1];
        snprintf(merged, sizeof(merged), "%s%s", ls ? ls : "?", rs ? rs : "?");

        /* Score = log of frequency (SentencePiece convention) */
        float score = (float)(-step);   /* rank-based score, negative = lower priority */

        vocab_add(vocab, merged, score, new_id);
        mergetable_add(merges, left, right, new_id);

        /* Apply the merge to the corpus */
        long replaced = apply_merge(corpus, left, right, new_id);
        (void)replaced;

        /* Progress reporting */
        if (step % 500 == 0 || step == n_merges_needed - 1) {
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("[bpe] step %5d/%d  |  merged '%s' (freq=%lld)  "
                   "|  vocab=%d  |  %.1fs\n",
                   step + 1, n_merges_needed,
                   merged, (long long)best_count,
                   vocab->size, elapsed);
        }
    }

    pairmap_free(pairs);
    printf("[bpe] done. final vocab size = %d, merges = %d\n",
           vocab->size, merges->size);
}


/* =========================================================================
 * Write .vocab binary file
 *
 * Format (matches tok_load() in tokenizer.c):
 *   [int32: vocab_size]
 *   [int32: n_merges]
 *   For each token:
 *     [int32: id][float32: score][int32: text_byte_len][char* text]
 *   For each merge:
 *     [int32: left][int32: right][int32: result]
 * ====================================================================== */

static int write_vocab(const char *path, const Vocab *vocab,
                       const MergeTable *merges) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    int32_t vs = (int32_t)vocab->size;
    int32_t nm = (int32_t)merges->size;
    fwrite(&vs, sizeof(int32_t), 1, f);
    fwrite(&nm, sizeof(int32_t), 1, f);

    for (int i = 0; i < vocab->size; i++) {
        const VocabEntry *e = &vocab->entries[i];
        int32_t id       = (int32_t)e->id;
        float   score    = e->score;
        int32_t text_len = (int32_t)(e->text ? strlen(e->text) : 0);

        fwrite(&id,       sizeof(int32_t), 1, f);
        fwrite(&score,    sizeof(float),   1, f);
        fwrite(&text_len, sizeof(int32_t), 1, f);
        if (text_len > 0) fwrite(e->text, 1, text_len, f);
    }

    for (int i = 0; i < merges->size; i++) {
        int32_t left   = (int32_t)merges->rules[i].left;
        int32_t right  = (int32_t)merges->rules[i].right;
        int32_t result = (int32_t)merges->rules[i].result;
        fwrite(&left,   sizeof(int32_t), 1, f);
        fwrite(&right,  sizeof(int32_t), 1, f);
        fwrite(&result, sizeof(int32_t), 1, f);
    }

    fclose(f);
    printf("[vocab] wrote %d tokens + %d merges to %s\n",
           vocab->size, merges->size, path);
    return 0;
}


/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s <corpus.txt> <vocab_size> <output.vocab>\n\n"
            "  corpus.txt   — raw UTF-8 text file to train BPE on\n"
            "  vocab_size   — target vocabulary size (e.g. 4096 or 32000)\n"
            "  output.vocab — path to write the binary vocab file\n",
            argv[0]);
        return 1;
    }

    const char *corpus_path = argv[1];
    int         vocab_size  = atoi(argv[2]);
    const char *vocab_path  = argv[3];

    if (vocab_size < BASE_VOCAB_SIZE) {
        fprintf(stderr, "[error] vocab_size must be >= %d (base vocabulary)\n",
                BASE_VOCAB_SIZE);
        return 1;
    }

    /* --- Load corpus --------------------------------------------------- */
    Corpus *corpus = load_corpus(corpus_path);
    if (!corpus) return 1;

    /* --- Initialise vocabulary with special + byte tokens -------------- */
    Vocab      *vocab  = vocab_alloc(vocab_size + 64);
    MergeTable *merges = mergetable_alloc(vocab_size);

    /* Special tokens (must match tokenizer.h defines) */
    vocab_add(vocab, "<unk>", 0.0f, TOK_UNK);
    vocab_add(vocab, "<bos>", 0.0f, TOK_BOS);
    vocab_add(vocab, "<eos>", 0.0f, TOK_EOS);
    vocab_add(vocab, "<pad>", 0.0f, TOK_PAD);
    assert(vocab->size == 4);

    /* One token per raw byte value 0x00-0xFF */
    for (int b = 0; b < 256; b++) {
        char buf[5];
        /* Printable ASCII: store the character itself */
        if (b >= 32 && b < 127) {
            buf[0] = (char)b;
            buf[1] = '\0';
        } else {
            /* Non-printable: use a <0xNN> placeholder string */
            snprintf(buf, sizeof(buf), "<%.2x>", b);
        }
        vocab_add(vocab, buf, 0.0f, TOK_BYTE_OFFSET + b);
    }
    assert(vocab->size == BASE_VOCAB_SIZE);

    printf("[vocab] base vocabulary: %d tokens (4 special + 256 bytes)\n",
           vocab->size);

    /* --- Run BPE -------------------------------------------------------- */
    train_bpe(corpus, vocab_size, vocab, merges);

    /* --- Write output --------------------------------------------------- */
    int ret = write_vocab(vocab_path, vocab, merges);

    /* --- Cleanup -------------------------------------------------------- */
    corpus_free(corpus);
    vocab_free(vocab);
    mergetable_free(merges);

    return ret == 0 ? 0 : 1;
}
