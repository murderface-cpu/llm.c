/**
 * tokenizer.c
 *
 * Byte-Pair Encoding tokenizer implementation.
 *
 * BPE algorithm recap:
 *   Training (done offline, produces the .vocab file):
 *     Start with every character as its own token.
 *     Count all adjacent pairs. Merge the most frequent pair into one token.
 *     Repeat until vocabulary reaches the target size.
 *
 *   Encoding (this file — inference side):
 *     1. Split text into individual UTF-8 code points.
 *     2. Map each code point to its token id (byte fallback for unknowns).
 *     3. Greedily scan left-to-right; apply the highest-priority merge rule
 *        that matches two adjacent tokens. Repeat until no merges apply.
 *
 * The merge table is stored as a priority-ranked list (rank 0 = merged first).
 * We use a hash map for fast pair lookup: (left_id, right_id) → (rank, result_id).
 *
 * Vocabulary file format (our custom binary format, not SentencePiece):
 *   [int32:  vocab_size]
 *   [int32:  n_merges]
 *   For each vocab entry (vocab_size entries):
 *     [int32:  id]
 *     [float32: score]
 *     [int32:  text_byte_len]
 *     [char*:  text bytes (not null-terminated in file)]
 *   For each merge rule (n_merges entries):
 *     [int32:  left_token_id]
 *     [int32:  right_token_id]
 *     [int32:  result_token_id]
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "../include/tokenizer.h"

/* =========================================================================
 * Internal: merge hash map
 *
 * Maps (left_id, right_id) → (merge_rank, result_id)
 * Used during encoding to find the best merge at each step.
 *
 * We use open-addressing with linear probing.
 * Key = 64-bit integer: high 32 bits = left_id, low 32 bits = right_id.
 * ====================================================================== */

#define MERGE_MAP_EMPTY  UINT64_MAX   /* sentinel for empty bucket          */
#define MERGE_MAP_LOAD   0.6f         /* max load factor before we'd resize */

typedef struct {
    uint64_t key;          /* packed (left_id << 32 | right_id), EMPTY if free */
    int      rank;         /* merge priority: lower = applied first            */
    int      result;       /* token id produced by this merge                  */
} MergeEntry;

typedef struct {
    MergeEntry *buckets;
    int         capacity;  /* must be a power of two                           */
    int         size;
} MergeMap;

static MergeMap *merge_map_alloc(int n_merges) {
    /* Size the table so load factor stays below MERGE_MAP_LOAD */
    int cap = 1;
    while (cap < (int)(n_merges / MERGE_MAP_LOAD) + 1)
        cap <<= 1;

    MergeMap *m = (MergeMap *)malloc(sizeof(MergeMap));
    m->capacity = cap;
    m->size     = 0;
    m->buckets  = (MergeEntry *)malloc((size_t)cap * sizeof(MergeEntry));
    for (int i = 0; i < cap; i++)
        m->buckets[i].key = MERGE_MAP_EMPTY;
    return m;
}

static void merge_map_free(MergeMap *m) {
    free(m->buckets);
    free(m);
}

static inline uint64_t merge_key(int left, int right) {
    return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
}

static void merge_map_insert(MergeMap *m, int left, int right,
                              int rank, int result) {
    uint64_t key  = merge_key(left, right);
    int      mask = m->capacity - 1;
    int      idx  = (int)(key & mask);

    while (m->buckets[idx].key != MERGE_MAP_EMPTY)
        idx = (idx + 1) & mask;

    m->buckets[idx].key    = key;
    m->buckets[idx].rank   = rank;
    m->buckets[idx].result = result;
    m->size++;
}

/**
 * merge_map_lookup — find the merge for (left, right).
 * Returns the MergeEntry*, or NULL if no merge rule exists for this pair.
 */
static const MergeEntry *merge_map_lookup(const MergeMap *m,
                                           int left, int right) {
    uint64_t key  = merge_key(left, right);
    int      mask = m->capacity - 1;
    int      idx  = (int)(key & mask);

    while (m->buckets[idx].key != MERGE_MAP_EMPTY) {
        if (m->buckets[idx].key == key)
            return &m->buckets[idx];
        idx = (idx + 1) & mask;
    }
    return NULL;
}


/* =========================================================================
 * Extended tokenizer struct
 *
 * We embed the merge map inside an extended struct that sits behind the
 * public Tokenizer pointer. Callers only ever see (Tokenizer *) so the
 * hash map is fully encapsulated.
 * ====================================================================== */
typedef struct {
    Tokenizer  pub;         /* public fields — must be first for safe casting */
    MergeMap  *merge_map;   /* fast (left,right) → (rank,result) lookup       */
} TokenizerInternal;

/* Helper: cast public pointer to internal struct */
static inline TokenizerInternal *tok_internal(const Tokenizer *t) {
    return (TokenizerInternal *)(void *)t;
}


/* =========================================================================
 * Helper: safe fread with error checking
 * ====================================================================== */
static int fread_ok(void *ptr, size_t sz, size_t n, FILE *f) {
    return fread(ptr, sz, n, f) == n ? 0 : -1;
}


/* =========================================================================
 * tok_load
 * ====================================================================== */

Tokenizer *tok_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    /* Read header */
    int32_t vocab_size = 0, n_merges = 0;
    if (fread_ok(&vocab_size, sizeof(int32_t), 1, f) < 0 ||
        fread_ok(&n_merges,   sizeof(int32_t), 1, f) < 0) {
        fprintf(stderr, "[tok_load] bad header in %s\n", path);
        fclose(f); return NULL;
    }

    /* Allocate internal struct */
    TokenizerInternal *ti = (TokenizerInternal *)calloc(
                                1, sizeof(TokenizerInternal));
    Tokenizer *t = &ti->pub;
    t->vocab_size = vocab_size;
    t->vocab      = (TokenEntry *)calloc(vocab_size, sizeof(TokenEntry));
    t->id_to_text = (char **)calloc(vocab_size, sizeof(char *));
    t->merge_left   = (int *)malloc((size_t)n_merges * sizeof(int));
    t->merge_right  = (int *)malloc((size_t)n_merges * sizeof(int));
    t->merge_result = (int *)malloc((size_t)n_merges * sizeof(int));
    t->n_merges     = n_merges;

    if (!t->vocab || !t->id_to_text ||
        !t->merge_left || !t->merge_right || !t->merge_result) {
        fprintf(stderr, "[tok_load] OOM\n");
        tok_free(t); fclose(f); return NULL;
    }

    /* Read vocab entries */
    for (int i = 0; i < vocab_size; i++) {
        int32_t  id  = 0;
        float    score = 0.0f;
        int32_t  text_len = 0;

        if (fread_ok(&id,       sizeof(int32_t), 1, f) < 0 ||
            fread_ok(&score,    sizeof(float),   1, f) < 0 ||
            fread_ok(&text_len, sizeof(int32_t), 1, f) < 0) {
            fprintf(stderr, "[tok_load] bad vocab entry %d\n", i);
            tok_free(t); fclose(f); return NULL;
        }

        char *text = (char *)malloc((size_t)text_len + 1);
        if (!text || fread_ok(text, 1, text_len, f) < 0) {
            free(text);
            fprintf(stderr, "[tok_load] bad token text %d\n", i);
            tok_free(t); fclose(f); return NULL;
        }
        text[text_len] = '\0';

        t->vocab[i].id    = id;
        t->vocab[i].score = score;
        t->vocab[i].text  = text;

        if (id >= 0 && id < vocab_size)
            t->id_to_text[id] = text;
    }

    /* Read merge rules and build the hash map */
    ti->merge_map = merge_map_alloc(n_merges);
    for (int i = 0; i < n_merges; i++) {
        int32_t left = 0, right = 0, result = 0;
        if (fread_ok(&left,   sizeof(int32_t), 1, f) < 0 ||
            fread_ok(&right,  sizeof(int32_t), 1, f) < 0 ||
            fread_ok(&result, sizeof(int32_t), 1, f) < 0) {
            fprintf(stderr, "[tok_load] bad merge rule %d\n", i);
            tok_free(t); fclose(f); return NULL;
        }
        t->merge_left[i]   = left;
        t->merge_right[i]  = right;
        t->merge_result[i] = result;
        /* rank = index in the file: lower index = higher priority */
        merge_map_insert(ti->merge_map, left, right, i, result);
    }

    /* Build byte-fallback table.
     * For each raw byte value 0..255, find the token whose text is exactly
     * that single byte.  If none exists (shouldn't happen in a well-formed
     * BPE vocab), fall back to TOK_UNK. */
    for (int b = 0; b < 256; b++)
        t->byte_tokens[b] = TOK_UNK;

    for (int i = 0; i < vocab_size; i++) {
        const char *txt = t->vocab[i].text;
        if (txt && strlen(txt) == 1) {
            unsigned char b = (unsigned char)txt[0];
            t->byte_tokens[b] = t->vocab[i].id;
        }
    }

    fclose(f);
    return t;
}


/* =========================================================================
 * tok_free
 * ====================================================================== */

void tok_free(Tokenizer *t) {
    if (!t) return;
    TokenizerInternal *ti = tok_internal(t);

    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++)
            free(t->vocab[i].text);
        free(t->vocab);
    }
    free(t->id_to_text);
    free(t->merge_left);
    free(t->merge_right);
    free(t->merge_result);
    if (ti->merge_map) merge_map_free(ti->merge_map);
    free(ti);
}


/* =========================================================================
 * UTF-8 helpers
 * ====================================================================== */

/**
 * utf8_codepoint_len — return the byte length of the UTF-8 sequence
 * starting at `s`, or 1 on invalid/continuation byte (byte fallback).
 */
static int utf8_codepoint_len(unsigned char c) {
    if      (c < 0x80) return 1;   /* ASCII                   */
    else if (c < 0xC0) return 1;   /* continuation — treat as byte */
    else if (c < 0xE0) return 2;   /* 2-byte sequence         */
    else if (c < 0xF0) return 3;   /* 3-byte sequence         */
    else               return 4;   /* 4-byte sequence         */
}


/* =========================================================================
 * BPE encoding
 *
 * We maintain a doubly-linked list of token nodes so merges are O(1) to
 * apply (just splice out the right node and update the left node's token).
 * Finding the best merge at each round is O(n) in the current token count.
 *
 * This gives overall O(n²) worst-case for a sequence of length n, which is
 * fine for typical sentence lengths (< 4096 bytes).
 * ====================================================================== */

/* One node in the linked list of token ids during BPE merge */
typedef struct TokNode {
    int           id;
    struct TokNode *prev;
    struct TokNode *next;
} TokNode;

int tok_encode(const Tokenizer *t, const char *text,
               int add_bos, int add_eos,
               int *out_ids, int *out_len) {
    if (!t || !text || !out_ids || !out_len) return -1;

    const TokenizerInternal *ti = tok_internal(t);
    int len = (int)strlen(text);

    /* Upper bound on initial token count: one per byte */
    TokNode *nodes = (TokNode *)malloc(
                         ((size_t)len + 4) * sizeof(TokNode));
    if (!nodes) return -1;

    /* Allocate a pool of free nodes for merge results */
    int      pos   = 0;   /* write head into nodes[] */

    /* --- Step 1: sentinel head and optional BOS ----------------------- */
    /* We use a circular sentinel approach: nodes[0] is a dummy head node */
    nodes[pos].id   = -1;     /* sentinel */
    nodes[pos].prev = NULL;
    nodes[pos].next = NULL;
    TokNode *head = &nodes[pos++];
    TokNode *tail = head;

/* Append a token node to the linked list — written as a macro to stay
     * ISO C99/C11 compliant (no nested functions). */
#define APPEND(tok_id) do { \
        nodes[pos].id   = (tok_id); \
        nodes[pos].prev = tail; \
        nodes[pos].next = NULL; \
        tail->next      = &nodes[pos]; \
        tail            = &nodes[pos]; \
        pos++;  \
    } while(0)

    if (add_bos) APPEND(TOK_BOS);

    /* --- Step 2: split text into UTF-8 code points, map to token ids -- */
    const char *p = text;
    while (*p) {
        int clen = utf8_codepoint_len((unsigned char)*p);

        /* Try to find this code point as a single vocab entry */
        /* Linear scan here — for large vocabs a hash map would be better.
         * In practice the initial split is a small fraction of total time. */
        int found = TOK_UNK;
        for (int i = 0; i < t->vocab_size; i++) {
            const char *vt = t->vocab[i].text;
            if (vt && (int)strlen(vt) == clen &&
                memcmp(vt, p, clen) == 0) {
                found = t->vocab[i].id;
                break;
            }
        }

        if (found == TOK_UNK) {
            /* Byte fallback: emit one token per raw byte */
            for (int b = 0; b < clen; b++)
                APPEND(t->byte_tokens[(unsigned char)p[b]]);
        } else {
            APPEND(found);
        }
        p += clen;
    }

    if (add_eos) APPEND(TOK_EOS);
#undef APPEND

    /* --- Step 3: greedily apply BPE merge rules ----------------------- */
    /*
     * Repeat until no merge can be applied:
     *   - Scan through adjacent pairs (a, b)
     *   - For each pair, check if there's a merge rule
     *   - Track the pair with the lowest rank (highest priority)
     *   - Apply it: replace (a, b) with result; re-check neighbours
     *
     * We restart from the neighbours after each merge so we don't miss
     * new pairs created by the merge.
     */
    int changed = 1;
    while (changed) {
        changed = 0;
        int      best_rank   = INT32_MAX;
        int      best_result = -1;
        TokNode *best_left   = NULL;

        /* Find best merge in this pass */
        for (TokNode *n = head->next; n && n->next; n = n->next) {
            const MergeEntry *e = merge_map_lookup(ti->merge_map,
                                                   n->id, n->next->id);
            if (e && e->rank < best_rank) {
                best_rank   = e->rank;
                best_result = e->result;
                best_left   = n;
            }
        }

        if (best_left) {
            /* Apply the merge: left node takes result id, right is removed */
            TokNode *right = best_left->next;
            best_left->id  = best_result;
            best_left->next = right->next;
            if (right->next) right->next->prev = best_left;
            else             tail = best_left;
            changed = 1;
        }
    }

    /* --- Step 4: collect results --------------------------------------- */
    int count = 0;
    for (TokNode *n = head->next; n; n = n->next) {
        out_ids[count++] = n->id;
    }
    *out_len = count;

    free(nodes);
    return 0;
}

void tok_encode_batch(const Tokenizer *t,
                      const char **texts, int n, int max_len,
                      int *out, int *lengths) {
    int *scratch = (int *)malloc((size_t)(max_len + 2) * sizeof(int));
    for (int i = 0; i < n; i++) {
        int len = 0;
        tok_encode(t, texts[i], 1, 1, scratch, &len);
        /* Truncate to max_len */
        int copy = len < max_len ? len : max_len;
        memcpy(out + i * max_len, scratch, copy * sizeof(int));
        /* Pad with TOK_PAD */
        for (int j = copy; j < max_len; j++)
            out[i * max_len + j] = TOK_PAD;
        if (lengths) lengths[i] = copy;
    }
    free(scratch);
}


/* =========================================================================
 * Decoding
 * ====================================================================== */

const char *tok_decode_token(const Tokenizer *t, int id) {
    if (id < 0 || id >= t->vocab_size) return "<unk>";
    return t->id_to_text[id] ? t->id_to_text[id] : "<unk>";
}

char *tok_decode(const Tokenizer *t, const int *ids, int n) {
    /* First pass: compute total output length */
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        const char *s = tok_decode_token(t, ids[i]);
        total += strlen(s);
    }

    char *out = (char *)malloc(total + 1);
    if (!out) return NULL;

    char *p = out;
    for (int i = 0; i < n; i++) {
        const char *s = tok_decode_token(t, ids[i]);
        size_t slen   = strlen(s);
        memcpy(p, s, slen);
        p += slen;
    }
    *p = '\0';
    return out;
}


/* =========================================================================
 * Vocabulary utilities
 * ====================================================================== */

int tok_token_to_id(const Tokenizer *t, const char *text) {
    /* Linear scan — acceptable for the occasional lookup outside encode() */
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab[i].text && strcmp(t->vocab[i].text, text) == 0)
            return t->vocab[i].id;
    }
    return TOK_UNK;
}

int tok_vocab_size(const Tokenizer *t) {
    return t->vocab_size;
}
