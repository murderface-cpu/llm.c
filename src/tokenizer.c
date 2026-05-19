/**
 * tokenizer.c  — Fast BPE tokenizer
 *
 * Optimisations vs original:
 *   1. Codepoint → token id:  O(vocab) linear scan → O(1) string hash map
 *   2. BPE merge loop:        O(n²) per doc → O(n·k) with priority queue
 *      where k = average number of merges applied per token
 *   3. All merges looked up in O(1) via the existing MergeMap hash table
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "../include/tokenizer.h"

/* =========================================================================
 * Merge hash map  (left_id, right_id) → (rank, result_id)
 * ====================================================================== */
#define MERGE_MAP_EMPTY UINT64_MAX

typedef struct {
    uint64_t key;
    int      rank;
    int      result;
} MergeEntry;

typedef struct {
    MergeEntry *buckets;
    int         capacity;
    int         size;
} MergeMap;

static MergeMap *merge_map_alloc(int n) {
    int cap = 1;
    while (cap < (int)(n / 0.6f) + 4) cap <<= 1;
    MergeMap *m = malloc(sizeof(MergeMap));
    m->capacity = cap; m->size = 0;
    m->buckets  = malloc((size_t)cap * sizeof(MergeEntry));
    for (int i = 0; i < cap; i++) m->buckets[i].key = MERGE_MAP_EMPTY;
    return m;
}
static void merge_map_free(MergeMap *m) { free(m->buckets); free(m); }

static inline uint64_t mkey(int l, int r) {
    return ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
}
static void merge_map_insert(MergeMap *m, int l, int r, int rank, int res) {
    uint64_t k = mkey(l, r);
    int mask = m->capacity - 1, idx = (int)(k & mask);
    while (m->buckets[idx].key != MERGE_MAP_EMPTY) idx = (idx + 1) & mask;
    m->buckets[idx] = (MergeEntry){ k, rank, res };
    m->size++;
}
static const MergeEntry *merge_map_lookup(const MergeMap *m, int l, int r) {
    uint64_t k = mkey(l, r);
    int mask = m->capacity - 1, idx = (int)(k & mask);
    while (m->buckets[idx].key != MERGE_MAP_EMPTY) {
        if (m->buckets[idx].key == k) return &m->buckets[idx];
        idx = (idx + 1) & mask;
    }
    return NULL;
}

/* =========================================================================
 * String → token id hash map  (for O(1) codepoint lookup during encode)
 * Maps null-terminated string keys to int token ids.
 * ====================================================================== */
typedef struct StrCell {
    const char *key;  /* points into vocab[i].text — not owned */
    int         id;
    struct StrCell *next;
} StrCell;

typedef struct {
    StrCell **buckets;
    int       cap;
    int       size;
} StrMap;

static uint32_t str_hash(const char *s, int len) {
    /* FNV-1a */
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static StrMap *strmap_alloc(int cap) {
    int c = 1; while (c < cap * 2) c <<= 1;
    StrMap *m = calloc(1, sizeof(StrMap));
    m->buckets = calloc(c, sizeof(StrCell *));
    m->cap = c;
    return m;
}
static void strmap_free(StrMap *m) {
    for (int i = 0; i < m->cap; i++) {
        StrCell *c = m->buckets[i];
        while (c) { StrCell *nx = c->next; free(c); c = nx; }
    }
    free(m->buckets); free(m);
}
static void strmap_insert(StrMap *m, const char *key, int len, int id) {
    uint32_t h  = str_hash(key, len);
    int      idx = (int)(h & (m->cap - 1));
    StrCell *c  = malloc(sizeof(StrCell));
    c->key  = key;
    c->id   = id;
    c->next = m->buckets[idx];
    m->buckets[idx] = c;
    m->size++;
}
static int strmap_lookup(const StrMap *m, const char *key, int len) {
    uint32_t h   = str_hash(key, len);
    int      idx = (int)(h & (m->cap - 1));
    for (StrCell *c = m->buckets[idx]; c; c = c->next)
        if ((int)strlen(c->key) == len && memcmp(c->key, key, len) == 0)
            return c->id;
    return TOK_UNK;
}

/* =========================================================================
 * Internal tokenizer struct
 * ====================================================================== */
typedef struct {
    Tokenizer pub;
    MergeMap *merge_map;
    StrMap   *str_map;    /* codepoint string → token id, O(1) lookup */
} TokenizerInternal;

static inline TokenizerInternal *tok_int(const Tokenizer *t) {
    return (TokenizerInternal *)(void *)t;
}

/* =========================================================================
 * Checked fread
 * ====================================================================== */
static int fread_ok(void *p, size_t sz, size_t n, FILE *f) {
    return fread(p, sz, n, f) == n ? 0 : -1;
}

/* =========================================================================
 * tok_load
 * ====================================================================== */
Tokenizer *tok_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    int32_t vocab_size = 0, n_merges = 0;
    if (fread_ok(&vocab_size, 4, 1, f) < 0 ||
        fread_ok(&n_merges,   4, 1, f) < 0) {
        fprintf(stderr, "[tok_load] bad header\n");
        fclose(f); return NULL;
    }

    TokenizerInternal *ti = calloc(1, sizeof(TokenizerInternal));
    Tokenizer *t = &ti->pub;
    t->vocab_size   = vocab_size;
    t->vocab        = calloc(vocab_size, sizeof(TokenEntry));
    t->id_to_text   = calloc(vocab_size, sizeof(char *));
    t->merge_left   = malloc((size_t)n_merges * sizeof(int));
    t->merge_right  = malloc((size_t)n_merges * sizeof(int));
    t->merge_result = malloc((size_t)n_merges * sizeof(int));
    t->n_merges     = n_merges;

    /* Read vocab */
    for (int i = 0; i < vocab_size; i++) {
        int32_t id, tlen; float score;
        if (fread_ok(&id,    4, 1, f) < 0 ||
            fread_ok(&score, 4, 1, f) < 0 ||
            fread_ok(&tlen,  4, 1, f) < 0) {
            fprintf(stderr, "[tok_load] bad vocab entry %d\n", i);
            tok_free(t); fclose(f); return NULL;
        }
        char *text = malloc(tlen + 1);
        if (!text || fread_ok(text, 1, tlen, f) < 0) {
            free(text); tok_free(t); fclose(f); return NULL;
        }
        text[tlen] = '\0';
        t->vocab[i].id = id; t->vocab[i].score = score; t->vocab[i].text = text;
        if (id >= 0 && id < vocab_size) t->id_to_text[id] = text;
    }

    /* Read merges + build merge map */
    ti->merge_map = merge_map_alloc(n_merges + 4);
    for (int i = 0; i < n_merges; i++) {
        int32_t l, r, res;
        if (fread_ok(&l, 4, 1, f) < 0 || fread_ok(&r, 4, 1, f) < 0 ||
            fread_ok(&res, 4, 1, f) < 0) {
            fprintf(stderr, "[tok_load] bad merge %d\n", i);
            tok_free(t); fclose(f); return NULL;
        }
        t->merge_left[i] = l; t->merge_right[i] = r; t->merge_result[i] = res;
        merge_map_insert(ti->merge_map, l, r, i, res);
    }
    fclose(f);

    /* Build O(1) string lookup map */
    ti->str_map = strmap_alloc(vocab_size + 4);
    for (int i = 0; i < vocab_size; i++) {
        if (!t->vocab[i].text) continue;
        int len = (int)strlen(t->vocab[i].text);
        strmap_insert(ti->str_map, t->vocab[i].text, len, t->vocab[i].id);
    }

    /* Byte fallback table */
    for (int b = 0; b < 256; b++) t->byte_tokens[b] = TOK_UNK;
    for (int i = 0; i < vocab_size; i++) {
        const char *tx = t->vocab[i].text;
        if (tx && strlen(tx) == 1)
            t->byte_tokens[(unsigned char)tx[0]] = t->vocab[i].id;
    }

    return t;
}

/* =========================================================================
 * tok_free
 * ====================================================================== */
void tok_free(Tokenizer *t) {
    if (!t) return;
    TokenizerInternal *ti = tok_int(t);
    if (t->vocab) { for (int i=0;i<t->vocab_size;i++) free(t->vocab[i].text); free(t->vocab); }
    free(t->id_to_text);
    free(t->merge_left); free(t->merge_right); free(t->merge_result);
    if (ti->merge_map) merge_map_free(ti->merge_map);
    if (ti->str_map)   strmap_free(ti->str_map);
    free(ti);
}

/* =========================================================================
 * UTF-8 helpers
 * ====================================================================== */
static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

/* =========================================================================
 * BPE encode — O(n · k) with priority-queue merge selection
 *
 * We maintain an array of (token_id, is_alive) nodes plus a min-heap
 * keyed on merge rank so the best merge is always O(log n) away.
 * After each merge we push the two new adjacent pairs onto the heap.
 * Stale heap entries (pairs that no longer exist) are detected lazily.
 * ====================================================================== */

typedef struct {
    int  left_pos;   /* index of left token in the tokens[] array */
    int  rank;       /* merge rank (lower = higher priority)       */
    int  result;     /* merged token id                            */
} PQEntry;

/* Min-heap on rank */
static int pq_cmp(const PQEntry *a, const PQEntry *b) {
    return a->rank - b->rank;
}

typedef struct {
    PQEntry *h;
    int size, cap;
} PriQueue;

static void pq_push(PriQueue *pq, PQEntry e) {
    if (pq->size >= pq->cap) {
        pq->cap = pq->cap ? pq->cap * 2 : 64;
        pq->h   = realloc(pq->h, pq->cap * sizeof(PQEntry));
    }
    int i = pq->size++;
    pq->h[i] = e;
    while (i > 0) {
        int p = (i-1)/2;
        if (pq_cmp(&pq->h[p], &pq->h[i]) <= 0) break;
        PQEntry tmp = pq->h[p]; pq->h[p] = pq->h[i]; pq->h[i] = tmp;
        i = p;
    }
}

static PQEntry pq_pop(PriQueue *pq) {
    PQEntry top = pq->h[0];
    pq->h[0] = pq->h[--pq->size];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, best = i;
        if (l < pq->size && pq_cmp(&pq->h[l], &pq->h[best]) < 0) best = l;
        if (r < pq->size && pq_cmp(&pq->h[r], &pq->h[best]) < 0) best = r;
        if (best == i) break;
        PQEntry tmp = pq->h[i]; pq->h[i] = pq->h[best]; pq->h[best] = tmp;
        i = best;
    }
    return top;
}

/* Try to push a pair (tokens[i], tokens[next[i]]) if a merge rule exists */
static void maybe_push(const TokenizerInternal *ti,
                        const int *toks, const int *nxt,
                        int i, int n, PriQueue *pq) {
    int j = nxt[i];
    if (j < 0 || j >= n) return;
    const MergeEntry *e = merge_map_lookup(ti->merge_map, toks[i], toks[j]);
    if (e) pq_push(pq, (PQEntry){ i, e->rank, e->result });
}

int tok_encode(const Tokenizer *t, const char *text,
               int add_bos, int add_eos,
               int *out_ids, int *out_len) {
    if (!t || !text || !out_ids || !out_len) return -1;
    const TokenizerInternal *ti = tok_int(t);
    int text_len = (int)strlen(text);
    if (text_len == 0) {
        int pos = 0;
        if (add_bos) out_ids[pos++] = TOK_BOS;
        if (add_eos) out_ids[pos++] = TOK_EOS;
        *out_len = pos;
        return 0;
    }

    /* Allocate token array — upper bound: one per byte + bos + eos */
    int max_toks = text_len + 4;
    int *toks = malloc(max_toks * sizeof(int));
    int *nxt  = malloc(max_toks * sizeof(int)); /* next[i] = next alive index */
    int *prv  = malloc(max_toks * sizeof(int)); /* prev[i]                    */

    int pos = 0;
    if (add_bos) { toks[pos] = TOK_BOS; pos++; }

    /* Split text into initial tokens using O(1) string hash map */
    const char *p = text;
    while (*p) {
        int clen = utf8_len((unsigned char)*p);
        /* Clamp to actual remaining bytes */
        int remaining = (int)(text + text_len - p);
        if (clen > remaining) clen = remaining;

        int id = strmap_lookup(ti->str_map, p, clen);
        if (id == TOK_UNK) {
            /* byte fallback */
            for (int b = 0; b < clen; b++) {
                if (pos >= max_toks - 4) {
                    max_toks *= 2;
                    toks = realloc(toks, max_toks * sizeof(int));
                    nxt  = realloc(nxt,  max_toks * sizeof(int));
                    prv  = realloc(prv,  max_toks * sizeof(int));
                }
                toks[pos++] = t->byte_tokens[(unsigned char)p[b]];
            }
        } else {
            if (pos >= max_toks - 4) {
                max_toks *= 2;
                toks = realloc(toks, max_toks * sizeof(int));
                nxt  = realloc(nxt,  max_toks * sizeof(int));
                prv  = realloc(prv,  max_toks * sizeof(int));
            }
            toks[pos++] = id;
        }
        p += clen;
    }

    if (add_eos) toks[pos++] = TOK_EOS;
    int n = pos;

    /* Initialise next/prev linked list */
    for (int i = 0; i < n; i++) { nxt[i] = i+1; prv[i] = i-1; }
    nxt[n-1] = n; /* sentinel */

    /* Priority queue: push all initially mergeable adjacent pairs */
    PriQueue pq = { NULL, 0, 0 };
    for (int i = 0; i < n - 1; i++)
        maybe_push(ti, toks, nxt, i, n, &pq);

    /* Process merges in priority order */
    while (pq.size > 0) {
        PQEntry e = pq_pop(&pq);
        int i = e.left_pos;
        int j = nxt[i];

        /* Stale check: pair (i,j) must still exist and match */
        if (j >= n || toks[i] < 0 || toks[j] < 0) continue;
        const MergeEntry *cur = merge_map_lookup(ti->merge_map, toks[i], toks[j]);
        if (!cur || cur->rank != e.rank) continue;  /* stale entry */

        /* Apply merge */
        toks[i] = e.result;
        toks[j] = -1;  /* mark dead */
        nxt[i]  = nxt[j];
        if (nxt[j] < n) prv[nxt[j]] = i;

        /* Push new pairs created by this merge */
        maybe_push(ti, toks, nxt, prv[i] >= 0 ? prv[i] : i, n, &pq);
        maybe_push(ti, toks, nxt, i, n, &pq);
    }

    /* Collect surviving tokens */
    int out = 0;
    for (int i = 0; i < n; i++)
        if (toks[i] >= 0) out_ids[out++] = toks[i];
    *out_len = out;

    free(toks); free(nxt); free(prv); free(pq.h);
    return 0;
}

/* =========================================================================
 * Batch encode
 * ====================================================================== */
void tok_encode_batch(const Tokenizer *t,
                      const char **texts, int n, int max_len,
                      int *out, int *lengths) {
    int *scratch = malloc((size_t)(max_len + 4) * sizeof(int));
    for (int i = 0; i < n; i++) {
        int len = 0;
        tok_encode(t, texts[i], 1, 1, scratch, &len);
        int copy = len < max_len ? len : max_len;
        memcpy(out + i * max_len, scratch, copy * sizeof(int));
        for (int j = copy; j < max_len; j++) out[i*max_len+j] = TOK_PAD;
        if (lengths) lengths[i] = copy;
    }
    free(scratch);
}

/* =========================================================================
 * Decode
 * ====================================================================== */
const char *tok_decode_token(const Tokenizer *t, int id) {
    if (id < 0 || id >= t->vocab_size) return "<unk>";
    return t->id_to_text[id] ? t->id_to_text[id] : "<unk>";
}

char *tok_decode(const Tokenizer *t, const int *ids, int n) {
    size_t total = 0;
    for (int i = 0; i < n; i++) total += strlen(tok_decode_token(t, ids[i]));
    char *out = malloc(total + 1), *p = out;
    if (!out) return NULL;
    for (int i = 0; i < n; i++) {
        const char *s = tok_decode_token(t, ids[i]);
        size_t sl = strlen(s); memcpy(p, s, sl); p += sl;
    }
    *p = '\0';
    return out;
}

int tok_token_to_id(const Tokenizer *t, const char *text) {
    const TokenizerInternal *ti = tok_int(t);
    int id = strmap_lookup(ti->str_map, text, (int)strlen(text));
    return id;
}

int tok_vocab_size(const Tokenizer *t) { return t->vocab_size; }
