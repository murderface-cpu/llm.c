/**
 * tokenizer.h
 *
 * Byte-Pair Encoding (BPE) tokenizer.
 *
 * BPE works by starting with individual bytes (or characters) as the base
 * vocabulary, then iteratively merging the most frequent adjacent pair into
 * a new token until reaching vocab_size.
 *
 * This implementation supports:
 *   - Loading a pre-trained vocabulary from a file (compatible with
 *     LLaMA / SentencePiece binary format for easy model distillation)
 *   - Encoding a UTF-8 string to a token id array
 *   - Decoding a token id array back to a UTF-8 string
 *   - Special tokens: <BOS>, <EOS>, <PAD>, <UNK>
 */

#ifndef LLM_TOKENIZER_H
#define LLM_TOKENIZER_H

#include <stddef.h>

/* ---------------------------------------------------------------------------
 * Special token ids (reserved positions in every vocabulary)
 * ------------------------------------------------------------------------- */
#define TOK_UNK  0   /* Unknown token                                        */
#define TOK_BOS  1   /* Beginning of sequence                                */
#define TOK_EOS  2   /* End of sequence                                      */
#define TOK_PAD  3   /* Padding                                              */

/* Maximum byte length of a single token string (UTF-8 aware)               */
#define TOK_MAX_LEN 256


/* ---------------------------------------------------------------------------
 * TokenEntry  —  one entry in the vocabulary table
 * ------------------------------------------------------------------------- */
typedef struct {
    char  *text;    /* UTF-8 string representation of this token             */
    float  score;   /* BPE merge priority score (higher = merged earlier)    */
    int    id;      /* Token id (index into vocabulary)                      */
} TokenEntry;


/* ---------------------------------------------------------------------------
 * Tokenizer  —  the full tokenizer state
 * ------------------------------------------------------------------------- */
typedef struct {
    TokenEntry *vocab;          /* Array of vocab_size entries               */
    int         vocab_size;

    /* Sorted merge table for encoding ------------------------------------- */
    /* Merges are stored as pairs of token ids; applied greedily in order.   */
    int        *merge_left;     /* [n_merges] left token of each merge rule  */
    int        *merge_right;    /* [n_merges] right token of each merge rule */
    int        *merge_result;   /* [n_merges] result token id                */
    int         n_merges;

    /* Reverse lookup: token id → vocab string (just vocab[id].text)        */
    /* Included explicitly to emphasise O(1) decode access                  */
    char      **id_to_text;    /* [vocab_size] pointers into vocab[i].text   */

    /* Byte-fallback: one token per raw byte (for unknown UTF-8 bytes)      */
    int         byte_tokens[256];  /* byte value → token id                  */
} Tokenizer;


/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */

/**
 * tok_load — load a tokenizer from a binary vocabulary file.
 *
 * File format (custom, simple):
 *   [int32: vocab_size]
 *   [int32: n_merges]
 *   For each token:  [int32: id][float32: score][int32: text_len][char*: text]
 *   For each merge:  [int32: left][int32: right][int32: result]
 *
 * Returns NULL on failure.
 */
Tokenizer *tok_load(const char *path);

/** tok_free — release all tokenizer memory. */
void tok_free(Tokenizer *t);


/* ---------------------------------------------------------------------------
 * Encoding
 * ------------------------------------------------------------------------- */

/**
 * tok_encode — encode a UTF-8 string into token ids.
 *
 * Algorithm:
 *   1. Split string into UTF-8 code points.
 *   2. Look up each code point in the vocabulary (or use byte fallback).
 *   3. Greedily apply BPE merge rules left-to-right.
 *
 * @text:       null-terminated UTF-8 input string
 * @add_bos:    if non-zero, prepend TOK_BOS
 * @add_eos:    if non-zero, append  TOK_EOS
 * @out_ids:    caller-allocated int array; must be large enough
 *              (safe size: strlen(text) + 2 is always sufficient)
 * @out_len:    set to the number of tokens written
 *
 * Returns 0 on success, -1 on error.
 */
int tok_encode(const Tokenizer *t, const char *text,
               int add_bos, int add_eos,
               int *out_ids, int *out_len);

/**
 * tok_encode_batch — encode multiple strings into a padded int matrix.
 * Useful for preparing training batches.
 *
 * @texts:    array of n_strings null-terminated strings
 * @n:        number of strings
 * @max_len:  output sequence length (truncate or pad to this)
 * @out:      [n × max_len] caller-allocated integer matrix (row-major)
 * @lengths:  [n]  actual token counts before padding (caller-allocated)
 */
void tok_encode_batch(const Tokenizer *t,
                      const char **texts, int n, int max_len,
                      int *out, int *lengths);


/* ---------------------------------------------------------------------------
 * Decoding
 * ------------------------------------------------------------------------- */

/**
 * tok_decode — decode a single token id to its string.
 * Returns a pointer into the tokenizer's internal vocabulary table.
 * Do NOT free the returned pointer.
 */
const char *tok_decode_token(const Tokenizer *t, int id);

/**
 * tok_decode — decode an array of token ids to a heap-allocated UTF-8 string.
 * Caller must free() the returned string.
 *
 * @ids:    token id array
 * @n:      number of tokens
 * Returns: null-terminated UTF-8 string, or NULL on allocation failure.
 */
char *tok_decode(const Tokenizer *t, const int *ids, int n);


/* ---------------------------------------------------------------------------
 * Vocabulary utilities
 * ------------------------------------------------------------------------- */

/**
 * tok_token_to_id — look up a string in the vocabulary, return its id.
 * Returns TOK_UNK if not found.
 */
int tok_token_to_id(const Tokenizer *t, const char *text);

/**
 * tok_vocab_size — return the vocabulary size.
 */
int tok_vocab_size(const Tokenizer *t);

#endif /* LLM_TOKENIZER_H */