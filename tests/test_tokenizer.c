/**
 * tests/test_tokenizer.c
 *
 * Unit tests for the BPE tokenizer.
 *
 * Because tok_load() requires a .vocab file we can't easily create one
 * from scratch in a test, so we also test the internal components that
 * are accessible through the public API and by building a tiny in-memory
 * vocab via the same binary format that tok_load() reads.
 *
 * Run:
 *   gcc -O2 tests/test_tokenizer.c src/tokenizer.c -o build/test_tokenizer -lm
 *   ./build/test_tokenizer
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#include "../include/tokenizer.h"

/* =========================================================================
 * Minimal test framework
 * ====================================================================== */

static int tests_run    = 0;
static int tests_passed = 0;

#define EXPECT(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
        printf("  PASS  %s\n", msg); \
    } else { \
        printf("  FAIL  %s  (line %d)\n", msg, __LINE__); \
    } \
} while(0)


/* =========================================================================
 * Build a tiny test vocabulary in memory and write it to a temp file.
 *
 * Vocabulary:
 *   id 0  <unk>   score 0.0
 *   id 1  <bos>   score 0.0
 *   id 2  <eos>   score 0.0
 *   id 3  <pad>   score 0.0
 *   id 4  'a'     score 0.0   (byte 0x61 = 97, offset 4 → id 4+97=101... )
 *
 * Actually we build a minimal vocab that just covers the characters we
 * need for the tests, using the exact same byte-offset convention as
 * build_vocab.c:  byte b → id = b + 4  (so 'a'=0x61 → id 101, etc.)
 *
 * We also add one merge rule:  'a' + 'b' → 'ab'  (rank 0)
 * ====================================================================== */

/* We only need ASCII printable range for the tests: 32..126 */
#define TEST_BYTE_OFFSET 4
#define TEST_VOCAB_BYTES 256
#define TEST_VOCAB_SPECIAL 4
#define TEST_VOCAB_BASE (TEST_VOCAB_SPECIAL + TEST_VOCAB_BYTES)  /* 260 */

/* One extra merged token: 'ab' */
#define TEST_VOCAB_SIZE (TEST_VOCAB_BASE + 1)  /* 261 */
#define TEST_N_MERGES 1

/* Token ids for the characters used in tests */
#define ID_a  (TEST_BYTE_OFFSET + 'a')   /* 4 + 97  = 101 */
#define ID_b  (TEST_BYTE_OFFSET + 'b')   /* 4 + 98  = 102 */
#define ID_c  (TEST_BYTE_OFFSET + 'c')   /* 4 + 99  = 103 */
#define ID_AB TEST_VOCAB_BASE             /* 260 — merged 'ab' */

static const char *tmp_vocab_path = "/tmp/llm_test.vocab";

/**
 * write_test_vocab — write a minimal vocab file to /tmp for testing.
 */
static int write_test_vocab(void) {
    FILE *f = fopen(tmp_vocab_path, "wb");
    if (!f) { perror(tmp_vocab_path); return -1; }

    int32_t vocab_size = TEST_VOCAB_SIZE;
    int32_t n_merges   = TEST_N_MERGES;
    fwrite(&vocab_size, sizeof(int32_t), 1, f);
    fwrite(&n_merges,   sizeof(int32_t), 1, f);

    /* Helper: write one vocab entry */
    #define WRITE_ENTRY(id_, score_, text_) do { \
        int32_t _id  = (id_); \
        float   _sc  = (score_); \
        int32_t _len = (int32_t)strlen(text_); \
        fwrite(&_id,  sizeof(int32_t), 1, f); \
        fwrite(&_sc,  sizeof(float),   1, f); \
        fwrite(&_len, sizeof(int32_t), 1, f); \
        fwrite((text_), 1, _len, f); \
    } while(0)

    /* Special tokens */
    WRITE_ENTRY(0, 0.0f, "<unk>");
    WRITE_ENTRY(1, 0.0f, "<bos>");
    WRITE_ENTRY(2, 0.0f, "<eos>");
    WRITE_ENTRY(3, 0.0f, "<pad>");

    /* One token per byte 0x00-0xFF */
    for (int b = 0; b < 256; b++) {
        char buf[8];
        if (b >= 32 && b < 127) {
            buf[0] = (char)b; buf[1] = '\0';
        } else {
            snprintf(buf, sizeof(buf), "<%.2x>", b);
        }
        WRITE_ENTRY(TEST_BYTE_OFFSET + b, 0.0f, buf);
    }

    /* One merged token: 'ab' at id TEST_VOCAB_BASE */
    WRITE_ENTRY(TEST_VOCAB_BASE, -1.0f, "ab");

    #undef WRITE_ENTRY

    /* One merge rule: ID_a + ID_b → ID_AB  (rank 0) */
    int32_t left   = ID_a;
    int32_t right  = ID_b;
    int32_t result = ID_AB;
    fwrite(&left,   sizeof(int32_t), 1, f);
    fwrite(&right,  sizeof(int32_t), 1, f);
    fwrite(&result, sizeof(int32_t), 1, f);

    fclose(f);
    return 0;
}


/* =========================================================================
 * Tests
 * ====================================================================== */

static void test_load(Tokenizer *tok) {
    printf("\n--- tok_load ---\n");
    EXPECT(tok != NULL,                       "tok_load returns non-null");
    EXPECT(tok_vocab_size(tok) == TEST_VOCAB_SIZE,
                                              "vocab size matches");
}

static void test_special_tokens(Tokenizer *tok) {
    printf("\n--- special tokens ---\n");
    EXPECT(tok_token_to_id(tok, "<unk>") == TOK_UNK, "unk id = 0");
    EXPECT(tok_token_to_id(tok, "<bos>") == TOK_BOS, "bos id = 1");
    EXPECT(tok_token_to_id(tok, "<eos>") == TOK_EOS, "eos id = 2");
    EXPECT(tok_token_to_id(tok, "<pad>") == TOK_PAD, "pad id = 3");
}

static void test_byte_tokens(Tokenizer *tok) {
    printf("\n--- byte token round-trip ---\n");
    /* 'a' should decode back to "a" */
    const char *a_str = tok_decode_token(tok, ID_a);
    EXPECT(a_str != NULL && strcmp(a_str, "a") == 0,
           "id(a) decodes to 'a'");

    const char *b_str = tok_decode_token(tok, ID_b);
    EXPECT(b_str != NULL && strcmp(b_str, "b") == 0,
           "id(b) decodes to 'b'");

    /* Merged token */
    const char *ab_str = tok_decode_token(tok, ID_AB);
    EXPECT(ab_str != NULL && strcmp(ab_str, "ab") == 0,
           "id(AB) decodes to 'ab'");
}

static void test_encode_no_merge(Tokenizer *tok) {
    printf("\n--- encode (no merge applicable) ---\n");

    int ids[32]; int n = 0;
    int ret = tok_encode(tok, "c", 0, 0, ids, &n);

    EXPECT(ret == 0,   "encode returns 0");
    EXPECT(n   == 1,   "single char → 1 token");
    EXPECT(ids[0] == ID_c, "'c' maps to ID_c");
}

static void test_encode_with_merge(Tokenizer *tok) {
    printf("\n--- encode (merge 'a'+'b' → 'ab') ---\n");

    int ids[32]; int n = 0;
    tok_encode(tok, "ab", 0, 0, ids, &n);

    EXPECT(n == 1,          "'ab' merges to 1 token");
    EXPECT(ids[0] == ID_AB, "merged token id is ID_AB");
}

static void test_encode_partial_merge(Tokenizer *tok) {
    printf("\n--- encode (partial: 'abc' = 'ab' + 'c') ---\n");

    int ids[32]; int n = 0;
    tok_encode(tok, "abc", 0, 0, ids, &n);

    /* 'ab' should merge, 'c' stays separate */
    EXPECT(n == 2,           "'abc' → 2 tokens");
    EXPECT(ids[0] == ID_AB,  "first token is merged 'ab'");
    EXPECT(ids[1] == ID_c,   "second token is 'c'");
}

static void test_encode_bos_eos(Tokenizer *tok) {
    printf("\n--- encode with BOS/EOS ---\n");

    int ids[32]; int n = 0;
    tok_encode(tok, "c", 1, 1, ids, &n);

    EXPECT(n == 3,              "BOS + c + EOS = 3 tokens");
    EXPECT(ids[0] == TOK_BOS,   "first token is BOS");
    EXPECT(ids[1] == ID_c,      "middle token is 'c'");
    EXPECT(ids[2] == TOK_EOS,   "last token is EOS");
}

static void test_encode_empty_string(Tokenizer *tok) {
    printf("\n--- encode empty string ---\n");

    int ids[32]; int n = 0;
    tok_encode(tok, "", 0, 0, ids, &n);
    EXPECT(n == 0, "empty string → 0 tokens");

    tok_encode(tok, "", 1, 1, ids, &n);
    EXPECT(n == 2,            "empty + BOS/EOS → 2 tokens");
    EXPECT(ids[0] == TOK_BOS, "first is BOS");
    EXPECT(ids[1] == TOK_EOS, "second is EOS");
}

static void test_decode_roundtrip(Tokenizer *tok) {
    printf("\n--- decode round-trip ---\n");

    /* Encode "abc", decode back — should be "abc" */
    int ids[32]; int n = 0;
    tok_encode(tok, "abc", 0, 0, ids, &n);

    char *decoded = tok_decode(tok, ids, n);
    EXPECT(decoded != NULL,              "tok_decode returns non-null");
    EXPECT(strcmp(decoded, "abc") == 0,  "decoded matches original 'abc'");
    free(decoded);

    /* Encode a longer string */
    const char *original = "abcabc";
    tok_encode(tok, original, 0, 0, ids, &n);
    char *decoded2 = tok_decode(tok, ids, n);
    EXPECT(decoded2 != NULL,                    "tok_decode longer string");
    EXPECT(strcmp(decoded2, original) == 0,     "decoded matches original");
    free(decoded2);
}

static void test_encode_repeated_merge(Tokenizer *tok) {
    printf("\n--- encode repeated pair ---\n");

    /* "ababab" should become three 'ab' tokens */
    int ids[32]; int n = 0;
    tok_encode(tok, "ababab", 0, 0, ids, &n);

    EXPECT(n == 3,           "'ababab' → 3 merged tokens");
    EXPECT(ids[0] == ID_AB,  "token 0 is 'ab'");
    EXPECT(ids[1] == ID_AB,  "token 1 is 'ab'");
    EXPECT(ids[2] == ID_AB,  "token 2 is 'ab'");
}

static void test_encode_no_cross_boundary(Tokenizer *tok) {
    printf("\n--- encode: no merge across word boundary ---\n");

    /* "a b" — the space is a boundary; 'a' and 'b' should NOT merge */
    int ids[32]; int n = 0;
    tok_encode(tok, "a b", 0, 0, ids, &n);

    /* Depending on how the space is encoded, there could be 2 or 3 tokens.
     * The key assertion: we must NOT see ID_AB. */
    int has_merged = 0;
    for (int i = 0; i < n; i++)
        if (ids[i] == ID_AB) has_merged = 1;

    EXPECT(!has_merged, "'a b' does not produce merged 'ab' token");
}

static void test_tok_token_to_id_unknown(Tokenizer *tok) {
    printf("\n--- tok_token_to_id unknown ---\n");
    int id = tok_token_to_id(tok, "zzz_not_in_vocab_zzz");
    EXPECT(id == TOK_UNK, "unknown text returns TOK_UNK");
}

static void test_decode_token_oob(Tokenizer *tok) {
    printf("\n--- tok_decode_token out-of-bounds ---\n");
    /* Should not crash and should return a non-null placeholder */
    const char *s1 = tok_decode_token(tok, -1);
    const char *s2 = tok_decode_token(tok, 999999);
    EXPECT(s1 != NULL, "decode id=-1  returns non-null");
    EXPECT(s2 != NULL, "decode id=OOB returns non-null");
}

/* =========================================================================
 * Main
 * ====================================================================== */

int main(void) {
    printf("=== llm.c tokenizer unit tests ===\n");

    /* Write minimal test vocab to a temp file */
    if (write_test_vocab() < 0) {
        fprintf(stderr, "FATAL: could not write test vocab\n");
        return 1;
    }

    /* Load it */
    Tokenizer *tok = tok_load(tmp_vocab_path);
    if (!tok) {
        fprintf(stderr, "FATAL: tok_load failed on test vocab\n");
        return 1;
    }

    /* Run all tests */
    test_load(tok);
    test_special_tokens(tok);
    test_byte_tokens(tok);
    test_encode_no_merge(tok);
    test_encode_with_merge(tok);
    test_encode_partial_merge(tok);
    test_encode_bos_eos(tok);
    test_encode_empty_string(tok);
    test_decode_roundtrip(tok);
    test_encode_repeated_merge(tok);
    test_encode_no_cross_boundary(tok);
    test_tok_token_to_id_unknown(tok);
    test_decode_token_oob(tok);

    tok_free(tok);

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
