/**
 * tests/test_matrix.c
 *
 * Unit tests for all matrix operations.
 * Each test is self-contained and prints PASS/FAIL.
 *
 * Run: gcc -O2 -lm tests/test_matrix.c src/matrix.c -o test_matrix && ./test_matrix
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/matrix.h"

/* -------------------------------------------------------------------------
 * Test framework — minimal, no dependencies
 * ---------------------------------------------------------------------- */
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

#define EXPECT_NEAR(a, b, tol, msg) \
    EXPECT(fabsf((a) - (b)) < (tol), msg)

/* =========================================================================
 * Tests
 * ====================================================================== */

static void test_alloc_free(void) {
    printf("\n--- alloc / free ---\n");

    Matrix m = mat_alloc(3, 4, 0);
    EXPECT(m.data != NULL,  "data pointer non-null");
    EXPECT(m.grad == NULL,  "grad pointer null when not requested");
    EXPECT(m.rows == 3,     "rows set correctly");
    EXPECT(m.cols == 4,     "cols set correctly");
    mat_free(&m);
    EXPECT(m.data == NULL,  "data pointer zeroed after free");

    Matrix g = mat_alloc(2, 2, 1);
    EXPECT(g.grad != NULL,  "grad pointer non-null when requested");
    /* Grad should be zero-initialised */
    EXPECT(g.grad[0] == 0.0f && g.grad[3] == 0.0f, "grad zero-initialised");
    mat_free(&g);
}

static void test_mat_mul(void) {
    printf("\n--- mat_mul ---\n");

    /* [2×3] × [3×2] = [2×2] */
    Matrix A = mat_alloc(2, 3, 0);
    Matrix B = mat_alloc(3, 2, 0);
    Matrix C = mat_alloc(2, 2, 0);

    /* A = [[1,2,3],[4,5,6]] */
    A.data[0]=1; A.data[1]=2; A.data[2]=3;
    A.data[3]=4; A.data[4]=5; A.data[5]=6;

    /* B = [[7,8],[9,10],[11,12]] */
    B.data[0]=7;  B.data[1]=8;
    B.data[2]=9;  B.data[3]=10;
    B.data[4]=11; B.data[5]=12;

    mat_mul(&A, &B, &C);

    /* Expected: [[58,64],[139,154]] */
    EXPECT_NEAR(C.data[0], 58.0f,  1e-4f, "C[0][0] = 58");
    EXPECT_NEAR(C.data[1], 64.0f,  1e-4f, "C[0][1] = 64");
    EXPECT_NEAR(C.data[2], 139.0f, 1e-4f, "C[1][0] = 139");
    EXPECT_NEAR(C.data[3], 154.0f, 1e-4f, "C[1][1] = 154");

    mat_free(&A); mat_free(&B); mat_free(&C);
}

static void test_mat_mul_T(void) {
    printf("\n--- mat_mul_T ---\n");

    /* A [2×3] × Bᵀ [2×3] = [2×2] */
    Matrix A = mat_alloc(2, 3, 0);
    Matrix B = mat_alloc(2, 3, 0);
    Matrix C = mat_alloc(2, 2, 0);

    A.data[0]=1; A.data[1]=2; A.data[2]=3;
    A.data[3]=4; A.data[4]=5; A.data[5]=6;

    B.data[0]=1; B.data[1]=0; B.data[2]=0;
    B.data[3]=0; B.data[4]=1; B.data[5]=0;

    mat_mul_T(&A, &B, &C);

    /* B is [1,0,0; 0,1,0], so A × Bᵀ = A[:, :2] = [[1,2],[4,5]] */
    EXPECT_NEAR(C.data[0], 1.0f, 1e-4f, "C[0][0]");
    EXPECT_NEAR(C.data[1], 2.0f, 1e-4f, "C[0][1]");
    EXPECT_NEAR(C.data[2], 4.0f, 1e-4f, "C[1][0]");
    EXPECT_NEAR(C.data[3], 5.0f, 1e-4f, "C[1][1]");

    mat_free(&A); mat_free(&B); mat_free(&C);
}

static void test_softmax(void) {
    printf("\n--- softmax_rows ---\n");

    Matrix m = mat_alloc(1, 3, 0);
    m.data[0] = 1.0f;
    m.data[1] = 2.0f;
    m.data[2] = 3.0f;

    softmax_rows(&m);

    /* Sum should be 1.0 */
    float sum = m.data[0] + m.data[1] + m.data[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f, "softmax sums to 1.0");

    /* Values should be monotonically increasing */
    EXPECT(m.data[0] < m.data[1] && m.data[1] < m.data[2],
           "softmax preserves order");

    /* Test numerical stability with large values */
    Matrix big = mat_alloc(1, 3, 0);
    big.data[0] = 1000.0f;
    big.data[1] = 1001.0f;
    big.data[2] = 1002.0f;
    softmax_rows(&big);
    float bigsum = big.data[0] + big.data[1] + big.data[2];
    EXPECT(isfinite(bigsum), "softmax stable with large values");
    EXPECT_NEAR(bigsum, 1.0f, 1e-5f, "softmax(large) sums to 1.0");

    mat_free(&m);
    mat_free(&big);
}

static void test_rms_norm(void) {
    printf("\n--- rms_norm ---\n");

    int d = 4;
    Matrix in  = mat_alloc(1, d, 0);
    Matrix out = mat_alloc(1, d, 0);
    float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    in.data[0] = 1.0f;
    in.data[1] = 2.0f;
    in.data[2] = 3.0f;
    in.data[3] = 4.0f;

    rms_norm(&in, weight, 1e-5f, &out);

    /* After RMSNorm with unit weights, rms(out) should ≈ 1 */
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += out.data[i] * out.data[i];
    float rms = sqrtf(ss / d);
    EXPECT_NEAR(rms, 1.0f, 1e-4f, "rms_norm output has unit RMS");

    /* Test that scaling weight by 2 doubles the output */
    float weight2[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    Matrix out2 = mat_alloc(1, d, 0);
    rms_norm(&in, weight2, 1e-5f, &out2);
    EXPECT_NEAR(out2.data[0], 2.0f * out.data[0], 1e-4f,
                "rms_norm scales with weight");

    mat_free(&in); mat_free(&out); mat_free(&out2);
}

static void test_swiglu(void) {
    printf("\n--- swiglu ---\n");

    Matrix gate = mat_alloc(1, 4, 0);
    Matrix up   = mat_alloc(1, 4, 0);
    Matrix out  = mat_alloc(1, 4, 0);

    /* SwiGLU(0) = silu(0)*up = 0*up = 0 */
    for (int i = 0; i < 4; i++) { gate.data[i] = 0.0f; up.data[i] = 5.0f; }
    swiglu(&gate, &up, &out);
    EXPECT_NEAR(out.data[0], 0.0f, 1e-5f, "swiglu(gate=0) = 0");

    /* silu(x) is always positive for x>0 */
    for (int i = 0; i < 4; i++) { gate.data[i] = 1.0f; up.data[i] = 1.0f; }
    swiglu(&gate, &up, &out);
    EXPECT(out.data[0] > 0.0f, "swiglu(gate=1, up=1) > 0");

    /* silu(x) = x*sigmoid(x); at x=1: 1 * (1/(1+e^-1)) ≈ 0.7311 */
    float expected = 1.0f / (1.0f + expf(-1.0f));
    EXPECT_NEAR(out.data[0], expected, 1e-4f, "swiglu(1,1) ≈ silu(1)");

    mat_free(&gate); mat_free(&up); mat_free(&out);
}

static void test_embed_lookup(void) {
    printf("\n--- embed_lookup ---\n");

    int vocab = 5, d = 3, seq = 2;
    Matrix table = mat_alloc(vocab, d, 1);
    Matrix out   = mat_alloc(seq, d, 0);

    /* Fill table with distinct values */
    for (int i = 0; i < vocab * d; i++)
        table.data[i] = (float)i;

    int tokens[2] = {1, 3};
    embed_lookup(&table, tokens, seq, &out);

    /* Row 1 of table: [3,4,5] */
    EXPECT_NEAR(out.data[0], 3.0f, 1e-5f, "embed row 1, dim 0");
    EXPECT_NEAR(out.data[1], 4.0f, 1e-5f, "embed row 1, dim 1");
    EXPECT_NEAR(out.data[2], 5.0f, 1e-5f, "embed row 1, dim 2");
    /* Row 3 of table: [9,10,11] */
    EXPECT_NEAR(out.data[3], 9.0f,  1e-5f, "embed row 3, dim 0");
    EXPECT_NEAR(out.data[4], 10.0f, 1e-5f, "embed row 3, dim 1");
    EXPECT_NEAR(out.data[5], 11.0f, 1e-5f, "embed row 3, dim 2");

    mat_free(&table); mat_free(&out);
}

static void test_randn_stats(void) {
    printf("\n--- mat_randn statistics ---\n");

    int n = 10000;
    Matrix m = mat_alloc(1, n, 0);
    mat_randn(&m, 1.0f);

    /* Check mean ≈ 0 and std ≈ 1 */
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += m.data[i];
    mean /= n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = m.data[i] - mean;
        var += diff * diff;
    }
    var /= n;

    EXPECT_NEAR(mean, 0.0f, 0.05f, "randn mean ≈ 0 (within 0.05)");
    EXPECT_NEAR(var,  1.0f, 0.10f, "randn variance ≈ 1 (within 0.10)");

    mat_free(&m);
}

/* =========================================================================
 * Main
 * ====================================================================== */

int main(void) {
    printf("=== llm.c matrix unit tests ===\n");

    test_alloc_free();
    test_mat_mul();
    test_mat_mul_T();
    test_softmax();
    test_rms_norm();
    test_swiglu();
    test_embed_lookup();
    test_randn_stats();

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
