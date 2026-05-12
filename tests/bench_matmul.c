/**
 * tests/bench_matmul.c
 *
 * Microbenchmark for mat_mul, mat_mul_T, rms_norm, and mat_add_inplace.
 *
 * Tests shapes that actually appear inside a 50M-param transformer:
 *   - Attention QKV projection:  [seq × d_model] × [d_model × qkv_dim]
 *   - Attention output proj:     [seq × qkv_dim] × [qkv_dim × d_model]
 *   - FFN gate/up projection:    [seq × d_model] × [d_model × ffn_hidden]
 *   - FFN down projection:       [seq × ffn_hidden] × [ffn_hidden × d_model]
 *   - Attention scores:          [seq × head_dim] × [head_dim × seq]  (mat_mul_T)
 *
 * Compile and run:
 *   gcc -O3 -march=native -ffast-math \
 *       tests/bench_matmul.c src/matrix.c -o build/bench_matmul -lm
 *   ./build/bench_matmul
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "../include/matrix.h"
#include "../include/simd.h"

/* =========================================================================
 * Timing helpers
 * ====================================================================== */

/** wall_ns — monotonic wall-clock time in nanoseconds */
static long long wall_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* =========================================================================
 * Benchmark infrastructure
 * ====================================================================== */

typedef struct {
    const char *name;
    double      gflops;
    double      ms_per_call;
    double      gb_per_s;      /* memory bandwidth (reads + writes) */
} BenchResult;

/**
 * bench_matmul — time mat_mul(A, B, out) for `iters` iterations.
 *
 * Reports:
 *   GFLOPS = 2·M·K·N / (time_per_call_in_seconds × 1e9)
 *            (factor 2 because each output element requires M FMAs = M muls + M adds)
 */
static BenchResult bench_matmul(const char *name,
                                 int M, int K, int N,
                                 int iters) {
    Matrix A   = mat_alloc(M, K, 0); mat_randn(&A,   1.0f);
    Matrix B   = mat_alloc(K, N, 0); mat_randn(&B,   1.0f);
    Matrix out = mat_alloc(M, N, 0);

    /* Warmup */
    for (int i = 0; i < 3; i++) mat_mul(&A, &B, &out);

    long long t0 = wall_ns();
    for (int i = 0; i < iters; i++) mat_mul(&A, &B, &out);
    long long t1 = wall_ns();

    double ms      = (double)(t1 - t0) / 1e6 / iters;
    double flops   = 2.0 * M * K * N;                   /* multiply-adds */
    double gflops  = flops / (ms * 1e6);                 /* GFLOP/s       */
    /* Bytes read: A + B; bytes written: out */
    double bytes   = ((long long)M * K + K * N + M * N) * sizeof(float);
    double gb_per_s= bytes / (ms * 1e6);

    mat_free(&A); mat_free(&B); mat_free(&out);

    return (BenchResult){ name, gflops, ms, gb_per_s };
}

/**
 * bench_matmul_T — time mat_mul_T(A, B, out).
 * Used for attention score computation Q × Kᵀ.
 */
static BenchResult bench_matmul_T(const char *name,
                                   int M, int K, int N,
                                   int iters) {
    /* mat_mul_T: A[M×K], B[N×K], out[M×N] */
    Matrix A   = mat_alloc(M, K, 0); mat_randn(&A, 1.0f);
    Matrix B   = mat_alloc(N, K, 0); mat_randn(&B, 1.0f);
    Matrix out = mat_alloc(M, N, 0);

    for (int i = 0; i < 3; i++) mat_mul_T(&A, &B, &out);

    long long t0 = wall_ns();
    for (int i = 0; i < iters; i++) mat_mul_T(&A, &B, &out);
    long long t1 = wall_ns();

    double ms     = (double)(t1 - t0) / 1e6 / iters;
    double gflops = 2.0 * M * K * N / (ms * 1e6);
    double bytes  = ((long long)M * K + N * K + M * N) * sizeof(float);
    double gb_per_s = bytes / (ms * 1e6);

    mat_free(&A); mat_free(&B); mat_free(&out);
    return (BenchResult){ name, gflops, ms, gb_per_s };
}

/**
 * bench_rms_norm — time rms_norm over a [seq × d_model] matrix.
 */
static BenchResult bench_rms_norm(const char *name, int rows, int cols,
                                   int iters) {
    Matrix in  = mat_alloc(rows, cols, 0); mat_randn(&in,  1.0f);
    Matrix out = mat_alloc(rows, cols, 0);
    float *w   = (float *)malloc(cols * sizeof(float));
    for (int i = 0; i < cols; i++) w[i] = 1.0f;

    for (int i = 0; i < 3; i++) rms_norm(&in, w, 1e-5f, &out);

    long long t0 = wall_ns();
    for (int i = 0; i < iters; i++) rms_norm(&in, w, 1e-5f, &out);
    long long t1 = wall_ns();

    double ms     = (double)(t1 - t0) / 1e6 / iters;
    /* RMSNorm: ~3 passes (ss accumulate, normalize, scale) over rows×cols floats */
    double flops  = 3.0 * rows * cols;
    double gflops = flops / (ms * 1e6);
    double bytes  = 2.0 * rows * cols * sizeof(float);
    double gb_per_s = bytes / (ms * 1e6);

    mat_free(&in); mat_free(&out); free(w);
    return (BenchResult){ name, gflops, ms, gb_per_s };
}

/**
 * bench_add — time mat_add_inplace (residual connections).
 */
static BenchResult bench_add(const char *name, int rows, int cols,
                               int iters) {
    Matrix A = mat_alloc(rows, cols, 0); mat_randn(&A, 1.0f);
    Matrix B = mat_alloc(rows, cols, 0); mat_randn(&B, 1.0f);

    for (int i = 0; i < 3; i++) mat_add_inplace(&A, &B);

    long long t0 = wall_ns();
    for (int i = 0; i < iters; i++) mat_add_inplace(&A, &B);
    long long t1 = wall_ns();

    double ms     = (double)(t1 - t0) / 1e6 / iters;
    double flops  = (double)rows * cols;
    double gflops = flops / (ms * 1e6);
    double bytes  = 3.0 * rows * cols * sizeof(float);  /* read A, read B, write A */
    double gb_per_s = bytes / (ms * 1e6);

    mat_free(&A); mat_free(&B);
    return (BenchResult){ name, gflops, ms, gb_per_s };
}

/* =========================================================================
 * Correctness check — compare SIMD output against reference scalar
 * ====================================================================== */

/**
 * scalar_matmul_ref — naive triple-loop mat_mul for correctness checking.
 * Always scalar regardless of SIMD flags.
 */
static void scalar_matmul_ref(const Matrix *A, const Matrix *B, Matrix *out) {
    int M = A->rows, K = A->cols, N = B->cols;
    mat_zero(out);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A->data[i * K + k];
            if (a == 0.0f) continue;
            for (int j = 0; j < N; j++)
                out->data[i * N + j] += a * B->data[k * N + j];
        }
}

static int check_close(const Matrix *got, const Matrix *ref,
                        float tol, const char *label) {
    int n = got->rows * got->cols;
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(got->data[i] - ref->data[i]);
        if (err > max_err) max_err = err;
    }
    int ok = max_err <= tol;
    printf("  %-40s max_err = %.2e  %s\n",
           label, (double)max_err, ok ? "OK" : "FAIL");
    return ok;
}

static void run_correctness_checks(void) {
    printf("=== Correctness checks ===\n");
    srand(42);

    /* mat_mul */
    {
        int M=64, K=128, N=64;
        Matrix A   = mat_alloc(M, K, 0); mat_randn(&A, 1.0f);
        Matrix B   = mat_alloc(K, N, 0); mat_randn(&B, 1.0f);
        Matrix got = mat_alloc(M, N, 0);
        Matrix ref = mat_alloc(M, N, 0);
        mat_mul(&A, &B, &got);
        scalar_matmul_ref(&A, &B, &ref);
        check_close(&got, &ref, 1e-3f, "mat_mul [64×128] × [128×64]");
        mat_free(&A); mat_free(&B); mat_free(&got); mat_free(&ref);
    }
    /* Non-multiple-of-8 dimensions */
    {
        int M=33, K=97, N=51;
        Matrix A   = mat_alloc(M, K, 0); mat_randn(&A, 0.1f);
        Matrix B   = mat_alloc(K, N, 0); mat_randn(&B, 0.1f);
        Matrix got = mat_alloc(M, N, 0);
        Matrix ref = mat_alloc(M, N, 0);
        mat_mul(&A, &B, &got);
        scalar_matmul_ref(&A, &B, &ref);
        check_close(&got, &ref, 1e-3f, "mat_mul [33×97] × [97×51] (odd dims)");
        mat_free(&A); mat_free(&B); mat_free(&got); mat_free(&ref);
    }
    /* mat_mul_T */
    {
        int M=64, K=64, N=64;
        Matrix A   = mat_alloc(M, K, 0); mat_randn(&A, 1.0f);
        Matrix B   = mat_alloc(N, K, 0); mat_randn(&B, 1.0f);
        Matrix got = mat_alloc(M, N, 0);
        Matrix ref = mat_alloc(M, N, 0);
        mat_mul_T(&A, &B, &got);
        /* Reference: manually transpose B and use scalar_matmul_ref */
        Matrix Bt = mat_alloc(K, N, 0);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < K; j++)
                Bt.data[j * N + i] = B.data[i * K + j];
        scalar_matmul_ref(&A, &Bt, &ref);
        check_close(&got, &ref, 1e-3f, "mat_mul_T [64×64] × [64×64]ᵀ");
        mat_free(&A); mat_free(&B); mat_free(&Bt); mat_free(&got); mat_free(&ref);
    }
    /* Larger shapes */
    {
        int M=128, K=512, N=512;
        Matrix A   = mat_alloc(M, K, 0); mat_randn(&A, 0.1f);
        Matrix B   = mat_alloc(K, N, 0); mat_randn(&B, 0.1f);
        Matrix got = mat_alloc(M, N, 0);
        Matrix ref = mat_alloc(M, N, 0);
        mat_mul(&A, &B, &got);
        scalar_matmul_ref(&A, &B, &ref);
        check_close(&got, &ref, 5e-3f, "mat_mul [128×512] × [512×512]");
        mat_free(&A); mat_free(&B); mat_free(&got); mat_free(&ref);
    }
    printf("\n");
}

/* =========================================================================
 * Main
 * ====================================================================== */

static void print_result(const BenchResult *r) {
    printf("  %-48s  %6.2f ms  %6.2f GFLOP/s  %5.1f GB/s\n",
           r->name, r->ms_per_call, r->gflops, r->gb_per_s);
}

int main(void) {
    printf("=== llm.c mat_mul benchmark ===\n");
    printf("SIMD level: %s\n\n", SIMD_LEVEL_STR);

    run_correctness_checks();

    printf("=== Benchmarks (shapes from a 50M-param transformer) ===\n");
    printf("  d_model=512, n_heads=8, head_dim=64, ffn_hidden=1536, seq=64\n\n");
    printf("  %-48s  %8s  %14s  %9s\n",
           "Shape", "ms/call", "GFLOP/s", "GB/s");
    printf("  %s\n", "------------------------------------------------------------"
                     "--------------------");

    int iters = 200;

    /* Q projection: [seq × d_model] × [d_model × (n_heads*head_dim)] */
    BenchResult r;

    r = bench_matmul("QKV proj [64x512]x[512x512]", 64, 512, 512, iters);
    print_result(&r);

    r = bench_matmul("Attn out [64x512]x[512x512]", 64, 512, 512, iters);
    print_result(&r);

    r = bench_matmul_T("Scores [64x64]x[64x64]T (per head)", 64, 64, 64, iters * 10);
    print_result(&r);

    r = bench_matmul("FFN gate [64x512]x[512x1536]", 64, 512, 1536, iters);
    print_result(&r);

    r = bench_matmul("FFN down [64x1536]x[1536x512]", 64, 1536, 512, iters);
    print_result(&r);

    r = bench_matmul("LM head [64x512]x[512x32000]", 64, 512, 32000, 50);
    print_result(&r);

    printf("\n--- RMSNorm ---\n");
    r = bench_rms_norm("RMSNorm [64x512]",  64, 512,  iters * 20); print_result(&r);
    r = bench_rms_norm("RMSNorm [512x512]", 512, 512, iters * 5);  print_result(&r);

    printf("\n--- Residual add ---\n");
    r = bench_add("add_inplace [64x512]",  64,  512, iters * 100); print_result(&r);
    r = bench_add("add_inplace [512x512]", 512, 512, iters * 20);  print_result(&r);

    printf("\nDone.\n");
    return 0;
}
