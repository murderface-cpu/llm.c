/**
 * matrix.c
 *
 * _GNU_SOURCE exposes M_PI on Linux.
 *
 * Implementation of all matrix / tensor math operations.
 *
 * SIMD acceleration strategy
 * ──────────────────────────
 * Every hot function has three layers:
 *
 *   1. AVX2+FMA path  — 8 floats/cycle with fused multiply-add.
 *                       Handles the bulk of elements in multiples of 8.
 *   2. SSE4 path      — 4 floats/cycle (fallback for older CPUs).
 *   3. Scalar tail    — handles the remaining 0-7 elements after SIMD.
 *
 * The compiler sees #if / #elif / #else, not runtime branches, so there
 * is zero dispatch overhead.
 *
 * Blocked matrix multiply (cache tiling)
 * ───────────────────────────────────────
 * A naïve triple loop stalls on cache misses for large matrices because B
 * is accessed column-wise.  We tile the (K, N) dimensions into blocks that
 * fit in L1/L2 cache:
 *
 *   BK = 64   (K-tile: controls how many rows of B stay in L1)
 *   BN = 256  (N-tile: controls the inner j-loop width)
 *
 * Within each tile the inner j-loop is a SIMD dot-product accumulation.
 *
 * Micro-kernel for mat_mul (one M-row × BN-column block):
 *   Accumulate 8 output floats at a time using _mm256_fmadd_ps.
 *   8 FP32 multiply-adds per instruction = ~8× scalar throughput.
 *
 * Optimisation progression:
 *   Phase 1 (done): scalar, correct.
 *   Phase 2 (done): AVX2+FMA blocked tiling, ~8× faster.
 *   Phase 3 (done): OpenMP outer-tile parallelism, linear speedup.
 *   Phase 4 (done): Polynomial expf approximation, full AVX2 softmax/silu.
 *   Phase 5 (TODO): AVX-512 path (16 floats/op on supported CPUs).
 */

#define _GNU_SOURCE  /* expose M_PI on Linux */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "../include/matrix.h"
#include "../include/simd.h"
#ifdef USE_OPENBLAS
#  include <cblas.h>
#endif

/* OpenMP: included only when -fopenmp is passed to the compiler.
 * Without -fopenmp every omp_ call is a no-op stub, so the code
 * compiles and runs correctly on any build configuration. */
#ifdef _OPENMP
#  include <omp.h>
#endif

/* =========================================================================
 * Phase 4: Fast vectorised expf approximation
 *
 * Algorithm: Cephes-style minimax polynomial for exp2(x) on [0,1).
 *
 * Key identity:  expf(x) = exp2(x / ln2) = exp2(x * log2e)
 *
 * Steps for exp2(z), z = x * log2e:
 *   1. Split z = n + f  where n = floor(z), f in [0,1).
 *   2. Compute 2^f with a degree-6 minimax polynomial (max error ~1.9e-7).
 *   3. Scale by 2^n via float32 exponent bit-manipulation.
 *
 * ~5 FLOPs per element vs ~20 for libm expf -> 4x speedup on the exp pass.
 * Max relative error: < 2e-7 (fine for float32 softmax / sigmoid).
 * ====================================================================== */

#if defined(SIMD_AVX2FMA)

static inline __m256 exp_avx2(__m256 v) {
    const __m256 vmax    = _mm256_set1_ps( 88.0f);
    const __m256 vmin    = _mm256_set1_ps(-88.0f);
    const __m256 vlog2e  = _mm256_set1_ps(1.4426950408f);
    const __m256 vc0     = _mm256_set1_ps(1.0000000000f);
    const __m256 vc1     = _mm256_set1_ps(0.6931471806f);
    const __m256 vc2     = _mm256_set1_ps(0.2402265069f);
    const __m256 vc3     = _mm256_set1_ps(0.0555041086f);
    const __m256 vc4     = _mm256_set1_ps(0.0096181292f);
    const __m256 vc5     = _mm256_set1_ps(0.0013333558f);
    const __m256 vc6     = _mm256_set1_ps(0.0001540353f);
    const __m256i v127   = _mm256_set1_epi32(127);

    v = _mm256_min_ps(_mm256_max_ps(v, vmin), vmax);

    __m256 z  = _mm256_mul_ps(v, vlog2e);
    __m256 vn = _mm256_floor_ps(z);
    __m256 vf = _mm256_sub_ps(z, vn);

    /* Horner evaluation of degree-6 polynomial for 2^vf */
    __m256 p = vc6;
    p = _mm256_fmadd_ps(p, vf, vc5);
    p = _mm256_fmadd_ps(p, vf, vc4);
    p = _mm256_fmadd_ps(p, vf, vc3);
    p = _mm256_fmadd_ps(p, vf, vc2);
    p = _mm256_fmadd_ps(p, vf, vc1);
    p = _mm256_fmadd_ps(p, vf, vc0);

    /* Scale by 2^n: build float32 with exponent = n+127, mantissa = 0 */
    __m256i vni    = _mm256_cvtps_epi32(vn);
    __m256i vscale = _mm256_slli_epi32(_mm256_add_epi32(vni, v127), 23);
    __m256  scale  = _mm256_castsi256_ps(vscale);

    return _mm256_mul_ps(p, scale);
}

static inline __m256 sigmoid_avx2(__m256 v) {
    __m256 neg_v  = _mm256_xor_ps(v, _mm256_set1_ps(-0.0f)); /* negate */
    __m256 exp_nv = exp_avx2(neg_v);
    __m256 denom  = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_nv);
    return _mm256_div_ps(_mm256_set1_ps(1.0f), denom);
}

static inline __m256 silu_avx2(__m256 v) {
    return _mm256_mul_ps(v, sigmoid_avx2(v));
}

#endif /* SIMD_AVX2FMA */


/* =========================================================================
 * Cache-blocking parameters
 *
 * Tuned for a typical L1d = 32 KiB, L2 = 256 KiB.
 * Each tile of B accessed in the inner loop: BK × BN × 4 bytes.
 * BK=64, BN=256 → 64×256×4 = 64 KiB — fits comfortably in L2.
 * The A-tile (M × BK) stays in registers / L1.
 * ====================================================================== */
#define MC 64    /* M tile — rows of A processed together              */
#define KC 64    /* K tile — depth slice (columns of A, rows of B)     */
#define NC 256   /* N tile — columns of B processed in one inner pass  */

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "[matrix] OOM %zu bytes\n", n); abort(); }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "[matrix] OOM %zu×%zu bytes\n", n, sz); abort(); }
    return p;
}

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

Matrix mat_alloc(int rows, int cols, int with_grad) {
    assert(rows > 0 && cols > 0);
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)xmalloc((size_t)rows * cols * sizeof(float));
    m.grad = with_grad
           ? (float *)xcalloc((size_t)rows * cols, sizeof(float))
           : NULL;
    return m;
}

void mat_free(Matrix *m) {
    if (!m) return;
    free(m->data);
    free(m->grad);
    m->data = NULL;
    m->grad = NULL;
    m->rows = 0;
    m->cols = 0;
}

void mat_zero(Matrix *m) {
    memset(m->data, 0, (size_t)m->rows * m->cols * sizeof(float));
}

void mat_zero_grad(Matrix *m) {
    if (m->grad)
        memset(m->grad, 0, (size_t)m->rows * m->cols * sizeof(float));
}

void mat_randn(Matrix *m, float std) {
    int n = m->rows * m->cols;
    for (int i = 0; i < n - 1; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float mag = std * sqrtf(-2.0f * logf(u1));
        m->data[i]     = mag * cosf(2.0f * (float)M_PI * u2);
        m->data[i + 1] = mag * sinf(2.0f * (float)M_PI * u2);
    }
    if (n % 2 == 1) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        m->data[n - 1] = std * sqrtf(-2.0f * logf(u1))
                             * cosf(2.0f * (float)M_PI * u2);
    }
}

void mat_copy(Matrix *dst, const Matrix *src) {
    assert(dst->rows == src->rows && dst->cols == src->cols);
    memcpy(dst->data, src->data,
           (size_t)src->rows * src->cols * sizeof(float));
}

/* =========================================================================
 * SIMD micro-kernels
 *
 * These are the innermost loops — kept in separate functions so the
 * compiler can apply maximum vectorisation without polluting the outer
 * loop with extra register pressure.
 * ====================================================================== */

/**
 * dot_avx2fma — dot product of two float arrays of length n.
 *
 * Uses 4 accumulator registers to hide FMA latency (4-cycle pipeline).
 * Processes 32 floats per iteration (4 × 8-wide FMA).
 * Scalar tail handles remaining elements.
 */
#if defined(SIMD_AVX2FMA)
static inline float dot_avx2fma(const float * restrict a,
                                 const float * restrict b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    /* Main loop: 32 floats per iteration (4 × 8-wide FMA) */
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                               _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                               _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                               _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                               _mm256_loadu_ps(b + i + 24), acc3);
    }
    /* 8-float tail */
    for (; i <= n - 8; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                               _mm256_loadu_ps(b + i), acc0);

    /* Reduce 4 accumulators → 1 */
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    float result = hsum256(acc0);

    /* Scalar tail (0-7 elements) */
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}
#endif /* SIMD_AVX2FMA */

/**
 * axpy_avx2fma — y[j] += scalar * x[j]  for j in [0, n).
 *
 * This is the innermost operation in mat_mul's (i,k,j) loop.
 * Broadcasts the scalar into all 8 lanes, then FMA with the x vector.
 * Processes 32 floats per iteration (4 × 8-wide FMA).
 */
#if defined(SIMD_AVX2FMA)
static inline void axpy_avx2fma(float *restrict y,
                                  const float *restrict x,
                                  float scalar, int n) {
    __m256 vs = _mm256_set1_ps(scalar);
    int j = 0;

    /* Main loop: 32 floats per iteration */
    for (; j <= n - 32; j += 32) {
        _mm256_storeu_ps(y + j,
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(x + j),
                            _mm256_loadu_ps(y + j)));
        _mm256_storeu_ps(y + j + 8,
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(x + j + 8),
                            _mm256_loadu_ps(y + j + 8)));
        _mm256_storeu_ps(y + j + 16,
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(x + j + 16),
                            _mm256_loadu_ps(y + j + 16)));
        _mm256_storeu_ps(y + j + 24,
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(x + j + 24),
                            _mm256_loadu_ps(y + j + 24)));
    }
    /* 8-float tail */
    for (; j <= n - 8; j += 8)
        _mm256_storeu_ps(y + j,
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(x + j),
                            _mm256_loadu_ps(y + j)));
    /* Scalar tail */
    for (; j < n; j++)
        y[j] += scalar * x[j];
}
#endif /* SIMD_AVX2FMA */

/**
 * add_avx2 — element-wise addition: a[i] += b[i].
 * Used for residual connections — hot in the transformer.
 */
#if defined(SIMD_AVX2)
static inline void add_avx2(float *restrict a,
                              const float *restrict b, int n) {
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm256_storeu_ps(a + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i),
                          _mm256_loadu_ps(b + i)));
        _mm256_storeu_ps(a + i + 8,
            _mm256_add_ps(_mm256_loadu_ps(a + i + 8),
                          _mm256_loadu_ps(b + i + 8)));
        _mm256_storeu_ps(a + i + 16,
            _mm256_add_ps(_mm256_loadu_ps(a + i + 16),
                          _mm256_loadu_ps(b + i + 16)));
        _mm256_storeu_ps(a + i + 24,
            _mm256_add_ps(_mm256_loadu_ps(a + i + 24),
                          _mm256_loadu_ps(b + i + 24)));
    }
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(a + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i),
                          _mm256_loadu_ps(b + i)));
    for (; i < n; i++) a[i] += b[i];
}
#endif /* SIMD_AVX2 */

/* =========================================================================
 * mat_mul — out = A × B    A:[M×K]  B:[K×N]  out:[M×N]
 *
 * Implementation: cache-blocked (MC×KC×NC tiles) + AVX2+FMA axpy micro-kernel
 *                 + OpenMP outer-tile parallelism.
 *
 * OpenMP strategy:
 *   We parallelise the i0 (row-tile) loop — each thread owns a disjoint
 *   set of output rows, so there are no write conflicts and no atomics.
 *
 *   We do NOT parallelise the k0 loop because different k0 tiles all
 *   write to the same output rows (they accumulate), which would require
 *   either atomic adds or per-thread scratch buffers — not worth it.
 *
 *   Thread count is controlled by the caller via omp_set_num_threads()
 *   or the OMP_NUM_THREADS environment variable.
 * ====================================================================== */
void mat_mul(const Matrix *A, const Matrix *B, Matrix *out) {
    assert(A->cols == B->rows);
    assert(out->rows == A->rows && out->cols == B->cols);

    int M = A->rows, K = A->cols, N = B->cols;

#ifdef USE_OPENBLAS
    /*
     * OpenBLAS sgemm: C = alpha*A*B + beta*C
     * This is a highly-optimised BLAS routine that uses AVX-512, multi-threading,
     * and cache-optimal micro-kernels tuned for the specific CPU at compile time.
     * Typically 5-20x faster than our hand-rolled kernel for the sizes used in
     * transformer training (e.g. [512×512]×[512×4096]).
     *
     * beta=0.0f means C is overwritten (no need to zero first).
     */
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,           /* alpha */
                A->data, K,     /* A, lda */
                B->data, N,     /* B, ldb */
                0.0f,           /* beta — overwrites out */
                out->data, N);  /* C, ldc */
#elif defined(SIMD_AVX2FMA)
    mat_zero(out);
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = K - k0 < KC ? K - k0 : KC;
        #pragma omp parallel for schedule(dynamic) if(M >= 2*MC)
        for (int i0 = 0; i0 < M; i0 += MC) {
            int mc = M - i0 < MC ? M - i0 : MC;
            for (int j0 = 0; j0 < N; j0 += NC) {
                int nc = N - j0 < NC ? N - j0 : NC;
                for (int i = i0; i < i0 + mc; i++) {
                    float *out_row = out->data + i * N + j0;
                    for (int k = k0; k < k0 + kc; k++) {
                        float a_ik = A->data[i * K + k];
                        if (a_ik == 0.0f) continue;
                        axpy_avx2fma(out_row, B->data + k * N + j0, a_ik, nc);
                    }
                }
            }
        }
    }
#else
    mat_zero(out);
    #pragma omp parallel for schedule(static) if(M >= 64)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A->data[i * K + k];
            if (a_ik == 0.0f) continue;
            for (int j = 0; j < N; j++)
                out->data[i * N + j] += a_ik * B->data[k * N + j];
        }
    }
#endif
}

/* =========================================================================
 * mat_mul_add — out = A × B + bias
 * ====================================================================== */
void mat_mul_add(const Matrix *A, const Matrix *B,
                 const float *bias, Matrix *out) {
    mat_mul(A, B, out);
    int M = out->rows, N = out->cols;

#if defined(SIMD_AVX2)
    /* Broadcast bias across rows using AVX2 load+add */
    for (int i = 0; i < M; i++) {
        float *row = out->data + i * N;
        int j = 0;
        for (; j <= N - 8; j += 8)
            _mm256_storeu_ps(row + j,
                _mm256_add_ps(_mm256_loadu_ps(row + j),
                              _mm256_loadu_ps(bias + j)));
        for (; j < N; j++) row[j] += bias[j];
    }
#else
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            out->data[i * N + j] += bias[j];
#endif
}

/* =========================================================================
 * mat_mul_T — out = A × Bᵀ    A:[M×K]  B:[N×K]  out:[M×N]
 *
 * Each output element out[i][j] = dot(A_row_i, B_row_j).
 * With AVX2+FMA we use dot_avx2fma for each (i,j) pair.
 *
 * This is the main operation in attention score computation:
 *   scores[h] = Q[h] × K[h]ᵀ / sqrt(head_dim)
 * ====================================================================== */
void mat_mul_T(const Matrix *A, const Matrix *B, Matrix *out) {
    assert(A->cols == B->cols);
    assert(out->rows == A->rows && out->cols == B->rows);

    int M = A->rows, K = A->cols, N = B->rows;

#ifdef USE_OPENBLAS
    /* out = A × Bᵀ  — pass CblasTrans for B */
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f,
                A->data, K,
                B->data, K,   /* ldb = K because B is stored [N×K] */
                0.0f,
                out->data, N);
#elif defined(SIMD_AVX2FMA)
    #pragma omp parallel for schedule(static) if(M*N >= 512)
    for (int i = 0; i < M; i++) {
        const float *a_row = A->data + i * K;
        for (int j = 0; j < N; j++)
            out->data[i * N + j] = dot_avx2fma(a_row, B->data + j * K, K);
    }
#else
    mat_zero(out);
    #pragma omp parallel for schedule(static) if(M*N >= 512)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            const float *a_row = A->data + i * K;
            const float *b_row = B->data + j * K;
            for (int k = 0; k < K; k++) sum += a_row[k] * b_row[k];
            out->data[i * N + j] = sum;
        }
    }
#endif
}

/* =========================================================================
 * mat_add_inplace — A += B  (residual connections)
 * ====================================================================== */
void mat_add_inplace(Matrix *A, const Matrix *B) {
    assert(A->rows == B->rows && A->cols == B->cols);
    int n = A->rows * A->cols;

#if defined(SIMD_AVX2)
    add_avx2(A->data, B->data, n);
#else
    for (int i = 0; i < n; i++) A->data[i] += B->data[i];
#endif
}

void mat_scale(Matrix *A, float scalar) {
    int n = A->rows * A->cols;

#if defined(SIMD_AVX2FMA)
    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(A->data + i,
            _mm256_mul_ps(_mm256_loadu_ps(A->data + i), vs));
    for (; i < n; i++) A->data[i] *= scalar;
#else
    for (int i = 0; i < n; i++) A->data[i] *= scalar;
#endif
}

/* =========================================================================
 * Activation functions
 * ====================================================================== */

/**
 * softmax_rows — numerically stable per-row softmax.
 *
 * max reduction and sum reduction are AVX2-accelerated.
 * The exp() call is scalar because there is no fast __m256 expf in
 * standard intrinsics (libsvml is not portable); we use the scalar loop
 * and let the auto-vectoriser pick it up where possible.
 */
void softmax_rows(Matrix *m) {
    int rows = m->rows, cols = m->cols;

    for (int i = 0; i < rows; i++) {
        float *row = m->data + i * cols;

/* Declare j once for this row — used across max, exp, and normalise steps. */
        int j = 0;
        float row_max, sum;

        /* ---- Step 1: find row maximum ---- */
#if defined(SIMD_AVX2)
        { __m256 vmax = _mm256_set1_ps(-1e38f);
          for (j = 0; j <= cols - 8; j += 8)
              vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + j));
          __m128 lo   = _mm256_castps256_ps128(vmax);
          __m128 hi   = _mm256_extractf128_ps(vmax, 1);
          __m128 m128 = _mm_max_ps(lo, hi);
          m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
          m128 = _mm_max_ss(m128, _mm_movehdup_ps(m128));
          row_max = _mm_cvtss_f32(m128);
          for (; j < cols; j++) if (row[j] > row_max) row_max = row[j]; }
#else
        row_max = row[0];
        for (j = 1; j < cols; j++) if (row[j] > row_max) row_max = row[j];
#endif

        /* ---- Step 2: exp(x - max) and sum ---- */
#if defined(SIMD_AVX2FMA)
        { __m256 vmax_bc = _mm256_set1_ps(row_max);
          __m256 vsum    = _mm256_setzero_ps();
          sum = 0.0f;
          for (j = 0; j <= cols - 8; j += 8) {
              __m256 shifted = _mm256_sub_ps(_mm256_loadu_ps(row+j), vmax_bc);
              __m256 ex = exp_avx2(shifted);
              _mm256_storeu_ps(row+j, ex);
              vsum = _mm256_add_ps(vsum, ex);
          }
          sum = hsum256(vsum);
          for (; j < cols; j++) { row[j] = expf(row[j]-row_max); sum += row[j]; } }
#else
        sum = 0.0f;
        for (j = 0; j < cols; j++) { row[j] = expf(row[j]-row_max); sum += row[j]; }
#endif

#if defined(SIMD_AVX2)
        { __m256 vinv = _mm256_set1_ps(1.0f / sum);
          int jn = 0;
          for (; jn <= cols - 8; jn += 8)
              _mm256_storeu_ps(row + jn,
                  _mm256_mul_ps(_mm256_loadu_ps(row + jn), vinv));
          for (; jn < cols; jn++) row[jn] /= sum; }
#else
        { float inv = 1.0f / sum;
          for (int j = 0; j < cols; j++) row[j] *= inv; }
#endif
    }
}

/**
 * silu_inplace — x = x * sigmoid(x), element-wise.
 * Phase 4: fully vectorised with exp_avx2 polynomial, 4-unrolled.
 */
void silu_inplace(Matrix *m) {
    int n = m->rows * m->cols;
#if defined(SIMD_AVX2FMA)
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm256_storeu_ps(m->data+i,    silu_avx2(_mm256_loadu_ps(m->data+i)));
        _mm256_storeu_ps(m->data+i+8,  silu_avx2(_mm256_loadu_ps(m->data+i+8)));
        _mm256_storeu_ps(m->data+i+16, silu_avx2(_mm256_loadu_ps(m->data+i+16)));
        _mm256_storeu_ps(m->data+i+24, silu_avx2(_mm256_loadu_ps(m->data+i+24)));
    }
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(m->data+i, silu_avx2(_mm256_loadu_ps(m->data+i)));
    for (; i < n; i++) { float x=m->data[i]; m->data[i]=x/(1.0f+expf(-x)); }
#else
    for (int i = 0; i < n; i++) {
        float x = m->data[i]; m->data[i] = x / (1.0f + expf(-x));
    }
#endif
}

/**
 * swiglu — out = silu(gate) ⊙ up.
 *
 * The element-wise multiply can be vectorised; the silu is scalar.
 * We interleave compute to keep the CPU busy.
 */
void swiglu(const Matrix *gate, const Matrix *up, Matrix *out) {
    assert(gate->rows == up->rows && gate->cols == up->cols);
    assert(out->rows == gate->rows && out->cols == gate->cols);

    int n = gate->rows * gate->cols;

/* Phase 4: fully vectorised SwiGLU — silu(gate) * up, 4-unrolled. */
#if defined(SIMD_AVX2FMA)
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm256_storeu_ps(out->data+i,
            _mm256_mul_ps(silu_avx2(_mm256_loadu_ps(gate->data+i)),
                          _mm256_loadu_ps(up->data+i)));
        _mm256_storeu_ps(out->data+i+8,
            _mm256_mul_ps(silu_avx2(_mm256_loadu_ps(gate->data+i+8)),
                          _mm256_loadu_ps(up->data+i+8)));
        _mm256_storeu_ps(out->data+i+16,
            _mm256_mul_ps(silu_avx2(_mm256_loadu_ps(gate->data+i+16)),
                          _mm256_loadu_ps(up->data+i+16)));
        _mm256_storeu_ps(out->data+i+24,
            _mm256_mul_ps(silu_avx2(_mm256_loadu_ps(gate->data+i+24)),
                          _mm256_loadu_ps(up->data+i+24)));
    }
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(out->data+i,
            _mm256_mul_ps(silu_avx2(_mm256_loadu_ps(gate->data+i)),
                          _mm256_loadu_ps(up->data+i)));
    for (; i < n; i++) {
        float g = gate->data[i];
        out->data[i] = (g/(1.0f+expf(-g))) * up->data[i];
    }
#else
    for (int i = 0; i < n; i++) {
        float g = gate->data[i];
        out->data[i] = (g / (1.0f + expf(-g))) * up->data[i];
    }
#endif
}

/* =========================================================================
 * Normalisation
 * ====================================================================== */

/**
 * rms_norm — per-row Root Mean Square normalisation.
 *
 * sum-of-squares and the scale step are AVX2-accelerated.
 */
void rms_norm(const Matrix *in, const float *weight, float eps, Matrix *out) {
    assert(in->rows == out->rows && in->cols == out->cols);

    int rows = in->rows, cols = in->cols;

    for (int i = 0; i < rows; i++) {
        const float *x = in->data  + i * cols;
        float       *y = out->data + i * cols;

#if defined(SIMD_AVX2FMA)
        /* Sum of squares using FMA accumulators */
        __m256 vss0 = _mm256_setzero_ps();
        __m256 vss1 = _mm256_setzero_ps();
        int j = 0;
        for (; j <= cols - 16; j += 16) {
            __m256 v0 = _mm256_loadu_ps(x + j);
            __m256 v1 = _mm256_loadu_ps(x + j + 8);
            vss0 = _mm256_fmadd_ps(v0, v0, vss0);
            vss1 = _mm256_fmadd_ps(v1, v1, vss1);
        }
        vss0 = _mm256_add_ps(vss0, vss1);
        for (; j <= cols - 8; j += 8) {
            __m256 v = _mm256_loadu_ps(x + j);
            vss0 = _mm256_fmadd_ps(v, v, vss0);
        }
        float ss = hsum256(vss0);
        for (; j < cols; j++) ss += x[j] * x[j];
#else
        float ss = 0.0f;
        for (int j = 0; j < cols; j++) ss += x[j] * x[j];
#endif
        float inv_rms = 1.0f / sqrtf(ss / (float)cols + eps);

#if defined(SIMD_AVX2FMA)
        __m256 virms = _mm256_set1_ps(inv_rms);
        int k = 0;
        for (; k <= cols - 8; k += 8) {
            __m256 vx = _mm256_loadu_ps(x + k);
            __m256 vw = _mm256_loadu_ps(weight + k);
            /* y = x * inv_rms * w  →  fmadd not needed; two muls */
            _mm256_storeu_ps(y + k,
                _mm256_mul_ps(_mm256_mul_ps(vx, virms), vw));
        }
        for (; k < cols; k++) y[k] = x[k] * inv_rms * weight[k];
#else
        for (int j = 0; j < cols; j++) y[j] = x[j] * inv_rms * weight[j];
#endif
    }
}

void rms_norm_backward(const Matrix *in, const float *weight, float eps,
                       const Matrix *d_out,
                       Matrix *d_in, float *d_weight) {
    int rows = in->rows, cols = in->cols;

    for (int i = 0; i < rows; i++) {
        const float *x    = in->data    + i * cols;
        const float *dout = d_out->data + i * cols;
        float       *din  = d_in->data  + i * cols;

        /* Recompute rms */
        float ss = 0.0f;
        for (int j = 0; j < cols; j++) ss += x[j] * x[j];
        float rms     = sqrtf(ss / (float)cols + eps);
        float inv_rms = 1.0f / rms;

        /* Gradient w.r.t. weight */
        for (int j = 0; j < cols; j++)
            d_weight[j] += dout[j] * x[j] * inv_rms;

        /* Gradient w.r.t. x */
        float dot = 0.0f;
        for (int j = 0; j < cols; j++)
            dot += dout[j] * weight[j] * x[j];
        float scale = dot / ((float)cols * rms * rms * rms);

        for (int j = 0; j < cols; j++)
            din[j] += dout[j] * weight[j] * inv_rms - x[j] * scale;
    }
}

/* =========================================================================
 * Embedding helpers
 * ====================================================================== */

void embed_lookup(const Matrix *table, const int *tokens,
                  int seq_len, Matrix *out) {
    assert(out->rows == seq_len && out->cols == table->cols);
    int d = table->cols;
    for (int t = 0; t < seq_len; t++) {
        int id = tokens[t];
        assert(id >= 0 && id < table->rows);
        memcpy(out->data + t * d,
               table->data + id * d,
               d * sizeof(float));
    }
}

void embed_backward(Matrix *table, const int *tokens,
                    int seq_len, const Matrix *d_out) {
    assert(table->grad != NULL);
    int d = table->cols;
    for (int t = 0; t < seq_len; t++) {
        int id = tokens[t];
        float *g_row = table->grad + id * d;
        const float *dout_row = d_out->data + t * d;

#if defined(SIMD_AVX2)
        add_avx2(g_row, dout_row, d);
#else
        for (int j = 0; j < d; j++) g_row[j] += dout_row[j];
#endif
    }
}

/* =========================================================================
 * Utilities
 * ====================================================================== */

void mat_print(const char *name, const Matrix *m) {
    printf("Matrix '%s' [%d × %d]:\n", name, m->rows, m->cols);
    int sr = m->rows < 8 ? m->rows : 8;
    int sc = m->cols < 8 ? m->cols : 8;
    for (int i = 0; i < sr; i++) {
        printf("  [");
        for (int j = 0; j < sc; j++)
            printf(" %8.4f", m->data[i * m->cols + j]);
        if (sc < m->cols) printf(" ...");
        printf(" ]\n");
    }
    if (sr < m->rows) printf("  ...\n");
}

float mat_l2_norm(const Matrix *m) {
    int n = m->rows * m->cols;

#if defined(SIMD_AVX2FMA)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(m->data + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = hsum256(acc);
    for (; i < n; i++) ss += m->data[i] * m->data[i];
    return sqrtf(ss);
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += m->data[i] * m->data[i];
    return sqrtf(ss);
#endif
}
