/**
 * simd.h
 *
 * SIMD capability detection and a thin portability layer.
 *
 * We detect at compile time what is available and expose a single set of
 * macros the implementation files use.  The fallback is always plain C so
 * the code compiles and runs correctly on any architecture.
 *
 * Capability tiers (best to worst):
 *   SIMD_AVX512   — AVX-512F + VNNI  (16 × 256-bit lanes = 16 floats/op)
 *   SIMD_AVX2FMA  — AVX2 + FMA       ( 8 × 256-bit lanes =  8 floats/op)
 *   SIMD_AVX2     — AVX2 only        ( 8 lanes, no fused multiply-add)
 *   SIMD_SSE4     — SSE4.1           ( 4 lanes)
 *   SIMD_NONE     — scalar fallback
 *
 * How to use:
 *   #include "simd.h"
 *   #if defined(SIMD_AVX2FMA)
 *       // fast path
 *   #else
 *       // scalar fallback
 *   #endif
 */

#ifndef LLM_SIMD_H
#define LLM_SIMD_H

/* ---------------------------------------------------------------------- */
/* Detect available ISA                                                     */
/* ---------------------------------------------------------------------- */

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
#  define SIMD_AVX512
#  define SIMD_AVX2FMA          /* AVX-512 implies AVX2+FMA */
#  define SIMD_AVX2
#  define SIMD_LEVEL_STR  "AVX-512F+VNNI"
#  define SIMD_FLOAT_WIDTH 16   /* floats processed per SIMD op */
#elif defined(__AVX2__) && defined(__FMA__)
#  define SIMD_AVX2FMA
#  define SIMD_AVX2
#  define SIMD_LEVEL_STR  "AVX2+FMA"
#  define SIMD_FLOAT_WIDTH 8
#elif defined(__AVX2__)
#  define SIMD_AVX2
#  define SIMD_LEVEL_STR  "AVX2"
#  define SIMD_FLOAT_WIDTH 8
#elif defined(__SSE4_1__)
#  define SIMD_SSE4
#  define SIMD_LEVEL_STR  "SSE4.1"
#  define SIMD_FLOAT_WIDTH 4
#else
#  define SIMD_NONE
#  define SIMD_LEVEL_STR  "scalar"
#  define SIMD_FLOAT_WIDTH 1
#endif

/* ---------------------------------------------------------------------- */
/* Include the right headers                                                */
/* ---------------------------------------------------------------------- */

#if defined(SIMD_AVX512) || defined(SIMD_AVX2FMA) || defined(SIMD_AVX2)
#  include <immintrin.h>
#elif defined(SIMD_SSE4)
#  include <smmintrin.h>   /* SSE4.1 */
#  include <emmintrin.h>   /* SSE2   */
#endif

/* ---------------------------------------------------------------------- */
/* Alignment helpers                                                        */
/* ---------------------------------------------------------------------- */

/* Align a pointer up to the next multiple of `align` (must be power-of-2) */
#define ALIGN_UP(ptr, align) \
    (((uintptr_t)(ptr) + (align) - 1) & ~((uintptr_t)(align) - 1))

/* Preferred allocation alignment for SIMD (32 bytes = 256-bit AVX2) */
#define SIMD_ALIGN 32

/* Aligned malloc — returns NULL on failure; caller must use aligned_free() */
static inline void *aligned_malloc(size_t size) {
    void *p = NULL;
#if defined(_MSC_VER)
    p = _aligned_malloc(size, SIMD_ALIGN);
#else
    if (posix_memalign(&p, SIMD_ALIGN, size) != 0) p = NULL;
#endif
    return p;
}

static inline void aligned_free(void *p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ---------------------------------------------------------------------- */
/* Horizontal sum helpers                                                   */
/* ---------------------------------------------------------------------- */

#if defined(SIMD_AVX2FMA) || defined(SIMD_AVX2)
/**
 * hsum256 — sum all 8 floats in a __m256 register into one scalar.
 *
 * Technique: fold the 256-bit register in half twice, then use SSE3
 * hadd to collapse the final 128 bits.
 */
static inline float hsum256(__m256 v) {
    /* Add upper 128 to lower 128 */
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    /* Pairwise add within 128 bits */
    __m128 shuf = _mm_movehdup_ps(sum);          /* [1,1,3,3]             */
    __m128 s2   = _mm_add_ps(sum, shuf);          /* [0+1, _, 2+3, _]     */
    shuf = _mm_movehl_ps(shuf, s2);               /* move [2+3,_] to low  */
    s2   = _mm_add_ss(s2, shuf);
    return _mm_cvtss_f32(s2);
}
#endif

#if defined(SIMD_SSE4)
/**
 * hsum128 — sum all 4 floats in a __m128 register into one scalar.
 */
static inline float hsum128(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 s2   = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, s2);
    s2   = _mm_add_ss(s2, shuf);
    return _mm_cvtss_f32(s2);
}
#endif

#endif /* LLM_SIMD_H */
