/**
 * matrix.h
 *
 * Core matrix and tensor types plus all math primitives.
 *
 * Design decisions:
 *   - Row-major storage (C default) — good for cache when walking rows.
 *   - Separate forward-value and gradient arrays, allocated together so a
 *     single free() cleans up both.
 *   - All ops take explicit output pointers; no hidden allocation inside ops.
 *     Callers own memory; this keeps allocations visible and controllable.
 *
 * Naming convention:
 *   mat_*   — operations on Matrix (2-D)
 *   vec_*   — operations on 1-D float arrays (used for biases, norms, etc.)
 */

#ifndef LLM_MATRIX_H
#define LLM_MATRIX_H

#include <stddef.h>

/* ---------------------------------------------------------------------------
 * Matrix
 *
 * Represents a 2-D array of floats stored in row-major order.
 *   element (r, c)  =  data[r * cols + c]
 *
 * `grad` mirrors the layout of `data` and holds the accumulated gradient
 * during backprop.  It is NULL for non-parameter tensors (activations that
 * don't need persistent gradients between steps).
 * ------------------------------------------------------------------------- */
typedef struct {
    float  *data;   /* Forward values — shape [rows × cols]                  */
    float  *grad;   /* Gradient buffer — same shape; NULL if not needed       */
    int     rows;
    int     cols;
} Matrix;


/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */

/**
 * mat_alloc — allocate a matrix with uninitialised data.
 * @with_grad: if non-zero, allocate and zero-initialise the grad buffer.
 */
Matrix mat_alloc(int rows, int cols, int with_grad);

/**
 * mat_free — release all memory owned by the matrix.
 * Zeroes the struct fields so dangling-pointer bugs surface quickly.
 */
void mat_free(Matrix *m);

/** mat_zero — fill data buffer with 0.0f */
void mat_zero(Matrix *m);

/** mat_zero_grad — fill grad buffer with 0.0f (safe to call if grad==NULL) */
void mat_zero_grad(Matrix *m);

/**
 * mat_randn — fill data with samples from N(0, std).
 * Used for weight initialisation (pass std = sqrt(2/fan_in) for He init).
 */
void mat_randn(Matrix *m, float std);


/* ---------------------------------------------------------------------------
 * Core linear-algebra operations
 * All ops write results into a pre-allocated `out` matrix.
 * ------------------------------------------------------------------------- */

/**
 * mat_mul — general matrix multiply:  out = A × B
 *   A: [M × K],  B: [K × N],  out: [M × N]
 * The hot path — will be SIMD-accelerated in matrix.c.
 */
void mat_mul(const Matrix *A, const Matrix *B, Matrix *out);

/**
 * mat_mul_add — fused multiply-add:  out = A × B + bias
 *   bias is a 1-D array of length N broadcast across rows.
 */
void mat_mul_add(const Matrix *A, const Matrix *B,
                 const float *bias, Matrix *out);

/**
 * mat_mul_T — multiply A by the transpose of B:  out = A × Bᵀ
 *   A: [M × K],  B: [N × K] (stored as-is, transposed logically),  out: [M × N]
 * Used frequently in attention (Q × Kᵀ).
 */
void mat_mul_T(const Matrix *A, const Matrix *B, Matrix *out);

/**
 * mat_add_inplace — element-wise add:  A += B
 * A and B must have identical shapes.
 * Used for residual connections.
 */
void mat_add_inplace(Matrix *A, const Matrix *B);

/**
 * mat_scale — scalar multiply in-place:  A *= scalar
 */
void mat_scale(Matrix *A, float scalar);

/**
 * mat_copy — copy data from src into dst (shapes must match).
 */
void mat_copy(Matrix *dst, const Matrix *src);


/* ---------------------------------------------------------------------------
 * Activation functions (element-wise, in-place)
 * ------------------------------------------------------------------------- */

/** softmax_rows — apply softmax to each row independently (for attention). */
void softmax_rows(Matrix *m);

/**
 * swiglu — SwiGLU gated activation used in the FFN:
 *   out[i] = gate[i] * silu(up[i])
 *   where silu(x) = x * sigmoid(x)
 *
 * `gate` and `up` are both [rows × cols]; result written to `out` [rows × cols].
 * SwiGLU outperforms ReLU and GELU in practice (used in LLaMA, PaLM).
 */
void swiglu(const Matrix *gate, const Matrix *up, Matrix *out);

/**
 * silu_inplace — apply SiLU (Swish) element-wise in-place.
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
void silu_inplace(Matrix *m);


/* ---------------------------------------------------------------------------
 * Normalisation
 * ------------------------------------------------------------------------- */

/**
 * rms_norm — Root Mean Square Layer Normalisation (no bias).
 *   out[i] = (x[i] / rms(x)) * weight[i]
 *
 * Cheaper than LayerNorm (no mean subtraction); used in LLaMA, Mistral.
 * `weight` is a learned scale vector of length `cols`.
 * `eps` prevents division by zero (typically 1e-5).
 *
 * Operates row-by-row: each row of `in` is normalised independently.
 */
void rms_norm(const Matrix *in, const float *weight, float eps, Matrix *out);

/**
 * rms_norm_backward — backprop through rms_norm.
 * `d_out` is the incoming gradient [rows × cols].
 * Accumulates into `d_in` and `d_weight`.
 */
void rms_norm_backward(const Matrix *in,  const float *weight, float eps,
                       const Matrix *d_out,
                       Matrix *d_in, float *d_weight);


/* ---------------------------------------------------------------------------
 * Embedding helpers
 * ------------------------------------------------------------------------- */

/**
 * embed_lookup — gather embedding rows for a token sequence.
 *   table: [vocab_size × d_model]
 *   tokens: integer array of length seq_len
 *   out:  [seq_len × d_model]
 */
void embed_lookup(const Matrix *table, const int *tokens,
                  int seq_len, Matrix *out);

/**
 * embed_backward — scatter gradients back to the embedding table.
 * Accumulates (not overwrites) into table->grad.
 */
void embed_backward(Matrix *table, const int *tokens,
                    int seq_len, const Matrix *d_out);


/* ---------------------------------------------------------------------------
 * Utilities
 * ------------------------------------------------------------------------- */

/** mat_print — print a matrix to stdout (for debugging small tensors). */
void mat_print(const char *name, const Matrix *m);

/**
 * mat_l2_norm — compute the L2 norm of the data buffer.
 * Used for gradient clipping.
 */
float mat_l2_norm(const Matrix *m);

#endif /* LLM_MATRIX_H */