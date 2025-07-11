#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "memory.h"
#include <stdint.h>

// float16 conversion helpers
uint16_t float32_to_float16(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint16_t sign = (x >> 16) & 0x8000;
    uint32_t mantissa = x & 0x7fffff;
    int exp = ((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | (mantissa >> 13);
}
float float16_to_float32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t mantissa = (h & 0x3ff) << 13;
    if (exp == 0) exp = 127 - 15 + 1;
    else exp = exp - 15 + 127;
    uint32_t x = sign | (exp << 23) | mantissa;
    float f;
    memcpy(&f, &x, sizeof(float));
    return f;
}

Tensor* create_tensor(Arena* arena, int n_dims, int* dims, DataType dtype) {
    size_t total_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        total_elements *= dims[i];
    }

    size_t element_size = 0;
    if (dtype == TENSOR_TYPE_FLOAT) {
        element_size = sizeof(float);
    } else if (dtype == TENSOR_TYPE_INT) {
        element_size = sizeof(int);
    } else if (dtype == TENSOR_TYPE_FLOAT16) {
        element_size = sizeof(uint16_t);
    }

    Tensor* t = NULL;
    if (arena) {
        t = (Tensor*)arena_alloc(arena, sizeof(Tensor));
        if (!t) { fprintf(stderr, "[ERR] arena_alloc failed for Tensor\n"); return NULL; }
        t->dims = (int*)arena_alloc(arena, n_dims * sizeof(int));
        if (!t->dims) { fprintf(stderr, "[ERR] arena_alloc failed for dims\n"); return NULL; }
        t->data = arena_alloc(arena, total_elements * element_size);
        if (!t->data) { fprintf(stderr, "[ERR] arena_alloc failed for data\n"); return NULL; }
        t->grad = arena_alloc(arena, total_elements * element_size);
        if (!t->grad) { fprintf(stderr, "[ERR] arena_alloc failed for grad\n"); return NULL; }
    } else {
        t = (Tensor*)malloc(sizeof(Tensor));
        if (!t) { fprintf(stderr, "[ERR] malloc failed for Tensor\n"); return NULL; }
        t->dims = (int*)malloc(n_dims * sizeof(int));
        if (!t->dims) {
            fprintf(stderr, "[ERR] malloc failed for dims\n");
            free(t);
            return NULL;
        }
        t->data = calloc(total_elements, element_size);
        if (!t->data) {
            fprintf(stderr, "[ERR] calloc failed for data\n");
            free(t->dims);
            free(t);
            return NULL;
        }
        t->grad = calloc(total_elements, element_size);
        if (!t->grad) {
            fprintf(stderr, "[ERR] calloc failed for grad\n");
            free(t->data);
            free(t->dims);
            free(t);
            return NULL;
        }
    }
    if (!t || !t->dims || !t->data || !t->grad) {
        fprintf(stderr, "[ERR] tensor allocation failed\n");
        return NULL;
    }
    memset(t->data, 0, total_elements * element_size);
    memset(t->grad, 0, total_elements * element_size);
    t->n_dims = n_dims;
    memcpy(t->dims, dims, n_dims * sizeof(int));
    t->dtype = dtype;
    t->arena = arena;
    return t;
}

void free_tensor(Tensor* t) {
    if (t && t->arena == NULL) {
        if (t->dims) free(t->dims);
        if (t->data) free(t->data);
        if (t->grad) free(t->grad);
        free(t);
    }
}

void create_look_ahead_mask(Tensor* mask, int seq_len) {
    assert(mask->n_dims == 2 && mask->dims[0] == seq_len && mask->dims[1] == seq_len);
    assert(mask->dtype == TENSOR_TYPE_FLOAT);
    float* mask_data = (float*)mask->data;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                mask_data[i * seq_len + j] = -1e9; // Large negative number
            } else {
                mask_data[i * seq_len + j] = 0.0f;
            }
        }
    }
}

int save_tensor(Tensor* t, FILE* fp) {
    if (fwrite(&t->dtype, sizeof(DataType), 1, fp) != 1) return 0;
    if (fwrite(&t->n_dims, sizeof(int), 1, fp) != 1) return 0;
    if (fwrite(t->dims, sizeof(int), t->n_dims, fp) != t->n_dims) return 0;

    size_t total_elements = 1;
    for (int i = 0; i < t->n_dims; i++) total_elements *= t->dims[i];
    
    size_t element_size = (t->dtype == TENSOR_TYPE_FLOAT) ? sizeof(float) : (t->dtype == TENSOR_TYPE_INT) ? sizeof(int) : sizeof(uint16_t);

    if (fwrite(t->data, element_size, total_elements, fp) != total_elements) return 0;

    return 1;
}

Tensor* load_tensor(FILE* fp, Arena* arena) {
    DataType dtype;
    int n_dims;
    if (fread(&dtype, sizeof(DataType), 1, fp) != 1) return NULL;
    if (fread(&n_dims, sizeof(int), 1, fp) != 1) return NULL;
    int* dims = (int*)malloc(n_dims * sizeof(int));
    if (!dims) { fprintf(stderr, "[ERR] malloc failed for dims in load_tensor\n"); return NULL; }
    if (fread(dims, sizeof(int), n_dims, fp) != n_dims) {
        fprintf(stderr, "[ERR] fread failed for dims in load_tensor\n");
        free(dims);
        return NULL;
    }
    Tensor* t = create_tensor(arena, n_dims, dims, dtype);
    if (!t) {
        fprintf(stderr, "[ERR] create_tensor failed in load_tensor\n");
        free(dims);
        return NULL;
    }
    free(dims);
    size_t total_elements = 1;
    for (int i = 0; i < t->n_dims; i++) total_elements *= t->dims[i];
    size_t element_size = (t->dtype == TENSOR_TYPE_FLOAT) ? sizeof(float) :
                          (t->dtype == TENSOR_TYPE_INT) ? sizeof(int) : sizeof(uint16_t);
    if (fread(t->data, element_size, total_elements, fp) != total_elements) {
        fprintf(stderr, "[ERR] fread failed for tensor data in load_tensor\n");
        free_tensor(t);
        return NULL;
    }
    return t;
}

#ifdef USE_CBLAS
void matmul(Tensor* c, Tensor* a, Tensor* b) {
    assert(a->dtype == TENSOR_TYPE_FLOAT);
    assert(b->dtype == TENSOR_TYPE_FLOAT);
    assert(c->dtype == TENSOR_TYPE_FLOAT);

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* c_data = (float*)c->data;

    // Case 1: Simple 2D x 2D matrix multiplication
    if (a->n_dims == 2 && b->n_dims == 2) {
        assert(a->dims[1] == b->dims[0]);
        assert(c->n_dims == 2);
        assert(c->dims[0] == a->dims[0] && c->dims[1] == b->dims[1]);

        int M = a->dims[0];
        int K = a->dims[1];
        int N = b->dims[1];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    a_data, K, b_data, N, 0.0f, c_data, N);
        return;
    }

    // Case 2: Batched 3D x 2D -> 3D (e.g. (B, S, E) @ (E, F) -> (B, S, F))
    if (a->n_dims == 3 && b->n_dims == 2) {
        int B = a->dims[0];
        int S = a->dims[1];
        int E = a->dims[2];
        int F = b->dims[1];
        assert(E == b->dims[0]);
        assert(c->n_dims == 3 && c->dims[0] == B && c->dims[1] == S && c->dims[2] == F);

        // This can be done as a single large matmul (B*S, E) @ (E, F) -> (B*S, F)
        int M = B * S;
        int K = E;
        int N = F;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    a_data, K, b_data, N, 0.0f, c_data, N);
        return;
    }

    // Case 3: Batched 4D x 4D -> 4D (e.g. (B, H, S, D) @ (B, H, D, S') -> (B, H, S, S'))
    if (a->n_dims == 4 && b->n_dims == 4) {
        int B = a->dims[0];
        int H = a->dims[1];
        int S = a->dims[2];
        int D = a->dims[3];
        int S_prime = b->dims[3];
        assert(B == b->dims[0] && H == b->dims[1] && D == b->dims[2]);
        assert(c->n_dims == 4 && c->dims[0] == B && c->dims[1] == H && c->dims[2] == S && c->dims[3] == S_prime);

        for (int b_idx = 0; b_idx < B; b_idx++) {
            for (int h_idx = 0; h_idx < H; h_idx++) {
                int M = S;
                int K = D;
                int N = S_prime;
                float* a_offset = a_data + b_idx * H * S * D + h_idx * S * D;
                float* b_offset = b_data + b_idx * H * D * S_prime + h_idx * D * S_prime;
                float* c_offset = c_data + b_idx * H * S * S_prime + h_idx * S * S_prime;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                            a_offset, K, b_offset, N, 0.0f, c_offset, N);
            }
        }
        return;
    }
    
    // Case 4: Batched 3D x 3D -> 3D (e.g. (B, S, E) @ (B, E, T) -> (B, S, T))
    if (a->n_dims == 3 && b->n_dims == 3) {
        int B = a->dims[0];
        int S = a->dims[1];
        int E = a->dims[2];
        int T = b->dims[2];
        assert(B == b->dims[0] && E == b->dims[1]);
        assert(c->n_dims == 3 && c->dims[0] == B && c->dims[1] == S && c->dims[2] == T);

        for (int b_idx = 0; b_idx < B; b_idx++) {
            int M = S;
            int K = E;
            int N = T;
            float* a_offset = a_data + b_idx * S * E;
            float* b_offset = b_data + b_idx * E * T;
            float* c_offset = c_data + b_idx * S * T;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                        a_offset, K, b_offset, N, 0.0f, c_offset, N);
        }
        return;
    }


    fprintf(stderr, "Unsupported matmul dimensions with BLAS\n");
    assert(0);
}
#else
void matmul(Tensor* c, Tensor* a, Tensor* b) {
    assert(a->dtype == TENSOR_TYPE_FLOAT);
    assert(b->dtype == TENSOR_TYPE_FLOAT);
    assert(c->dtype == TENSOR_TYPE_FLOAT);

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* c_data = (float*)c->data;

    // Case 1: Simple 2D x 2D matrix multiplication
    if (a->n_dims == 2 && b->n_dims == 2) {
        assert(a->dims[1] == b->dims[0]);
        assert(c->n_dims == 2);
        assert(c->dims[0] == a->dims[0] && c->dims[1] == b->dims[1]);

        int a_rows = a->dims[0];
        int a_cols = a->dims[1];
        int b_cols = b->dims[1];

#if defined(__x86_64__) && defined(__AVX2__)
        for (int i = 0; i < a_rows; i++) {
            for (int j = 0; j < b_cols; j++) {
                float sum = 0.0f;
                int k = 0;
                __asm__ __volatile__(
                    "vxorps %%ymm0, %%ymm0, %%ymm0\n\t" // zero ymm0 (accumulator)
                    "1:\n\t"
                    "cmp %[a_cols], %[k]\n\t"
                    "jge 2f\n\t"
                    "vmovups (%[a_ptr], %[k], 4), %%ymm1\n\t" // load 8 floats from a
                    "vbroadcastss (%[b_ptr], %[k], 4), %%ymm2\n\t" // broadcast b[k]
                    "vfmadd231ps %%ymm1, %%ymm2, %%ymm0\n\t" // acc += a * b
                    "add $8, %[k]\n\t"
                    "jmp 1b\n\t"
                    "2:\n\t"
                    "vhaddps %%ymm0, %%ymm0, %%ymm0\n\t"
                    "vhaddps %%ymm0, %%ymm0, %%ymm0\n\t"
                    "vextractf128 $1, %%ymm0, %%xmm1\n\t"
                    "vaddps %%xmm0, %%xmm1, %%xmm0\n\t"
                    "vmovss %%xmm0, %[sum]\n\t"
                    : [sum] "=m" (sum), [k] "+r" (k)
                    : [a_ptr] "r" (&a_data[i * a_cols]), [b_ptr] "r" (&b_data[j]), [a_cols] "r" (a_cols)
                    : "ymm0", "ymm1", "ymm2", "xmm1", "memory"
                );
                c_data[i * b_cols + j] = sum;
            }
        }
#elif defined(__x86_64__) && defined(__AVX__)
        for (int i = 0; i < a_rows; i++) {
            for (int j = 0; j < b_cols; j++) {
                float sum = 0.0f;
                int k = 0;
                __asm__ __volatile__(
                    "vxorps %%ymm0, %%ymm0, %%ymm0\n\t" // zero ymm0 (accumulator)
                    "1:\n\t"
                    "cmp %[a_cols], %[k]\n\t"
                    "jge 2f\n\t"
                    "vmovups (%[a_ptr], %[k], 4), %%ymm1\n\t" // load 8 floats from a
                    "vbroadcastss (%[b_ptr], %[k], 4), %%ymm2\n\t" // broadcast b[k]
                    "vmulps %%ymm1, %%ymm2, %%ymm3\n\t" // ymm3 = a * b
                    "vaddps %%ymm3, %%ymm0, %%ymm0\n\t" // acc += ymm3
                    "add $8, %[k]\n\t"
                    "jmp 1b\n\t"
                    "2:\n\t"
                    "vhaddps %%ymm0, %%ymm0, %%ymm0\n\t"
                    "vhaddps %%ymm0, %%ymm0, %%ymm0\n\t"
                    "vextractf128 $1, %%ymm0, %%xmm1\n\t"
                    "vaddps %%xmm0, %%xmm1, %%xmm0\n\t"
                    "vmovss %%xmm0, %[sum]\n\t"
                    : [sum] "=m" (sum), [k] "+r" (k)
                    : [a_ptr] "r" (&a_data[i * a_cols]), [b_ptr] "r" (&b_data[j]), [a_cols] "r" (a_cols)
                    : "ymm0", "ymm1", "ymm2", "ymm3", "xmm1", "memory"
                );
                c_data[i * b_cols + j] = sum;
            }
        }
#else
        #pragma omp parallel for
        for (int i = 0; i < a_rows; i++) {
            for (int j = 0; j < b_cols; j++) {
                float sum = 0.0f;
                for (int k = 0; k < a_cols; k++) {
                    sum += a_data[i * a_cols + k] * b_data[k * b_cols + j];
                }
                c_data[i * b_cols + j] = sum;
            }
        }
#endif
        return;
    }
    
    // Case 2: Batched 3D x 2D -> 3D (e.g. (B, S, E) @ (E, F) -> (B, S, F))
    if (a->n_dims == 3 && b->n_dims == 2) {
        int B = a->dims[0];
        int S = a->dims[1];
        int E = a->dims[2];
        int F = b->dims[1];
        assert(E == b->dims[0]);
        assert(c->n_dims == 3 && c->dims[0] == B && c->dims[1] == S && c->dims[2] == F);

        #pragma omp parallel for collapse(2)
        for (int b_idx = 0; b_idx < B; b_idx++) {
            for (int s_idx = 0; s_idx < S; s_idx++) {
                for (int f_idx = 0; f_idx < F; f_idx++) {
                    float sum = 0.0f;
                    for (int e_idx = 0; e_idx < E; e_idx++) {
                        sum += a_data[b_idx * S * E + s_idx * E + e_idx] * b_data[e_idx * F + f_idx];
                    }
                    c_data[b_idx * S * F + s_idx * F + f_idx] = sum;
                }
            }
        }
        return;
    }

    // Case 3: Batched 4D x 4D -> 4D (e.g. (B, H, S, D) @ (B, H, D, S') -> (B, H, S, S'))
    if (a->n_dims == 4 && b->n_dims == 4) {
        int B = a->dims[0];
        int H = a->dims[1];
        int S = a->dims[2];
        int D = a->dims[3];
        int S_prime = b->dims[3];
        assert(B == b->dims[0] && H == b->dims[1] && D == b->dims[2]);
        assert(c->n_dims == 4 && c->dims[0] == B && c->dims[1] == H && c->dims[2] == S && c->dims[3] == S_prime);

        #pragma omp parallel for collapse(3)
        for (int b_idx = 0; b_idx < B; b_idx++) {
            for (int h_idx = 0; h_idx < H; h_idx++) {
                for (int s_idx = 0; s_idx < S; s_idx++) {
                    for (int sp_idx = 0; sp_idx < S_prime; sp_idx++) {
                        float sum = 0.0f;
                        for (int d_idx = 0; d_idx < D; d_idx++) {
                            sum += a_data[b_idx*H*S*D + h_idx*S*D + s_idx*D + d_idx] * 
                                   b_data[b_idx*H*D*S_prime + h_idx*D*S_prime + d_idx*S_prime + sp_idx];
                        }
                        c_data[b_idx*H*S*S_prime + h_idx*S*S_prime + s_idx*S_prime + sp_idx] = sum;
                    }
                }
            }
        }
        return;
    }

    // Case 4: Batched 3D x 3D -> 3D (e.g. (B, S, E) @ (B, E, T) -> (B, S, T))
    if (a->n_dims == 3 && b->n_dims == 3) {
        int B = a->dims[0];
        int S = a->dims[1];
        int E = a->dims[2];
        int T = b->dims[2];
        assert(B == b->dims[0] && E == b->dims[1]);
        assert(c->n_dims == 3 && c->dims[0] == B && c->dims[1] == S && c->dims[2] == T);

        #pragma omp parallel for collapse(2)
        for (int b_idx = 0; b_idx < B; b_idx++) {
            for (int s_idx = 0; s_idx < S; s_idx++) {
                for (int t_idx = 0; t_idx < T; t_idx++) {
                    float sum = 0.0f;
                    for (int e_idx = 0; e_idx < E; e_idx++) {
                        sum += a_data[b_idx * S * E + s_idx * E + e_idx] * b_data[b_idx * E * T + e_idx * T + t_idx];
                    }
                    c_data[b_idx * S * T + s_idx * T + t_idx] = sum;
                }
            }
        }
        return;
    }

    fprintf(stderr, "Unsupported matmul dimensions\n");
    assert(0);
}
#endif

void add(Tensor* c, Tensor* a, Tensor* b) {
    assert(a->dtype == TENSOR_TYPE_FLOAT);
    assert(b->dtype == TENSOR_TYPE_FLOAT);
    assert(c->dtype == TENSOR_TYPE_FLOAT);

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* c_data = (float*)c->data;
    // Check if shapes are identical for element-wise addition
    int same_shape = a->n_dims == b->n_dims;
    if (same_shape) {
        for (int i = 0; i < a->n_dims; i++) {
            if (a->dims[i] != b->dims[i]) {
                same_shape = 0;
                break;
            }
        }
    }

    if (same_shape) {
        assert(c->n_dims == a->n_dims);
        size_t total_elements = 1;
        for (int i = 0; i < a->n_dims; i++) {
            assert(c->dims[i] == a->dims[i]);
            total_elements *= a->dims[i];
        }
#if defined(__x86_64__) && defined(__AVX2__)
        size_t i = 0;
        for (; i + 8 <= total_elements; i += 8) {
            __asm__ __volatile__(
                "vmovups (%[a]), %%ymm0\n\t"
                "vmovups (%[b]), %%ymm1\n\t"
                "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
                "vmovups %%ymm0, (%[c])\n\t"
                :
                : [a] "r" (a_data + i), [b] "r" (b_data + i), [c] "r" (c_data + i)
                : "ymm0", "ymm1", "memory"
            );
        }
        for (; i < total_elements; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
#elif defined(__x86_64__) && defined(__AVX__)
        size_t i = 0;
        for (; i + 8 <= total_elements; i += 8) {
            __asm__ __volatile__(
                "vmovups (%[a]), %%ymm0\n\t"
                "vmovups (%[b]), %%ymm1\n\t"
                "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
                "vmovups %%ymm0, (%[c])\n\t"
                :
                : [a] "r" (a_data + i), [b] "r" (b_data + i), [c] "r" (c_data + i)
                : "ymm0", "ymm1", "memory"
            );
        }
        for (; i < total_elements; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
#else
        #pragma omp parallel for
        for (size_t i = 0; i < total_elements; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
#endif
        return;
    }

    // Broadcasting: b is broadcast over a
    // Example: a is (batch, seq, features), b is (features)
    assert(c->n_dims == a->n_dims);
    for(int i = 0; i < a->n_dims; i++) {
        assert(a->dims[i] == c->dims[i]);
    }

    int b_last_dim = b->dims[b->n_dims - 1];
    assert(a->dims[a->n_dims - 1] == b_last_dim);
    
    size_t outer_size = 1;
    for (int i = 0; i < a->n_dims - 1; i++) {
        outer_size *= a->dims[i];
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        for (int j = 0; j < b_last_dim; j++) {
            c_data[i * b_last_dim + j] = a_data[i * b_last_dim + j] + b_data[j];
        }
    }
}

void softmax(Tensor* t) {
    assert(t->dtype == TENSOR_TYPE_FLOAT);
    assert(t->n_dims > 0);
    
    float* t_data = (float*)t->data;
    int last_dim = t->dims[t->n_dims - 1];
    size_t outer_size = 1;
    for (int i = 0; i < t->n_dims - 1; i++) {
        outer_size *= t->dims[i];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        float* row = t_data + i * last_dim;

        float max_val = row[0];
        for (int j = 1; j < last_dim; j++) {
            if (row[j] > max_val) {
                max_val = row[j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }

        for (int j = 0; j < last_dim; j++) {
            row[j] /= sum;
        }
    }
}

void transpose(Tensor* out, Tensor* in, int dim1, int dim2) {
    assert(in->dtype == TENSOR_TYPE_FLOAT);
    assert(out->dtype == TENSOR_TYPE_FLOAT);
    assert(in->n_dims > dim1 && in->n_dims > dim2);
    assert(out->n_dims == in->n_dims);
    // check output dims
    for (int i = 0; i < in->n_dims; i++) {
        if (i == dim1) assert(out->dims[i] == in->dims[dim2]);
        else if (i == dim2) assert(out->dims[i] == in->dims[dim1]);
        else assert(out->dims[i] == in->dims[i]);
    }
    if (dim1 == dim2) {
        // no-op transpose
        memcpy(out->data, in->data, tensor_numel(in) * sizeof(float));
        return;
    }
    int n_dims = in->n_dims;
    int* in_strides = (int*)malloc(n_dims * sizeof(int));
    int* out_strides = (int*)malloc(n_dims * sizeof(int));
    in_strides[n_dims-1] = 1;
    out_strides[n_dims-1] = 1;
    for (int i = n_dims-2; i >= 0; i--) {
        in_strides[i] = in_strides[i+1] * in->dims[i+1];
        out_strides[i] = out_strides[i+1] * out->dims[i+1];
    }
    int total = tensor_numel(in);
    float* in_data = (float*)in->data;
    float* out_data = (float*)out->data;
    // for each element, compute its output index with dim1/dim2 swapped
    for (int idx = 0; idx < total; idx++) {
        int in_idx = idx;
        int coord[16]; // supports up to 16D
        for (int d = 0; d < n_dims; d++) {
            coord[d] = in_idx / in_strides[d];
            in_idx = in_idx % in_strides[d];
        }
        // swap dim1 and dim2 for output
        int out_coord[16];
        for (int d = 0; d < n_dims; d++) out_coord[d] = coord[d];
        int tmp = out_coord[dim1];
        out_coord[dim1] = out_coord[dim2];
        out_coord[dim2] = tmp;
        // compute output flat index
        int out_idx = 0;
        for (int d = 0; d < n_dims; d++) {
            out_idx += out_coord[d] * out_strides[d];
        }
        out_data[out_idx] = in_data[idx];
    }
    free(in_strides);
    free(out_strides);
}

void scale(Tensor* out, Tensor* in, float scalar) {
    assert(in->dtype == TENSOR_TYPE_FLOAT);
    assert(out->dtype == TENSOR_TYPE_FLOAT);
    assert(in->n_dims == out->n_dims);
    size_t total_elements = 1;
    for (int i = 0; i < in->n_dims; i++) {
        assert(in->dims[i] == out->dims[i]);
        total_elements *= in->dims[i];
    }
    float* in_data = (float*)in->data;
    float* out_data = (float*)out->data;
#if defined(__x86_64__) && defined(__AVX2__)
    size_t i = 0;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    for (; i + 8 <= total_elements; i += 8) {
        __asm__ __volatile__(
            "vmovups (%[in]), %%ymm0\n\t"
            "vmulps %[scalar], %%ymm0, %%ymm0\n\t"
            "vmovups %%ymm0, (%[out])\n\t"
            :
            : [in] "r" (in_data + i), [out] "r" (out_data + i), [scalar] "x" (scalar_vec)
            : "ymm0", "memory"
        );
    }
    for (; i < total_elements; i++) {
        out_data[i] = in_data[i] * scalar;
    }
#elif defined(__x86_64__) && defined(__AVX__)
    size_t i = 0;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    for (; i + 8 <= total_elements; i += 8) {
        __asm__ __volatile__(
            "vmovups (%[in]), %%ymm0\n\t"
            "vmulps %[scalar], %%ymm0, %%ymm0\n\t"
            "vmovups %%ymm0, (%[out])\n\t"
            :
            : [in] "r" (in_data + i), [out] "r" (out_data + i), [scalar] "x" (scalar_vec)
            : "ymm0", "memory"
        );
    }
    for (; i < total_elements; i++) {
        out_data[i] = in_data[i] * scalar;
    }
#else
    #pragma omp parallel for
    for (size_t i = 0; i < total_elements; i++) {
        out_data[i] = in_data[i] * scalar;
    }
#endif
}
