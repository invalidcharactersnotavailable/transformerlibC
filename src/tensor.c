#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "memory.h"

Tensor* create_tensor(int n_dims, int* dims, DataType dtype, Arena* arena) {
    size_t total_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        total_elements *= dims[i];
    }

    size_t element_size = 0;
    if (dtype == TENSOR_TYPE_FLOAT) {
        element_size = sizeof(float);
    } else if (dtype == TENSOR_TYPE_INT) {
        element_size = sizeof(int);
    }

    Tensor* t;
    if (arena) {
        t = (Tensor*)arena_alloc(arena, sizeof(Tensor));
        if (!t) return NULL;
        t->dims = (int*)arena_alloc(arena, n_dims * sizeof(int));
        if (!t->dims) return NULL; // Arena will be reset, so no need to free
        t->data = arena_alloc(arena, total_elements * element_size);
        if (!t->data) return NULL;
        memset(t->data, 0, total_elements * element_size); // calloc behavior
    } else {
        t = (Tensor*)malloc(sizeof(Tensor));
        if (!t) return NULL;
        t->dims = (int*)malloc(n_dims * sizeof(int));
        if (!t->dims) {
            free(t);
            return NULL;
        }
        t->data = calloc(total_elements, element_size);
        if (!t->data) {
            free(t->dims);
            free(t);
            return NULL;
        }
    }

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
        free(t);
    }
}

#ifdef USE_BLAS
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

        for (int i = 0; i < a_rows; i++) {
            for (int j = 0; j < b_cols; j++) {
                float sum = 0.0f;
                for (int k = 0; k < a_cols; k++) {
                    sum += a_data[i * a_cols + k] * b_data[k * b_cols + j];
                }
                c_data[i * b_cols + j] = sum;
            }
        }
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
        for (size_t i = 0; i < total_elements; i++) {
            c_data[i] = a_data[i] + b_data[i];
        }
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
    for (int i = 0; i < in->n_dims; i++) {
        if (i == dim1) assert(out->dims[i] == in->dims[dim2]);
        else if (i == dim2) assert(out->dims[i] == in->dims[dim1]);
        else assert(out->dims[i] == in->dims[i]);
    }

    float* in_data = (float*)in->data;
    float* out_data = (float*)out->data;
    // This is a generic N-dimensional transpose, but it's complex.
    // Let's implement the specific 4D case we need for attention: (B, S, H, D) -> (B, H, S, D)
    if (in->n_dims == 4 && dim1 == 1 && dim2 == 2) {
        int B = in->dims[0];
        int S = in->dims[1];
        int H = in->dims[2];
        int D = in->dims[3];

        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                for (int h = 0; h < H; h++) {
                    for (int d = 0; d < D; d++) {
                        // Source index: b*S*H*D + s*H*D + h*D + d
                        // Destination index: b*H*S*D + h*S*D + s*D + d
                        out_data[b*H*S*D + h*S*D + s*D + d] = in_data[b*S*H*D + s*H*D + h*D + d];
                    }
                }
            }
        }
        return;
    }
    
    // Transposing last two dimensions, e.g. for attention scores (B, H, S, D) -> (B, H, D, S)
    if (in->n_dims == 4 && dim1 == 2 && dim2 == 3) {
        int B = in->dims[0];
        int H = in->dims[1];
        int S = in->dims[2];
        int D = in->dims[3];

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                for (int s = 0; s < S; s++) {
                    for (int d = 0; d < D; d++) {
                        // Source index: b*H*S*D + h*S*D + s*D + d
                        // Destination index: b*H*D*S + h*D*S + d*S + s
                        out_data[b*H*D*S + h*D*S + d*S + s] = in_data[b*H*S*D + h*S*D + s*D + d];
                    }
                }
            }
        }
        return;
    }

    // Transposing last two dimensions of a 3D tensor (B, S, D) -> (B, D, S)
    if (in->n_dims == 3 && dim1 == 1 && dim2 == 2) {
        int B = in->dims[0];
        int S = in->dims[1];
        int D = in->dims[2];
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                for (int d = 0; d < D; d++) {
                    out_data[b*D*S + d*S + s] = in_data[b*S*D + s*D + d];
                }
            }
        }
        return;
    }


    fprintf(stderr, "Unsupported transpose dimensions\n");
    assert(0);
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
    for (size_t i = 0; i < total_elements; i++) {
        out_data[i] = in_data[i] * scalar;
    }
}
