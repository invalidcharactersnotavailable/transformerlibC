#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

Tensor* create_tensor(int n_dims, int* dims) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->n_dims = n_dims;
    t->dims = (int*)malloc(n_dims * sizeof(int));
    if (!t->dims) {
        free(t);
        return NULL;
    }
    memcpy(t->dims, dims, n_dims * sizeof(int));

    size_t total_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        total_elements *= dims[i];
    }

    t->data = (float*)calloc(total_elements, sizeof(float));
    if (!t->data) {
        free(t->dims);
        free(t);
        return NULL;
    }

    return t;
}

void free_tensor(Tensor* t) {
    if (t) {
        if (t->dims) free(t->dims);
        if (t->data) free(t->data);
        free(t);
    }
}

void matmul(Tensor* c, Tensor* a, Tensor* b) {
    assert(a->n_dims == 2 && b->n_dims == 2);
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
                sum += a->data[i * a_cols + k] * b->data[k * b_cols + j];
            }
            c->data[i * b_cols + j] = sum;
        }
    }
}

void add(Tensor* c, Tensor* a, Tensor* b) {
    assert(a->n_dims == b->n_dims);
    assert(c->n_dims == a->n_dims);
    size_t total_elements = 1;
    for (int i = 0; i < a->n_dims; i++) {
        assert(a->dims[i] == b->dims[i]);
        assert(c->dims[i] == a->dims[i]);
        total_elements *= a->dims[i];
    }

    for (size_t i = 0; i < total_elements; i++) {
        c->data[i] = a->data[i] + b->data[i];
    }
}

void softmax(Tensor* t) {
    assert(t->n_dims > 0);
    
    int last_dim = t->dims[t->n_dims - 1];
    size_t outer_size = 1;
    for (int i = 0; i < t->n_dims - 1; i++) {
        outer_size *= t->dims[i];
    }

    for (size_t i = 0; i < outer_size; i++) {
        float* row = t->data + i * last_dim;

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
