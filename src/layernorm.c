#include "layernorm.h"
#include "tensor.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

LayerNorm* create_layernorm(int size) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    if (!ln) return NULL;

    ln->size = size;
    int dims[] = {size};
    ln->gamma = create_tensor(1, dims);
    ln->beta = create_tensor(1, dims);

    // Initialize gamma to 1s and beta to 0s
    for (int i = 0; i < size; i++) {
        ln->gamma->data[i] = 1.0f;
        ln->beta->data[i] = 0.0f;
    }

    return ln;
}

void free_layernorm(LayerNorm* ln) {
    if (ln) {
        free_tensor(ln->gamma);
        free_tensor(ln->beta);
        free(ln);
    }
}

void layernorm_forward(Tensor* out, Tensor* in, LayerNorm* ln) {
    assert(in->n_dims > 0 && out->n_dims == in->n_dims);
    assert(in->dims[in->n_dims - 1] == ln->size);
    
    size_t outer_size = 1;
    for (int i = 0; i < in->n_dims - 1; i++) {
        outer_size *= in->dims[i];
    }
    int feature_size = ln->size;
    float eps = 1e-5f;

    for (size_t i = 0; i < outer_size; i++) {
        float* in_row = in->data + i * feature_size;
        float* out_row = out->data + i * feature_size;

        float mean = 0.0f;
        for (int j = 0; j < feature_size; j++) {
            mean += in_row[j];
        }
        mean /= feature_size;

        float variance = 0.0f;
        for (int j = 0; j < feature_size; j++) {
            variance += (in_row[j] - mean) * (in_row[j] - mean);
        }
        variance /= feature_size;

        float inv_std = 1.0f / sqrtf(variance + eps);

        for (int j = 0; j < feature_size; j++) {
            out_row[j] = (in_row[j] - mean) * inv_std * ln->gamma->data[j] + ln->beta->data[j];
        }
    }
}
