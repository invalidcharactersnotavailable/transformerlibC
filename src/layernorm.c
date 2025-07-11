#include "layernorm.h"
#include "tensor.h"
#include "autodiff.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

LayerNorm* create_layernorm(int embed_dim) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    ln->embed_dim = embed_dim;
    int dims[] = {embed_dim};
    ln->gamma = create_tensor(1, dims, TENSOR_TYPE_FLOAT, NULL);
    ln->beta = create_tensor(1, dims, TENSOR_TYPE_FLOAT, NULL);
    // Initialize gamma to 1s and beta to 0s
    for (int i = 0; i < embed_dim; i++) {
        ((float*)ln->gamma->data)[i] = 1.0f;
        ((float*)ln->beta->data)[i] = 0.0f;
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
    assert(in->dtype == TENSOR_TYPE_FLOAT);
    assert(out->dtype == TENSOR_TYPE_FLOAT);
    assert(in->n_dims > 0 && out->n_dims == in->n_dims);
    assert(in->dims[in->n_dims - 1] == ln->embed_dim);
    
    size_t outer_size = 1;
    for (int i = 0; i < in->n_dims - 1; i++) {
        outer_size *= in->dims[i];
    }
    int feature_size = ln->embed_dim;
    float eps = 1e-5f;

    for (size_t i = 0; i < outer_size; i++) {
        float* in_row = (float*)in->data + i * feature_size;
        float* out_row = (float*)out->data + i * feature_size;

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
            out_row[j] = (in_row[j] - mean) * inv_std * ((float*)ln->gamma->data)[j] + ((float*)ln->beta->data)[j];
        }
    }
}

void backward_layernorm(Value* v) {
    // Placeholder for layernorm backward pass
}

Value* layernorm_forward_ad(Value* in, LayerNorm* ln, Arena* arena) {
    Tensor* out_data = create_tensor(in->data->n_dims, in->data->dims, TENSOR_TYPE_FLOAT, arena);
    layernorm_forward(out_data, in->data, ln);
    
    // The backward pass will need gamma and beta as inputs as well.
    Value** prev = (Value**)arena_alloc(arena, 3 * sizeof(Value*));
    prev[0] = in;
    prev[1] = create_value(ln->gamma, NULL, 0, NULL, arena, NULL);
    prev[2] = create_value(ln->beta, NULL, 0, NULL, arena, NULL);

    return create_value(out_data, prev, 3, NULL, arena, backward_layernorm);
}
