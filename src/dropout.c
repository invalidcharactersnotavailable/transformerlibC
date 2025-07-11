#include "dropout.h"
#include <stdlib.h>
#include <assert.h>
#include <time.h>

Dropout* create_dropout(float rate) {
    Dropout* d = (Dropout*)malloc(sizeof(Dropout));
    d->rate = rate;
    // Seed random number generator
    srand(time(NULL));
    return d;
}

void free_dropout(Dropout* d) {
    if (d) free(d);
}

void dropout_forward(Tensor* out, Tensor* in, Dropout* d, int training) {
    if (!training || d->rate == 0) {
        // Just copy input to output if not training or dropout rate is zero
        memcpy(out->data, in->data, out->dims[0] * out->dims[1] * sizeof(float));
        return;
    }
    float scale = 1.0f / (1.0f - d->rate);
    float* in_data = (float*)in->data;
    float* out_data = (float*)out->data;
    size_t size = in->dims[0] * in->dims[1];
    for (size_t i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX > d->rate) {
            out_data[i] = in_data[i] * scale;
        } else {
            out_data[i] = 0.0f;
        }
    }
}

typedef struct {
    Tensor* mask;
    float rate;
} DropoutContext;

void backward_dropout(Value* v) {
    DropoutContext* ctx = (DropoutContext*)v->op_context;
    Value* in_val = v->prev[0];
    float scale = 1.0f / (1.0f - ctx->rate);
    
    size_t size = 1;
    for (int i=0; i < v->grad->n_dims; i++) size *= v->grad->dims[i];

    float* d_out = (float*)v->grad->data;
    float* d_in = (float*)in_val->grad->data;
    float* mask = (float*)ctx->mask->data;

    for (size_t i = 0; i < size; i++) {
        d_in[i] += d_out[i] * mask[i] * scale;
    }
}

Value* dropout_forward_ad(Arena* arena, Value* in, Dropout* d, int training) {
    if (!training || d->rate == 0) {
        return in;
    }

    Tensor* out_data = create_tensor(arena, in->data->n_dims, in->data->dims, TENSOR_TYPE_FLOAT);
    Tensor* mask = create_tensor(arena, in->data->n_dims, in->data->dims, TENSOR_TYPE_FLOAT);
    float scale = 1.0f / (1.0f - d->rate);

    float* in_data = (float*)in->data->data;
    float* out_data_p = (float*)out_data->data;
    float* mask_data = (float*)mask->data;
    
    size_t size = 1;
    for (int i = 0; i < in->data->n_dims; i++) size *= in->data->dims[i];
    
    for (size_t i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX > d->rate) {
            mask_data[i] = 1.0f;
            out_data_p[i] = in_data[i] * scale;
        } else {
            mask_data[i] = 0.0f;
            out_data_p[i] = 0.0f;
        }
    }

    DropoutContext* ctx = (DropoutContext*)arena_alloc(arena, sizeof(DropoutContext));
    ctx->mask = mask;
    ctx->rate = d->rate;

    Value** prev = (Value**)arena_alloc(arena, 1 * sizeof(Value*));
    prev[0] = in;

    return create_value(arena, out_data, prev, 1, ctx, backward_dropout);
} 