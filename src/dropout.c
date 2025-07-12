#include "dropout.h"
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <stdio.h> // Added for fprintf

// global random seed flag
static int dropout_seeded = 0;

// create a dropout layer with given rate
Dropout* create_dropout(float rate) {
    if (!dropout_seeded) {
        srand((unsigned int)time(NULL));
        dropout_seeded = 1;
    }
    Dropout* d = (Dropout*)malloc(sizeof(Dropout));
    if (!d) { fprintf(stderr, "[ERR] malloc failed for Dropout\n"); return NULL; }
    d->rate = rate;
    return d;
}

// free a dropout layer
void free_dropout(Dropout* d) {
    if (d) free(d);
}

// apply dropout to input tensor, write to out tensor
void dropout_forward(Tensor* out, Tensor* in, Dropout* d, int training) {
    assert(out && in && d);
    assert(in->dtype == TENSOR_TYPE_FLOAT && out->dtype == TENSOR_TYPE_FLOAT);
    assert(in->n_dims == out->n_dims);
    for (int i = 0; i < in->n_dims; i++) assert(in->dims[i] == out->dims[i]);
    size_t size = 1;
    for (int i = 0; i < in->n_dims; i++) size *= in->dims[i];
    float* in_data = (float*)in->data;
    float* out_data = (float*)out->data;
    if (!training || d->rate == 0) {
        memcpy(out_data, in_data, size * sizeof(float));
        return;
    }
    float scale = 1.0f / (1.0f - d->rate);
    for (size_t i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX > d->rate) {
            out_data[i] = in_data[i] * scale;
        } else {
            out_data[i] = 0.0f;
        }
    }
}

// context for autodiff dropout
typedef struct {
    Tensor* mask;
    float rate;
} DropoutContext;

// backward pass for dropout
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

// forward pass for dropout in autodiff
Value* dropout_forward_ad(Value* in, Dropout* d, int training) {
    assert(in && d);
    if (!training || d->rate == 0) {
        return in;
    }
    Tensor* out_data = create_tensor(in->data->n_dims, in->data->dims, TENSOR_TYPE_FLOAT);
    Tensor* mask = create_tensor(in->data->n_dims, in->data->dims, TENSOR_TYPE_FLOAT);
    if (!out_data || !mask) return NULL;
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
    DropoutContext* ctx = (DropoutContext*)malloc(sizeof(DropoutContext));
    if (!ctx) return NULL;
    ctx->mask = mask;
    ctx->rate = d->rate;
    Value** prev = (Value**)malloc(sizeof(Value*));
    if (!prev) return NULL;
    prev[0] = in;
    return create_value(out_data, prev, 1, ctx, backward_dropout);
} 