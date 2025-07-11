#include "optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// count params in model
static int count_params(Transformer* model) {
    int count = 0;
    count++; // embedding
    for (int i = 0; i < model->n_layers; i++) {
        EncoderBlock* enc = model->encoder_layers[i];
        count += 12; // encoder params
        DecoderBlock* dec = model->decoder_layers[i];
        count += 16; // decoder params
    }
    count++; // output layer
    return count;
}

// fill params in model
static void fill_params(Transformer* model, Tensor** params) {
    int count = 0;
    params[count++] = model->embedding->weights;
    for (int i = 0; i < model->n_layers; i++) {
        EncoderBlock* enc = model->encoder_layers[i];
        params[count++] = enc->attention->w_q;
        params[count++] = enc->attention->w_k;
        params[count++] = enc->attention->w_v;
        params[count++] = enc->attention->w_o;
        params[count++] = enc->ff->w1;
        params[count++] = enc->ff->b1;
        params[count++] = enc->ff->w2;
        params[count++] = enc->ff->b2;
        params[count++] = enc->ln1->gamma;
        params[count++] = enc->ln1->beta;
        params[count++] = enc->ln2->gamma;
        params[count++] = enc->ln2->beta;
        DecoderBlock* dec = model->decoder_layers[i];
        params[count++] = dec->masked_attention->w_q;
        params[count++] = dec->masked_attention->w_k;
        params[count++] = dec->masked_attention->w_v;
        params[count++] = dec->masked_attention->w_o;
        params[count++] = dec->cross_attention->w_q;
        params[count++] = dec->cross_attention->w_k;
        params[count++] = dec->cross_attention->w_v;
        params[count++] = dec->cross_attention->w_o;
        params[count++] = dec->ff->w1;
        params[count++] = dec->ff->b1;
        params[count++] = dec->ff->w2;
        params[count++] = dec->ff->b2;
        params[count++] = dec->ln1->gamma;
        params[count++] = dec->ln1->beta;
        params[count++] = dec->ln2->gamma;
        params[count++] = dec->ln2->beta;
        params[count++] = dec->ln3->gamma;
        params[count++] = dec->ln3->beta;
    }
    params[count++] = model->output_layer;
}

int gather_params(Transformer* model, Tensor** params) {
    fill_params(model, params);
    return count_params(model);
}

Optimizer* create_optimizer(Transformer* model, float learning_rate) {
    return create_optimizer_with_type(model, learning_rate, OPTIMIZER_ADAM);
}

Optimizer* create_optimizer_with_type(Transformer* model, float learning_rate, int type) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->model = model;
    opt->learning_rate = learning_rate;
    int n = count_params(model);
    opt->params = (Tensor**)malloc(n * sizeof(Tensor*));
    fill_params(model, opt->params);
    opt->num_params = n;
    opt->type = type;
    opt->mixed_precision = 0;
    if (type == OPTIMIZER_ADAM) {
        opt->adam = (struct AdamState*)malloc(sizeof(struct AdamState));
        opt->adam->m = (float**)malloc(n * sizeof(float*));
        opt->adam->v = (float**)malloc(n * sizeof(float*));
        for (int i = 0; i < n; i++) {
            size_t sz = 1;
            for (int d = 0; d < opt->params[i]->n_dims; d++) sz *= opt->params[i]->dims[d];
            opt->adam->m[i] = (float*)calloc(sz, sizeof(float));
            opt->adam->v[i] = (float*)calloc(sz, sizeof(float));
        }
        opt->adam->t = 0;
    } else {
        opt->adam = NULL;
    }
    return opt;
}

void optimizer_enable_mixed_precision(Optimizer* opt, int enable) {
    opt->mixed_precision = enable;
}

void free_optimizer(Optimizer* opt) {
    if (opt->adam) {
        for (int i = 0; i < opt->num_params; i++) {
            free(opt->adam->m[i]);
            free(opt->adam->v[i]);
        }
        free(opt->adam->m);
        free(opt->adam->v);
        free(opt->adam);
    }
    free(opt->params);
    free(opt);
}

void optimizer_step(Optimizer* opt) {
    float lr = opt->learning_rate;
    if (opt->type == OPTIMIZER_SGD) {
        for (int i = 0; i < opt->num_params; i++) {
            Tensor* param = opt->params[i];
            float* p = (float*)param->data;
            float* g = (float*)param->grad;
            size_t sz = 1;
            for (int d = 0; d < param->n_dims; d++) sz *= param->dims[d];
            if (opt->mixed_precision && param->dtype == TENSOR_TYPE_FLOAT16) {
                // convert grad to float32, update, then cast back
                for (size_t j = 0; j < sz; j++) {
                    float grad32 = float16_to_float32(((uint16_t*)g)[j]);
                    float param32 = float16_to_float32(((uint16_t*)p)[j]);
                    param32 -= lr * grad32;
                    ((uint16_t*)p)[j] = float32_to_float16(param32);
                }
            } else {
                for (size_t j = 0; j < sz; j++) {
                    p[j] -= lr * g[j];
                }
            }
        }
    } else if (opt->type == OPTIMIZER_ADAM) {
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        opt->adam->t++;
        for (int i = 0; i < opt->num_params; i++) {
            Tensor* param = opt->params[i];
            float* p = (float*)param->data;
            float* g = (float*)param->grad;
            float* m = opt->adam->m[i];
            float* v = opt->adam->v[i];
            size_t sz = 1;
            for (int d = 0; d < param->n_dims; d++) sz *= param->dims[d];
            if (opt->mixed_precision && param->dtype == TENSOR_TYPE_FLOAT16) {
                for (size_t j = 0; j < sz; j++) {
                    float grad32 = float16_to_float32(((uint16_t*)g)[j]);
                    float param32 = float16_to_float32(((uint16_t*)p)[j]);
                    m[j] = beta1 * m[j] + (1 - beta1) * grad32;
                    v[j] = beta2 * v[j] + (1 - beta2) * grad32 * grad32;
                    float m_hat = m[j] / (1 - powf(beta1, opt->adam->t));
                    float v_hat = v[j] / (1 - powf(beta2, opt->adam->t));
                    param32 -= lr * m_hat / (sqrtf(v_hat) + eps);
                    ((uint16_t*)p)[j] = float32_to_float16(param32);
                }
            } else {
                for (size_t j = 0; j < sz; j++) {
                    m[j] = beta1 * m[j] + (1 - beta1) * g[j];
                    v[j] = beta2 * v[j] + (1 - beta2) * g[j] * g[j];
                    float m_hat = m[j] / (1 - powf(beta1, opt->adam->t));
                    float v_hat = v[j] / (1 - powf(beta2, opt->adam->t));
                    p[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
                }
            }
        }
    }
}

void zero_grad(Optimizer* opt) {
    for (int i = 0; i < opt->num_params; i++) {
        Tensor* param = opt->params[i];
        float* g = (float*)param->grad;
        size_t sz = 1;
        for (int d = 0; d < param->n_dims; d++) sz *= param->dims[d];
        if (opt->mixed_precision && param->dtype == TENSOR_TYPE_FLOAT16) {
            memset(g, 0, sz * sizeof(uint16_t));
        } else {
            memset(g, 0, sz * sizeof(float));
        }
    }
} 