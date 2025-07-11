#include "optimizer.h"
#include <stdlib.h>
#include <string.h>

int gather_params(Transformer* model, Tensor** params) {
    int count = 0;
    // gather embedding weights
    params[count++] = model->embedding->weights;
    // gather encoder and decoder block params
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
    // output layer
    params[count++] = model->output_layer;
    return count;
}

Optimizer* create_optimizer(Transformer* model, float learning_rate) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->model = model;
    opt->learning_rate = learning_rate;
    // estimate: 32 params per layer * n_layers + 2 (embedding, output)
    int max_params = model->n_layers * 32 + 2;
    opt->params = (Tensor**)malloc(max_params * sizeof(Tensor*));
    opt->num_params = gather_params(model, opt->params);
    return opt;
}

void free_optimizer(Optimizer* opt) {
    free(opt->params);
    free(opt);
}

void optimizer_step(Optimizer* opt) {
    for (int i = 0; i < opt->num_params; i++) {
        Tensor* param = opt->params[i];
        (void)param;
        // placeholder for parameter update logic
    }
}

void zero_grad(Optimizer* opt) {
    for (int i = 0; i < opt->num_params; i++) {
        Tensor* param = opt->params[i];
        (void)param;
        // placeholder for zeroing gradients
    }
} 