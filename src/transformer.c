#include "transformer.h"
#include <stdlib.h>

Transformer* create_transformer(int n_layers, int embed_dim, int n_heads, int ff_hidden_dim) {
    Transformer* t = (Transformer*)malloc(sizeof(Transformer));
    t->n_layers = n_layers;
    t->layers = (EncoderBlock**)malloc(n_layers * sizeof(EncoderBlock*));
    for (int i = 0; i < n_layers; i++) {
        t->layers[i] = create_encoder_block(embed_dim, n_heads, ff_hidden_dim);
    }
    return t;
}

void free_transformer(Transformer* t) {
    if (t) {
        for (int i = 0; i < t->n_layers; i++) {
            free_encoder_block(t->layers[i]);
        }
        free(t->layers);
        free(t);
    }
}

void transformer_forward(Tensor* out, Tensor* in, Transformer* t) {
    int dims[] = {in->dims[0], in->dims[1], in->dims[2]};
    Tensor* current_in = create_tensor(3, dims);
    Tensor* current_out = create_tensor(3, dims);

    // For simplicity, we just copy input. A real implementation would handle embeddings.
    memcpy(current_in->data, in->data, in->dims[0] * in->dims[1] * in->dims[2] * sizeof(float));

    for (int i = 0; i < t->n_layers; i++) {
        encoder_block_forward(current_out, current_in, t->layers[i]);
        // Swap pointers for next iteration
        Tensor* temp = current_in;
        current_in = current_out;
        current_out = temp;
    }

    // Copy final result to out tensor
    memcpy(out->data, current_in->data, out->dims[0] * out->dims[1] * out->dims[2] * sizeof(float));

    free_tensor(current_in);
    free_tensor(current_out);
}
