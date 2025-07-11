#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "encoder.h"
#include "tensor.h"

typedef struct {
    int n_layers;
    EncoderBlock** layers;
    // We would also have token and positional embeddings here
} Transformer;

Transformer* create_transformer(int n_layers, int embed_dim, int n_heads, int ff_hidden_dim);
void free_transformer(Transformer* t);
void transformer_forward(Tensor* out, Tensor* in, Transformer* t);

#endif // TRANSFORMER_H
