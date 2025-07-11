#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    int embed_dim;
    Tensor *gamma, *beta; // Trainable parameters
} LayerNorm;

LayerNorm* create_layernorm(int embed_dim);
void free_layernorm(LayerNorm* ln);
void layernorm_forward(Tensor* out, Tensor* in, LayerNorm* ln);
Value* layernorm_forward_ad(Value* in, LayerNorm* ln, Arena* arena);

#endif // LAYERNORM_H
