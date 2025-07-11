#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

typedef struct {
    int size;
    Tensor* gamma;
    Tensor* beta;
} LayerNorm;

LayerNorm* create_layernorm(int size);
void free_layernorm(LayerNorm* ln);
void layernorm_forward(Tensor* out, Tensor* in, LayerNorm* ln);

#endif // LAYERNORM_H
