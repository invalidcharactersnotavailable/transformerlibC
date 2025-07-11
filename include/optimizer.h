#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "transformer.h"
#include <stddef.h>

typedef struct {
    Transformer* model;
    float learning_rate;
    Tensor** params;
    int num_params;
} Optimizer;

Optimizer* create_optimizer(Transformer* model, float learning_rate);
void free_optimizer(Optimizer* opt);
void optimizer_step(Optimizer* opt);
void zero_grad(Optimizer* opt);
int gather_params(Transformer* model, Tensor** params);

#endif // OPTIMIZER_H 