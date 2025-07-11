#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "transformer.h"
#include <stddef.h>

typedef enum { OPTIMIZER_SGD, OPTIMIZER_ADAM } OptimizerType;

struct AdamState {
    float** m;
    float** v;
    int t;
};

typedef struct Optimizer {
    Transformer* model;
    float learning_rate;
    Tensor** params;
    int num_params;
    OptimizerType type;
    struct AdamState* adam;
    int mixed_precision;
} Optimizer;

Optimizer* create_optimizer(Transformer* model, float learning_rate);
Optimizer* create_optimizer_with_type(Transformer* model, float learning_rate, int type);
void free_optimizer(Optimizer* opt);
void optimizer_step(Optimizer* opt);
void zero_grad(Optimizer* opt);
int gather_params(Transformer* model, Tensor** params);

#endif // OPTIMIZER_H 