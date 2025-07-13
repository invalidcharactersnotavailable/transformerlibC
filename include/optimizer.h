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

/**
 * create_optimizer - allocate an adam optimizer for a model
 * @model: transformer model
 * @learning_rate: learning rate
 * returns: pointer to optimizer or NULL on failure
 */
Optimizer* create_optimizer(Transformer* model, float learning_rate);

/**
 * create_optimizer_with_type - allocate an optimizer for a model with specific type
 * @model: transformer model
 * @learning_rate: learning rate
 * @type: optimizer type (sgd or adam)
 * returns: pointer to optimizer or NULL on failure
 */
Optimizer* create_optimizer_with_type(Transformer* model, float learning_rate, int type);

/**
 * free_optimizer - free an optimizer and all associated memory
 * @opt: pointer to optimizer
 */
void free_optimizer(Optimizer* opt);

/**
 * optimizer_step - perform an optimizer step
 * @opt: pointer to optimizer
 */
void optimizer_step(Optimizer* opt);

/**
 * zero_grad - zero all gradients in the optimizer
 * @opt: pointer to optimizer
 */
void zero_grad(Optimizer* opt);

/**
 * optimizer_set_type - set the optimizer type
 * @opt: pointer to optimizer
 * @type: optimizer type (OPTIMIZER_SGD or OPTIMIZER_ADAM)
 */
void optimizer_set_type(Optimizer* opt, OptimizerType type);

/**
 * gather_params - gather all parameters from a model
 * @model: transformer model
 * @params: array to fill with parameter pointers
 * returns: number of parameters
 */
int gather_params(Transformer* model, Tensor** params);

#endif // OPTIMIZER_H 