#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    float rate;
} Dropout;

/**
 * create_dropout - allocate a dropout layer
 * @rate: dropout rate
 * returns: pointer to dropout layer or NULL on failure
 */
Dropout* create_dropout(float rate);

/**
 * free_dropout - free a dropout layer
 * @d: pointer to dropout layer
 */
void free_dropout(Dropout* d);

/**
 * dropout_forward - run forward pass of dropout
 * @out: output tensor
 * @in: input tensor
 * @d: dropout layer
 * @training: training mode flag
 */
void dropout_forward(Tensor* out, Tensor* in, Dropout* d, int training);

/**
 * dropout_forward_ad - autodiff forward pass for dropout
 * @in: input value
 * @d: dropout layer
 * @training: training mode flag
 * returns: autodiff value
 */
Value* dropout_forward_ad(Value* in, Dropout* d, int training);

#endif // DROPOUT_H 