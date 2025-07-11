#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.h"

typedef struct {
    float p; // Dropout probability
} Dropout;

Dropout* create_dropout(float p);
void free_dropout(Dropout* d);
void dropout_forward(Tensor* out, Tensor* in, Dropout* d, int training);

#endif // DROPOUT_H 