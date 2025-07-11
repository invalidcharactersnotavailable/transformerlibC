#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    float rate;
} Dropout;

Dropout* create_dropout(float rate);
void free_dropout(Dropout* d);
void dropout_forward(Tensor* out, Tensor* in, Dropout* d, int training);
Value* dropout_forward_ad(Arena* arena, Value* in, Dropout* d, int training);

#endif // DROPOUT_H 