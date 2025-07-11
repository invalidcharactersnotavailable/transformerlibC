#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"

typedef struct {
    int dim;
    int hidden_dim;
    Tensor* W1;
    Tensor* b1;
    Tensor* W2;
    Tensor* b2;
} FeedForward;

FeedForward* create_feedforward(int dim, int hidden_dim);
void free_feedforward(FeedForward* ff);
void feedforward_forward(Tensor* out, Tensor* in, FeedForward* ff);

#endif // FEEDFORWARD_H
