#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    Tensor *w1, *b1, *w2, *b2;
} FeedForward;

FeedForward* create_feedforward(int embed_dim, int hidden_dim);
void free_feedforward(FeedForward* ff);
void feedforward_forward(Tensor* out, Tensor* in, FeedForward* ff, Arena* arena);
Value* feedforward_forward_ad(Value* in, FeedForward* ff, Arena* arena);

#endif // FEEDFORWARD_H
