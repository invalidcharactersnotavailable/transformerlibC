#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    Tensor *w1, *b1, *w2, *b2;
} FeedForward;

FeedForward* create_feedforward(int embed_dim, int hidden_dim);
void free_feedforward(FeedForward* ff);
void feedforward_forward(Tensor* out, Tensor* in, FeedForward* ff);
Value* feedforward_forward_ad(Arena* arena, Value* in, FeedForward* ff);
int save_feedforward(FeedForward* ff, FILE* fp);
int load_feedforward(FeedForward* ff, FILE* fp);

#endif // FEEDFORWARD_H
