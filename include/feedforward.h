#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    Tensor *w1, *b1, *w2, *b2;
} FeedForward;

/**
 * create_feedforward - allocate a feedforward block
 * @embed_dim: embedding dimension
 * @hidden_dim: hidden layer dimension
 * returns: pointer to feedforward block or NULL on failure
 */
FeedForward* create_feedforward(int embed_dim, int hidden_dim);

/**
 * free_feedforward - free a feedforward block
 * @ff: pointer to feedforward block
 */
void free_feedforward(FeedForward* ff);

/**
 * feedforward_forward - run forward pass of feedforward block
 * @out: output tensor
 * @in: input tensor
 * @ff: feedforward block
 */
void feedforward_forward(Tensor* out, Tensor* in, FeedForward* ff);

/**
 * feedforward_forward_ad - autodiff forward pass for feedforward block
 * @in: input value
 * @ff: feedforward block
 * returns: autodiff value
 */
Value* feedforward_forward_ad(Value* in, FeedForward* ff);

/**
 * save_feedforward - write feedforward block to file
 * @ff: feedforward block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int save_feedforward(FeedForward* ff, FILE* fp);

/**
 * load_feedforward - read feedforward block from file
 * @ff: feedforward block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int load_feedforward(FeedForward* ff, FILE* fp);

#endif // FEEDFORWARD_H
