#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include "autodiff.h"

// Computes the cross-entropy loss.
// `logits` are the raw output from the model.
// `targets` are the ground truth labels (integer indices).
float cross_entropy_loss(Tensor* logits, Tensor* targets);

// Autodiff-enabled version.
Value* cross_entropy_loss_ad(Value* logits, Tensor* targets);

#endif // LOSS_H 