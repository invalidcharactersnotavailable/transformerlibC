#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "tensor.h"
#include "memory.h"

// Forward declaration
struct Value;

typedef void (*BackwardOp)(struct Value*);

typedef struct Value {
    Tensor* data;
    Tensor* grad;
    struct Value** prev;
    int n_prev;
    void (*_backward)(struct Value*);
    void* op_context; // Context for the backward operation
    Arena* arena;
} Value;

Value* create_value(Tensor* data, Value** prev, int n_prev, void* op_context, Arena* arena, void (*_backward)(struct Value*));
void free_value(Value* v); // free_value should not be used with arena
void backward(Value* v, Arena* arena);

// --- Autodiff-enabled operations ---
// All these functions create new values and need an arena for temporary allocations
Value* matmul_ad(Value* a, Value* b, Arena* arena);
Value* add_ad(Value* a, Value* b, Arena* arena);
Value* softmax_ad(Value* v, Arena* arena);
Value* scale_ad(Value* a, float scalar, Arena* arena);
Value* transpose_ad(Value* a, int dim1, int dim2, Arena* arena);
Value* relu_ad(Value* a, Arena* arena);

#endif // AUTODIFF_H 