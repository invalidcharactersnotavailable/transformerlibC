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

Value* create_value(Arena* arena, Tensor* data, Value** prev, int n_prev, void* op_context, void (*_backward)(struct Value*));
void free_value(Value* v);
void backward(Value* v);
void free_graph(Value* v);

// autodiff-enabled operations (no arena needed)
Value* matmul_ad(Arena* arena, Value* a, Value* b);
Value* add_ad(Arena* arena, Value* a, Value* b);
Value* softmax_ad(Arena* arena, Value* v);
Value* scale_ad(Arena* arena, Value* a, float scalar);
Value* transpose_ad(Arena* arena, Value* a, int dim1, int dim2);
Value* relu_ad(Arena* arena, Value* a);
Value* reshape_ad(Arena* arena, Value* a, int n_dims, int* dims);

#endif // AUTODIFF_H 