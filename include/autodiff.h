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

/**
 * create_value - allocate a value node for autodiff
 * @arena: memory arena (or NULL)
 * @data: tensor data
 * @prev: previous value nodes
 * @n_prev: number of previous nodes
 * @op_context: context for backward op
 * @_backward: backward function pointer
 * returns: pointer to value or NULL on failure
 */
Value* create_value(Arena* arena, Tensor* data, Value** prev, int n_prev, void* op_context, void (*_backward)(struct Value*));

/**
 * free_value - free a value node
 * @v: pointer to value
 */
void free_value(Value* v);

/**
 * backward - run backward pass from a value node
 * @v: pointer to value
 */
void backward(Value* v);

/**
 * free_graph - free a computation graph
 * @v: pointer to root value
 */
void free_graph(Value* v);

/**
 * matmul_ad - autodiff matmul operation
 * @arena: memory arena
 * @a: left value
 * @b: right value
 * returns: new value node
 */
Value* matmul_ad(Arena* arena, Value* a, Value* b);

/**
 * add_ad - autodiff add operation
 * @arena: memory arena
 * @a: left value
 * @b: right value
 * returns: new value node
 */
Value* add_ad(Arena* arena, Value* a, Value* b);

/**
 * softmax_ad - autodiff softmax operation
 * @arena: memory arena
 * @v: input value
 * returns: new value node
 */
Value* softmax_ad(Arena* arena, Value* v);

/**
 * scale_ad - autodiff scale operation
 * @arena: memory arena
 * @a: input value
 * @scalar: scale factor
 * returns: new value node
 */
Value* scale_ad(Arena* arena, Value* a, float scalar);

/**
 * transpose_ad - autodiff transpose operation
 * @arena: memory arena
 * @a: input value
 * @dim1: first dimension
 * @dim2: second dimension
 * returns: new value node
 */
Value* transpose_ad(Arena* arena, Value* a, int dim1, int dim2);

/**
 * relu_ad - autodiff relu operation
 * @arena: memory arena
 * @a: input value
 * returns: new value node
 */
Value* relu_ad(Arena* arena, Value* a);

/**
 * reshape_ad - autodiff reshape operation
 * @arena: memory arena
 * @a: input value
 * @n_dims: number of dimensions
 * @dims: new dimensions
 * returns: new value node
 */
Value* reshape_ad(Arena* arena, Value* a, int n_dims, int* dims);

#endif // AUTODIFF_H 