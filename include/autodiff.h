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
} Value;

/**
 * create_value - allocate a value node for autodiff
 * @data: tensor data
 * @prev: previous value nodes
 * @n_prev: number of previous nodes
 * @op_context: context for backward op
 * @_backward: backward function pointer
 * returns: pointer to value or NULL on failure
 */
Value* create_value(Tensor* data, Value** prev, int n_prev, void* op_context, void (*_backward)(struct Value*));

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
 * @a: left value
 * @b: right value
 * returns: new value node
 */
Value* matmul_ad(Value* a, Value* b);

/**
 * add_ad - autodiff add operation
 * @a: left value
 * @b: right value
 * returns: new value node
 */
Value* add_ad(Value* a, Value* b);

/**
 * softmax_ad - autodiff softmax operation
 * @v: input value
 * returns: new value node
 */
Value* softmax_ad(Value* v);

/**
 * scale_ad - autodiff scale operation
 * @a: input value
 * @scalar: scale factor
 * returns: new value node
 */
Value* scale_ad(Value* a, float scalar);

/**
 * transpose_ad - autodiff transpose operation
 * @a: input value
 * @dim1: first dimension
 * @dim2: second dimension
 * returns: new value node
 */
Value* transpose_ad(Value* a, int dim1, int dim2);

/**
 * relu_ad - autodiff relu operation
 * @a: input value
 * returns: new value node
 */
Value* relu_ad(Value* a);

/**
 * reshape_ad - autodiff reshape operation
 * @a: input value
 * @n_dims: number of dimensions
 * @dims: new dimensions
 * returns: new value node
 */
Value* reshape_ad(Value* a, int n_dims, int* dims);

#endif // AUTODIFF_H 