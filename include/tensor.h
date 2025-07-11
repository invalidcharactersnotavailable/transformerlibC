#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

#include "memory.h"

typedef enum {
    TENSOR_TYPE_FLOAT,
    TENSOR_TYPE_INT
} DataType;

typedef struct {
    int n_dims;
    int *dims;
    void *data;
    DataType dtype;
    Arena* arena; // If not NULL, data is allocated in this arena
} Tensor;

Tensor* create_tensor(int n_dims, int *dims, DataType dtype, Arena* arena);
void free_tensor(Tensor *t);

void matmul(Tensor* c, Tensor* a, Tensor* b);
void add(Tensor* c, Tensor* a, Tensor* b);
void softmax(Tensor* t);
void transpose(Tensor* out, Tensor* in, int dim1, int dim2);
void scale(Tensor* out, Tensor* in, float scalar);

#endif // TENSOR_H
