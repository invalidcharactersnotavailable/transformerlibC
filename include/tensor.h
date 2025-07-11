#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    int n_dims;
    int *dims;
    float *data;
} Tensor;

Tensor* create_tensor(int n_dims, int *dims);
void free_tensor(Tensor *t);

void matmul(Tensor* c, Tensor* a, Tensor* b);
void add(Tensor* c, Tensor* a, Tensor* b);
void softmax(Tensor* t);

#endif // TENSOR_H
