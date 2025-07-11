#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

#include "memory.h"

#define HAS_FLOAT16 1

typedef enum {
    TENSOR_TYPE_FLOAT,
    TENSOR_TYPE_INT,
    TENSOR_TYPE_FLOAT16
} DataType;

typedef struct {
    int n_dims;
    int *dims;
    void *data;
    void *grad;
    DataType dtype;
    // Arena* arena; // removed, all memory is dynamic
} Tensor;

Tensor* create_tensor(Arena* arena, int n_dims, int* dims, DataType dtype);
void free_tensor(Tensor *t);

// Tensor utility functions
void create_look_ahead_mask(Tensor* mask, int seq_len);

// I/O functions
int save_tensor(Tensor* t, FILE* fp);
Tensor* load_tensor(FILE* fp, Arena* arena);

// Tensor operations (in-place and out-of-place versions)
void matmul(Tensor* c, Tensor* a, Tensor* b);
void add(Tensor* out, Tensor* a, Tensor* b);
void softmax(Tensor* t);
void transpose(Tensor* out, Tensor* in, int dim1, int dim2);
void scale(Tensor* out, Tensor* in, float scalar);

float float16_to_float32(uint16_t h);
uint16_t float32_to_float16(float f);

#endif // TENSOR_H
