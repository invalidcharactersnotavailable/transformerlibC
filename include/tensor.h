#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

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

/**
 * create_tensor - allocate a tensor with the given shape and type
 * @n_dims: number of dimensions
 * @dims: array of dimension sizes
 * @dtype: data type
 * returns: pointer to tensor or NULL on failure
 */
Tensor* create_tensor(int n_dims, int* dims, DataType dtype);

/**
 * free_tensor - free a tensor allocated with create_tensor
 * @t: tensor to free
 */
void free_tensor(Tensor *t);

/**
 * create_look_ahead_mask - fill a tensor with a look-ahead mask for seq2seq
 * @mask: tensor to fill (must be correct shape)
 * @seq_len: sequence length
 */
void create_look_ahead_mask(Tensor* mask, int seq_len);

/**
 * save_tensor - write tensor to file
 * @t: tensor to save
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int save_tensor(Tensor* t, FILE* fp);

/**
 * load_tensor - read tensor from file
 * @fp: file pointer
 * returns: pointer to tensor or NULL on failure
 */
Tensor* load_tensor(FILE* fp);

// Tensor operations (in-place and out-of-place versions)
void matmul(Tensor* c, Tensor* a, Tensor* b);
void add(Tensor* out, Tensor* a, Tensor* b);
void softmax(Tensor* out, Tensor* in);
void transpose(Tensor* out, Tensor* in, int dim1, int dim2);
void scale(Tensor* out, Tensor* in, float scalar);
void sum(Tensor* out, Tensor* in, int* axes, int n_axes);

float float16_to_float32(uint16_t h);
uint16_t float32_to_float16(float f);

long tensor_numel(Tensor* t);

#endif // TENSOR_H
