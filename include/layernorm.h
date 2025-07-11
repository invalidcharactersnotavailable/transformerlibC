#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    int embed_dim;
    Tensor *gamma, *beta; // Trainable parameters
} LayerNorm;

/**
 * create_layernorm - allocate a layer normalization block
 * @dim: embedding dimension
 * returns: pointer to layernorm block or NULL on failure
 */
LayerNorm* create_layernorm(int dim);

/**
 * free_layernorm - free a layer normalization block
 * @ln: pointer to layernorm block
 */
void free_layernorm(LayerNorm* ln);

/**
 * layernorm_forward - run forward pass of layer normalization
 * @out: output tensor
 * @in: input tensor
 * @ln: layernorm block
 */
void layernorm_forward(Tensor* out, Tensor* in, LayerNorm* ln);

/**
 * layernorm_forward_ad - autodiff forward pass for layer normalization
 * @arena: memory arena
 * @in: input value
 * @ln: layernorm block
 * returns: autodiff value
 */
Value* layernorm_forward_ad(Arena* arena, Value* in, LayerNorm* ln);

/**
 * save_layernorm - write layernorm block to file
 * @ln: layernorm block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int save_layernorm(LayerNorm* ln, FILE* fp);

/**
 * load_layernorm - read layernorm block from file
 * @ln: layernorm block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int load_layernorm(LayerNorm* ln, FILE* fp);

#endif // LAYERNORM_H
