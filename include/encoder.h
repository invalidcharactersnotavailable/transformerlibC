#ifndef ENCODER_H
#define ENCODER_H

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"
#include "tensor.h"
#include "dropout.h"
#include "autodiff.h"

typedef struct {
    int embed_dim;
    int n_heads;
    int ff_hidden_dim;

    MultiHeadAttention* attention;
    FeedForward* ff;
    LayerNorm* ln1;
    LayerNorm* ln2;
    Dropout* dropout;
} EncoderBlock;

/**
 * create_encoder_block - allocate an encoder block
 * @embed_dim: embedding dimension
 * @n_heads: number of heads
 * @ff_hidden_dim: feedforward hidden dimension
 * returns: pointer to encoder block or NULL on failure
 */
EncoderBlock* create_encoder_block(int embed_dim, int n_heads, int ff_hidden_dim);

/**
 * free_encoder_block - free an encoder block
 * @block: pointer to encoder block
 */
void free_encoder_block(EncoderBlock* block);

/**
 * encoder_block_forward - run forward pass of encoder block
 * @out: output tensor
 * @in: input tensor
 * @block: encoder block
 * @training: training mode flag
 */
void encoder_block_forward(Tensor* out, Tensor* in, EncoderBlock* block, int training);

/**
 * encoder_block_forward_ad - autodiff forward pass for encoder block
 * @arena: memory arena
 * @in: input value
 * @block: encoder block
 * @training: training mode flag
 * returns: autodiff value
 */
Value* encoder_block_forward_ad(Arena* arena, Value* in, EncoderBlock* block, int training);

/**
 * save_encoder_block - write encoder block to file
 * @block: encoder block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int save_encoder_block(EncoderBlock* block, FILE* fp);

/**
 * load_encoder_block - read encoder block from file
 * @block: encoder block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int load_encoder_block(EncoderBlock* block, FILE* fp);

#endif // ENCODER_H
