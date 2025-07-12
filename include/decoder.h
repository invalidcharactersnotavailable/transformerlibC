#ifndef DECODER_H
#define DECODER_H

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

    MultiHeadAttention* masked_attention;
    MultiHeadAttention* cross_attention;
    FeedForward* ff;
    LayerNorm* ln1;
    LayerNorm* ln2;
    LayerNorm* ln3;
    Dropout* dropout;
} DecoderBlock;

/**
 * create_decoder_block - allocate a decoder block
 * @embed_dim: embedding dimension
 * @n_heads: number of heads
 * @ff_hidden_dim: feedforward hidden dimension
 * returns: pointer to decoder block or NULL on failure
 */
DecoderBlock* create_decoder_block(int embed_dim, int n_heads, int ff_hidden_dim);

/**
 * free_decoder_block - free a decoder block
 * @block: pointer to decoder block
 */
void free_decoder_block(DecoderBlock* block);

/**
 * decoder_block_forward - run forward pass of decoder block
 * @out: output tensor
 * @tgt_in: target input tensor
 * @encoder_out: encoder output tensor
 * @block: decoder block
 * @training: training mode flag
 */
void decoder_block_forward(Tensor* out, Tensor* tgt_in, Tensor* encoder_out, DecoderBlock* block, int training);

/**
 * decoder_block_forward_ad - autodiff forward pass for decoder block
 * @tgt: target input value
 * @encoder_out: encoder output value
 * @block: decoder block
 * @training: training mode flag
 * @look_ahead_mask: look-ahead mask tensor
 * returns: autodiff value
 */
Value* decoder_block_forward_ad(Value* tgt, Value* encoder_out, DecoderBlock* block, int training, Tensor* look_ahead_mask);

/**
 * save_decoder_block - write decoder block to file
 * @block: decoder block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int save_decoder_block(DecoderBlock* block, FILE* fp);

/**
 * load_decoder_block - read decoder block from file
 * @block: decoder block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int load_decoder_block(DecoderBlock* block, FILE* fp);

#endif // DECODER_H 