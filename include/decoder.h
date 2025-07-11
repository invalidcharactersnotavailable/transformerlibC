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

DecoderBlock* create_decoder_block(int embed_dim, int n_heads, int ff_hidden_dim);
void free_decoder_block(DecoderBlock* block);
void decoder_block_forward(Tensor* out, Tensor* tgt_in, Tensor* encoder_out, DecoderBlock* block, Arena* arena, int training);
Value* decoder_block_forward_ad(Value* tgt, Value* encoder_out, DecoderBlock* block, Arena* arena, int training);

#endif // DECODER_H 