#ifndef ENCODER_H
#define ENCODER_H

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"
#include "tensor.h"

typedef struct {
    int embed_dim;
    int n_heads;
    int ff_hidden_dim;

    MultiHeadAttention* attention;
    FeedForward* ff;
    LayerNorm* ln1;
    LayerNorm* ln2;
} EncoderBlock;

EncoderBlock* create_encoder_block(int embed_dim, int n_heads, int ff_hidden_dim);
void free_encoder_block(EncoderBlock* block);
void encoder_block_forward(Tensor* out, Tensor* in, EncoderBlock* block);

#endif // ENCODER_H
