#include "encoder.h"
#include <stdlib.h>

EncoderBlock* create_encoder_block(int embed_dim, int n_heads, int ff_hidden_dim) {
    EncoderBlock* block = (EncoderBlock*)malloc(sizeof(EncoderBlock));
    block->embed_dim = embed_dim;
    block->n_heads = n_heads;
    block->ff_hidden_dim = ff_hidden_dim;

    block->attention = create_multihead_attention(embed_dim, n_heads);
    block->ff = create_feedforward(embed_dim, ff_hidden_dim);
    block->ln1 = create_layernorm(embed_dim);
    block->ln2 = create_layernorm(embed_dim);

    return block;
}

void free_encoder_block(EncoderBlock* block) {
    if (block) {
        free_multihead_attention(block->attention);
        free_feedforward(block->ff);
        free_layernorm(block->ln1);
        free_layernorm(block->ln2);
        free(block);
    }
}

void encoder_block_forward(Tensor* out, Tensor* in, EncoderBlock* block) {
    int dims[] = {in->dims[0], in->dims[1], in->dims[2]};
    Tensor* attn_out = create_tensor(3, dims);
    Tensor* norm1_out = create_tensor(3, dims);
    Tensor* ff_out = create_tensor(3, dims);

    // 1. LayerNorm + MultiHeadAttention
    layernorm_forward(norm1_out, in, block->ln1);
    multihead_attention_forward(attn_out, norm1_out, block->attention);
    
    // Residual connection
    add(attn_out, attn_out, in);

    // 2. LayerNorm + FeedForward
    layernorm_forward(norm1_out, attn_out, block->ln2); // reuse norm1_out as temp buffer
    feedforward_forward(ff_out, norm1_out, block->ff);
    
    // Residual connection
    add(out, ff_out, attn_out);

    free_tensor(attn_out);
    free_tensor(norm1_out);
    free_tensor(ff_out);
}
