#include "encoder.h"
#include <stdlib.h>

EncoderBlock* create_encoder_block(int embed_dim, int n_heads, int ff_hidden_dim) {
    EncoderBlock* block = (EncoderBlock*)malloc(sizeof(EncoderBlock));
    if (!block) return NULL;

    block->embed_dim = embed_dim;
    block->n_heads = n_heads;
    block->ff_hidden_dim = ff_hidden_dim;

    block->attention = create_multihead_attention(embed_dim, n_heads);
    block->ff = create_feedforward(embed_dim, ff_hidden_dim);
    block->ln1 = create_layernorm(embed_dim);
    block->ln2 = create_layernorm(embed_dim);
    block->dropout = create_dropout(0.1); // Default 0.1 dropout rate

    return block;
}

void free_encoder_block(EncoderBlock* block) {
    if (block) {
        free_multihead_attention(block->attention);
        free_feedforward(block->ff);
        free_layernorm(block->ln1);
        free_layernorm(block->ln2);
        free_dropout(block->dropout);
        free(block);
    }
}

void encoder_block_forward(Tensor* out, Tensor* in, EncoderBlock* block, Arena* arena, int training) {
    // Self-attention part
    Tensor* attn_out = create_tensor(in->n_dims, in->dims, TENSOR_TYPE_FLOAT, arena);
    multihead_attention_forward(attn_out, in, block->attention, NULL, arena);

    // Dropout, Add & Norm
    Tensor* dropout_out = create_tensor(in->n_dims, in->dims, TENSOR_TYPE_FLOAT, arena);
    dropout_forward(dropout_out, attn_out, block->dropout, training);
    
    Tensor* add_norm1 = create_tensor(in->n_dims, in->dims, TENSOR_TYPE_FLOAT, arena);
    add(add_norm1, in, dropout_out);
    layernorm_forward(add_norm1, add_norm1, block->ln1);

    // Feed-forward part
    Tensor* ff_out = create_tensor(in->n_dims, in->dims, TENSOR_TYPE_FLOAT, arena);
    feedforward_forward(ff_out, add_norm1, block->ff, arena);
    
    // Add & Norm
    dropout_forward(out, ff_out, block->dropout, training);
    
    Tensor* add_norm2 = create_tensor(in->n_dims, in->dims, TENSOR_TYPE_FLOAT, arena);
    add(add_norm2, add_norm1, out);
    layernorm_forward(out, add_norm2, block->ln2);
}

Value* encoder_block_forward_ad(Value* in, EncoderBlock* block, Arena* arena, int training) {
    // Note: Dropout is omitted in the AD version for now.
    Value* attn_out = multihead_attention_forward_ad(in, block->attention, NULL, arena);
    Value* add1 = add_ad(in, attn_out, arena);
    Value* norm1 = layernorm_forward_ad(add1, block->ln1, arena);

    Value* ff_out = feedforward_forward_ad(norm1, block->ff, arena);
    Value* add2 = add_ad(norm1, ff_out, arena);
    Value* norm2 = layernorm_forward_ad(add2, block->ln2, arena);

    return norm2;
}
