#include "decoder.h"
#include <stdlib.h>

DecoderBlock* create_decoder_block(int embed_dim, int n_heads, int ff_hidden_dim) {
    DecoderBlock* block = (DecoderBlock*)malloc(sizeof(DecoderBlock));
    if (!block) return NULL;

    block->embed_dim = embed_dim;
    block->n_heads = n_heads;
    block->ff_hidden_dim = ff_hidden_dim;

    block->masked_attention = create_multihead_attention(embed_dim, n_heads);
    block->cross_attention = create_multihead_attention(embed_dim, n_heads);
    block->ff = create_feedforward(embed_dim, ff_hidden_dim);
    block->ln1 = create_layernorm(embed_dim);
    block->ln2 = create_layernorm(embed_dim);
    block->ln3 = create_layernorm(embed_dim);
    block->dropout = create_dropout(0.1);

    return block;
}

void free_decoder_block(DecoderBlock* block) {
    if (block) {
        free_multihead_attention(block->masked_attention);
        free_multihead_attention(block->cross_attention);
        free_feedforward(block->ff);
        free_layernorm(block->ln1);
        free_layernorm(block->ln2);
        free_layernorm(block->ln3);
        free_dropout(block->dropout);
        free(block);
    }
}

void decoder_block_forward(Tensor* out, Tensor* target, Tensor* encoder_out, DecoderBlock* block, Arena* arena, int training) {
    // Masked self-attention
    Tensor* masked_attn_out = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    multihead_attention_forward(masked_attn_out, target, block->masked_attention, NULL, arena); // Mask would be passed here

    Tensor* dropout_out1 = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    dropout_forward(dropout_out1, masked_attn_out, block->dropout, training);
    
    Tensor* add_norm1 = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    add(add_norm1, target, dropout_out1);
    layernorm_forward(add_norm1, add_norm1, block->ln1);

    // Cross-attention
    Tensor* cross_attn_out = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    multihead_attention_forward(cross_attn_out, add_norm1, block->cross_attention, NULL, arena);

    Tensor* dropout_out2 = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    dropout_forward(dropout_out2, cross_attn_out, block->dropout, training);

    Tensor* add_norm2 = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    add(add_norm2, add_norm1, dropout_out2);
    layernorm_forward(add_norm2, add_norm2, block->ln2);
    
    // Feed-forward
    Tensor* ff_out = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    feedforward_forward(ff_out, add_norm2, block->ff, arena);

    Tensor* dropout_out3 = create_tensor(target->n_dims, target->dims, TENSOR_TYPE_FLOAT, arena);
    dropout_forward(dropout_out3, ff_out, block->dropout, training);

    add(out, add_norm2, dropout_out3);
    layernorm_forward(out, out, block->ln3);
}

Value* decoder_block_forward_ad(Value* tgt, Value* encoder_out, DecoderBlock* block, Arena* arena, int training) {
    // Note: Dropout is omitted in the AD version for now.
    // TODO: The cross attention should use encoder_out for K,V and tgt for Q.
    Value* masked_attn_out = multihead_attention_forward_ad(tgt, block->masked_attention, NULL, arena);
    Value* add1 = add_ad(tgt, masked_attn_out, arena);
    Value* norm1 = layernorm_forward_ad(add1, block->ln1, arena);

    Value* cross_attn_out = multihead_attention_forward_ad(norm1, block->cross_attention, NULL, arena);
    Value* add2 = add_ad(norm1, cross_attn_out, arena);
    Value* norm2 = layernorm_forward_ad(add2, block->ln2, arena);
    
    Value* ff_out = feedforward_forward_ad(norm2, block->ff, arena);
    Value* add3 = add_ad(norm2, ff_out, arena);
    Value* norm3 = layernorm_forward_ad(add3, block->ln3, arena);

    return norm3;
} 