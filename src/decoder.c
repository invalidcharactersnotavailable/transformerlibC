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

void decoder_block_forward(Tensor* out, Tensor* tgt_in, Tensor* encoder_out, DecoderBlock* block, int training) {
    Tensor* masked_attn_out = create_tensor(NULL, tgt_in->n_dims, tgt_in->dims, TENSOR_TYPE_FLOAT);
    multihead_attention_forward(masked_attn_out, tgt_in, tgt_in, tgt_in, block->masked_attention, NULL); // TODO: Add mask
    Tensor* cross_attn_out = create_tensor(NULL, tgt_in->n_dims, tgt_in->dims, TENSOR_TYPE_FLOAT);
    multihead_attention_forward(cross_attn_out, masked_attn_out, encoder_out, encoder_out, block->cross_attention, NULL);
    Tensor* dropout_out = create_tensor(NULL, tgt_in->n_dims, tgt_in->dims, TENSOR_TYPE_FLOAT);
    dropout_forward(dropout_out, cross_attn_out, block->dropout, training);
    Tensor* add_norm1 = create_tensor(NULL, tgt_in->n_dims, tgt_in->dims, TENSOR_TYPE_FLOAT);
    add(add_norm1, tgt_in, dropout_out);
    layernorm_forward(add_norm1, add_norm1, block->ln1);
    Tensor* ff_out = create_tensor(NULL, tgt_in->n_dims, tgt_in->dims, TENSOR_TYPE_FLOAT);
    feedforward_forward(ff_out, add_norm1, block->ff);
    dropout_forward(out, ff_out, block->dropout, training);
    free_tensor(masked_attn_out);
    free_tensor(cross_attn_out);
    free_tensor(dropout_out);
    free_tensor(add_norm1);
    free_tensor(ff_out);
}

Value* decoder_block_forward_ad(Arena* arena, Value* tgt, Value* encoder_out, DecoderBlock* block, int training, Tensor* look_ahead_mask) {
    // 1. Masked Self-Attention sublayer
    Value* masked_attn_out = multihead_attention_forward_ad(arena, tgt, tgt, tgt, block->masked_attention, look_ahead_mask);
    Value* dropout1_out = dropout_forward_ad(arena, masked_attn_out, block->dropout, training);
    Value* add1 = add_ad(arena, tgt, dropout1_out);
    Value* norm1 = layernorm_forward_ad(arena, add1, block->ln1);

    // 2. Cross-Attention sublayer
    Value* cross_attn_out = multihead_attention_forward_ad(arena, norm1, encoder_out, encoder_out, block->cross_attention, NULL);
    Value* dropout2_out = dropout_forward_ad(arena, cross_attn_out, block->dropout, training);
    Value* add2 = add_ad(arena, norm1, dropout2_out);
    Value* norm2 = layernorm_forward_ad(arena, add2, block->ln2);

    // 3. Feed-Forward sublayer
    Value* ff_out = feedforward_forward_ad(arena, norm2, block->ff);
    Value* dropout3_out = dropout_forward_ad(arena, ff_out, block->dropout, training);
    Value* add3 = add_ad(arena, norm2, dropout3_out);
    Value* out = layernorm_forward_ad(arena, add3, block->ln3);

    return out;
} 

int save_decoder_block(DecoderBlock* block, FILE* fp) {
    if (!save_multihead_attention(block->masked_attention, fp)) return 0;
    if (!save_multihead_attention(block->cross_attention, fp)) return 0;
    if (!save_feedforward(block->ff, fp)) return 0;
    if (!save_layernorm(block->ln1, fp)) return 0;
    if (!save_layernorm(block->ln2, fp)) return 0;
    if (!save_layernorm(block->ln3, fp)) return 0;
    return 1;
}

int load_decoder_block(DecoderBlock* block, FILE* fp) {
    if (!load_multihead_attention(block->masked_attention, fp)) return 0;
    if (!load_multihead_attention(block->cross_attention, fp)) return 0;
    if (!load_feedforward(block->ff, fp)) return 0;
    if (!load_layernorm(block->ln1, fp)) return 0;
    if (!load_layernorm(block->ln2, fp)) return 0;
    if (!load_layernorm(block->ln3, fp)) return 0;
    return 1;
} 