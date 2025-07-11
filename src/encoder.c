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

void encoder_block_forward(Tensor* out, Tensor* in, EncoderBlock* block, int training) {
    // This function is not arena-aware, and will leak memory if used repeatedly.
    // It's not used in the main training loop.
    // Self-attention part
    Tensor* attn_out = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    multihead_attention_forward(attn_out, in, in, in, block->attention, NULL);
    // Dropout, Add & Norm
    Tensor* dropout_out = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    dropout_forward(dropout_out, attn_out, block->dropout, training);
    Tensor* add_norm1 = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    add(add_norm1, in, dropout_out);
    layernorm_forward(add_norm1, add_norm1, block->ln1);
    // Feed-forward part
    Tensor* ff_out = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    feedforward_forward(ff_out, add_norm1, block->ff);
    // Add & Norm
    dropout_forward(out, ff_out, block->dropout, training);
    free_tensor(attn_out);
    free_tensor(dropout_out);
    free_tensor(add_norm1);
    free_tensor(ff_out);
}

Value* encoder_block_forward_ad(Arena* arena, Value* in, EncoderBlock* block, int training) {
    Value* attn_out = multihead_attention_forward_ad(arena, in, in, in, block->attention, NULL);
    Value* dropout_out1 = dropout_forward_ad(arena, attn_out, block->dropout, training);
    Value* add1 = add_ad(arena, in, dropout_out1);
    Value* norm1 = layernorm_forward_ad(arena, add1, block->ln1);

    Value* ff_out = feedforward_forward_ad(arena, norm1, block->ff);
    Value* dropout_out2 = dropout_forward_ad(arena, ff_out, block->dropout, training);
    Value* add2 = add_ad(arena, norm1, dropout_out2);
    Value* out = layernorm_forward_ad(arena, add2, block->ln2);
    
    return out;
}

int save_encoder_block(EncoderBlock* block, FILE* fp) {
    if (!save_multihead_attention(block->attention, fp)) return 0;
    if (!save_feedforward(block->ff, fp)) return 0;
    if (!save_layernorm(block->ln1, fp)) return 0;
    if (!save_layernorm(block->ln2, fp)) return 0;
    return 1;
}

int load_encoder_block(EncoderBlock* block, FILE* fp) {
    if (!load_multihead_attention(block->attention, fp)) return 0;
    if (!load_feedforward(block->ff, fp)) return 0;
    if (!load_layernorm(block->ln1, fp)) return 0;
    if (!load_layernorm(block->ln2, fp)) return 0;
    return 1;
}
