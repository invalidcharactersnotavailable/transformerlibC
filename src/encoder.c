#include "encoder.h"
#include <stdlib.h>

EncoderBlock* create_encoder_block(int embed_dim, int n_heads, int ff_hidden_dim) {
    EncoderBlock* block = (EncoderBlock*)malloc(sizeof(EncoderBlock));
    if (!block) { fprintf(stderr, "[ERR] malloc failed for EncoderBlock\n"); return NULL; }
    block->embed_dim = embed_dim;
    block->n_heads = n_heads;
    block->ff_hidden_dim = ff_hidden_dim;
    block->attention = create_multihead_attention(embed_dim, n_heads);
    if (!block->attention) { fprintf(stderr, "[ERR] create_multihead_attention failed in EncoderBlock\n"); free(block); return NULL; }
    block->ff = create_feedforward(embed_dim, ff_hidden_dim);
    if (!block->ff) { fprintf(stderr, "[ERR] create_feedforward failed in EncoderBlock\n"); free_multihead_attention(block->attention); free(block); return NULL; }
    block->ln1 = create_layernorm(embed_dim);
    if (!block->ln1) { fprintf(stderr, "[ERR] create_layernorm failed for ln1 in EncoderBlock\n"); free_feedforward(block->ff); free_multihead_attention(block->attention); free(block); return NULL; }
    block->ln2 = create_layernorm(embed_dim);
    if (!block->ln2) { fprintf(stderr, "[ERR] create_layernorm failed for ln2 in EncoderBlock\n"); free_layernorm(block->ln1); free_feedforward(block->ff); free_multihead_attention(block->attention); free(block); return NULL; }
    block->dropout = create_dropout(0.1); // Default 0.1 dropout rate
    if (!block->dropout) { fprintf(stderr, "[ERR] create_dropout failed in EncoderBlock\n"); free_layernorm(block->ln2); free_layernorm(block->ln1); free_feedforward(block->ff); free_multihead_attention(block->attention); free(block); return NULL; }
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
    // self-attention
    Tensor* attn_out = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    multihead_attention_forward(attn_out, in, in, in, block->attention, NULL);
    // dropout after attention
    Tensor* dropout_out1 = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    dropout_forward(dropout_out1, attn_out, block->dropout, training);
    // first residual connection and layernorm
    Tensor* res1 = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    add(res1, in, dropout_out1);
    Tensor* norm1 = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    layernorm_forward(norm1, res1, block->ln1);
    // feedforward
    Tensor* ff_out = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    feedforward_forward(ff_out, norm1, block->ff);
    // dropout after feedforward
    Tensor* dropout_out2 = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    dropout_forward(dropout_out2, ff_out, block->dropout, training);
    // second residual connection and layernorm
    Tensor* res2 = create_tensor(NULL, in->n_dims, in->dims, TENSOR_TYPE_FLOAT);
    add(res2, norm1, dropout_out2);
    layernorm_forward(out, res2, block->ln2);
    // free intermediates
    free_tensor(attn_out);
    free_tensor(dropout_out1);
    free_tensor(res1);
    free_tensor(norm1);
    free_tensor(ff_out);
    free_tensor(dropout_out2);
    free_tensor(res2);
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
