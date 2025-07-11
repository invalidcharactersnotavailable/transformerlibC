#include "attention.h"
#include "autodiff.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

MultiHeadAttention* create_multihead_attention(int embed_dim, int n_heads) {
    assert(embed_dim % n_heads == 0);
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    mha->embed_dim = embed_dim;
    mha->n_heads = n_heads;

    int w_dims[] = {embed_dim, embed_dim};

    mha->w_q = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT, NULL);
    mha->w_k = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT, NULL);
    mha->w_v = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT, NULL);
    mha->w_o = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT, NULL);

    return mha;
}

void free_multihead_attention(MultiHeadAttention* mha) {
    if (mha) {
        free_tensor(mha->w_q);
        free_tensor(mha->w_k);
        free_tensor(mha->w_v);
        free_tensor(mha->w_o);
        free(mha);
    }
}

// Helper to reshape and transpose for multi-head attention
Tensor* split_heads_arena(Tensor* x, int n_heads, Arena* arena) {
    int batch_size = x->dims[0];
    int seq_len = x->dims[1];
    int embed_dim = x->dims[2];
    int head_dim = embed_dim / n_heads;

    int reshaped_dims[] = {batch_size, seq_len, n_heads, head_dim};
    Tensor* reshaped = create_tensor(4, reshaped_dims, TENSOR_TYPE_FLOAT, arena);
    memcpy(reshaped->data, x->data, batch_size * seq_len * embed_dim * sizeof(float));

    int transposed_dims[] = {batch_size, n_heads, seq_len, head_dim};
    Tensor* transposed = create_tensor(4, transposed_dims, TENSOR_TYPE_FLOAT, arena);
    transpose(transposed, reshaped, 1, 2);

    return transposed;
}

Tensor* combine_heads_arena(Tensor* x, Arena* arena) {
    int batch_size = x->dims[0];
    int n_heads = x->dims[1];
    int seq_len = x->dims[2];
    int head_dim = x->dims[3];
    int embed_dim = n_heads * head_dim;

    int transposed_dims[] = {batch_size, seq_len, n_heads, head_dim};
    Tensor* transposed = create_tensor(4, transposed_dims, TENSOR_TYPE_FLOAT, arena);
    transpose(transposed, x, 1, 2);

    int final_dims[] = {batch_size, seq_len, embed_dim};
    Tensor* final_tensor = create_tensor(3, final_dims, TENSOR_TYPE_FLOAT, arena);
    memcpy(final_tensor->data, transposed->data, batch_size * seq_len * embed_dim * sizeof(float));

    return final_tensor;
}


void multihead_attention_forward(Tensor* out, Tensor* in, MultiHeadAttention* mha, Tensor* mask, Arena* arena) {
    int batch_size = in->dims[0];
    int seq_len = in->dims[1];
    int embed_dim = mha->embed_dim;

    Tensor* q = create_tensor(3, (int[]){batch_size, seq_len, embed_dim}, TENSOR_TYPE_FLOAT, arena);
    Tensor* k = create_tensor(3, (int[]){batch_size, seq_len, embed_dim}, TENSOR_TYPE_FLOAT, arena);
    Tensor* v = create_tensor(3, (int[]){batch_size, seq_len, embed_dim}, TENSOR_TYPE_FLOAT, arena);

    matmul(q, in, mha->w_q);
    matmul(k, in, mha->w_k);
    matmul(v, in, mha->w_v);

    Tensor* q_split = split_heads_arena(q, mha->n_heads, arena);
    Tensor* k_split = split_heads_arena(k, mha->n_heads, arena);
    Tensor* v_split = split_heads_arena(v, mha->n_heads, arena);

    int k_t_dims[] = {k_split->dims[0], k_split->dims[1], k_split->dims[3], k_split->dims[2]};
    Tensor* k_t = create_tensor(4, k_t_dims, TENSOR_TYPE_FLOAT, arena);
    transpose(k_t, k_split, 2, 3);

    int scores_dims[] = {q_split->dims[0], q_split->dims[1], q_split->dims[2], k_t->dims[3]};
    Tensor* scores = create_tensor(4, scores_dims, TENSOR_TYPE_FLOAT, arena);
    matmul(scores, q_split, k_t);

    if (mask) {
        add(scores, scores, mask);
    }
    
    softmax(scores);

    Tensor* attn_out_split = create_tensor(4, q_split->dims, TENSOR_TYPE_FLOAT, arena);
    matmul(attn_out_split, scores, v_split);
    
    Tensor* attn_out = combine_heads_arena(attn_out_split, arena);
    matmul(out, attn_out, mha->w_o);
}

void backward_attention(Value* v) {
    // Placeholder for attention backward pass
}

Value* multihead_attention_forward_ad(Value* in, MultiHeadAttention* mha, Tensor* mask, Arena* arena) {
    // NOTE: This is a simplified single-head self-attention implementation for now.
    // Multi-head attention would require reshape and more complex transpose operations.
    int embed_dim = mha->embed_dim;
    float scale_factor = 1.0f / sqrtf(embed_dim);

    Value* wq_val = create_value(mha->w_q, NULL, 0, NULL, arena, NULL);
    Value* wk_val = create_value(mha->w_k, NULL, 0, NULL, arena, NULL);
    Value* wv_val = create_value(mha->w_v, NULL, 0, NULL, arena, NULL);
    Value* wo_val = create_value(mha->w_o, NULL, 0, NULL, arena, NULL);

    // 1. Project to Q, K, V
    Value* q = matmul_ad(in, wq_val, arena);
    Value* k = matmul_ad(in, wk_val, arena);
    Value* v = matmul_ad(in, wv_val, arena);

    // 2. Transpose K: (B, S, E) -> (B, E, S)
    Value* k_t = transpose_ad(k, 1, 2, arena);

    // 3. Q @ K^T
    Value* scores = matmul_ad(q, k_t, arena);

    // 4. Scale scores
    Value* scaled_scores = scale_ad(scores, scale_factor, arena);

    // 5. Mask (optional)
    if (mask) {
        Value* mask_val = create_value(mask, NULL, 0, NULL, arena, NULL);
        scaled_scores = add_ad(scaled_scores, mask_val, arena);
    }

    // 6. Softmax
    Value* attn_probs = softmax_ad(scaled_scores, arena);

    // 7. Attn_probs @ V
    Value* context = matmul_ad(attn_probs, v, arena);

    // 8. Final projection
    Value* out = matmul_ad(context, wo_val, arena);

    return out;
}
