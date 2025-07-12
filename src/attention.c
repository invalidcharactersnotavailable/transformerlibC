#include "attention.h"
#include "autodiff.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int mha_seeded = 0;

MultiHeadAttention* create_multihead_attention(int embed_dim, int n_heads) {
    assert(embed_dim % n_heads == 0);
    if (!mha_seeded) { srand((unsigned int)time(NULL)); mha_seeded = 1; }
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    if (!mha) { fprintf(stderr, "[ERR] malloc failed for MultiHeadAttention\n"); return NULL; }
    mha->embed_dim = embed_dim;
    mha->n_heads = n_heads;
    int w_dims[] = {embed_dim, embed_dim};
    mha->w_q = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT);
    mha->w_k = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT);
    mha->w_v = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT);
    mha->w_o = create_tensor(2, w_dims, TENSOR_TYPE_FLOAT);
    if (!mha->w_q || !mha->w_k || !mha->w_v || !mha->w_o) {
        fprintf(stderr, "[ERR] create_tensor failed for attention weights\n");
        free_tensor(mha->w_q); free_tensor(mha->w_k); free_tensor(mha->w_v); free_tensor(mha->w_o); free(mha); return NULL;
    }
    // xavier uniform initialization
    float limit = sqrtf(6.0f / (embed_dim + embed_dim));
    float* params[] = {(float*)mha->w_q->data, (float*)mha->w_k->data, (float*)mha->w_v->data, (float*)mha->w_o->data};
    size_t n = embed_dim * embed_dim;
    for (int t = 0; t < 4; t++) {
        for (size_t i = 0; i < n; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            params[t][i] = -limit + 2.0f * limit * r;
        }
    }
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

Tensor* split_heads(Tensor* x, int n_heads) {
    assert(x && x->n_dims == 3);
    int batch_size = x->dims[0];
    int seq_len = x->dims[1];
    int embed_dim = x->dims[2];
    int head_dim = embed_dim / n_heads;
    assert(embed_dim % n_heads == 0);
    int reshaped_dims[] = {batch_size, seq_len, n_heads, head_dim};
    Tensor* reshaped = create_tensor(4, reshaped_dims, TENSOR_TYPE_FLOAT);
    if (!reshaped) { fprintf(stderr, "[ERR] create_tensor failed in split_heads (reshaped)\n"); return NULL; }
    memcpy(reshaped->data, x->data, batch_size * seq_len * embed_dim * sizeof(float));
    int transposed_dims[] = {batch_size, n_heads, seq_len, head_dim};
    Tensor* transposed = create_tensor(4, transposed_dims, TENSOR_TYPE_FLOAT);
    if (!transposed) { fprintf(stderr, "[ERR] create_tensor failed in split_heads (transposed)\n"); free_tensor(reshaped); return NULL; }
    transpose(transposed, reshaped, 1, 2);
    free_tensor(reshaped);
    return transposed;
}

Tensor* combine_heads(Tensor* x) {
    assert(x && x->n_dims == 4);
    int batch_size = x->dims[0];
    int n_heads = x->dims[1];
    int seq_len = x->dims[2];
    int head_dim = x->dims[3];
    int embed_dim = n_heads * head_dim;
    int transposed_dims[] = {batch_size, seq_len, n_heads, head_dim};
    Tensor* transposed = create_tensor(4, transposed_dims, TENSOR_TYPE_FLOAT);
    if (!transposed) { fprintf(stderr, "[ERR] create_tensor failed in combine_heads (transposed)\n"); return NULL; }
    transpose(transposed, x, 1, 2);
    int final_dims[] = {batch_size, seq_len, embed_dim};
    Tensor* final_tensor = create_tensor(3, final_dims, TENSOR_TYPE_FLOAT);
    if (!final_tensor) { fprintf(stderr, "[ERR] create_tensor failed in combine_heads (final_tensor)\n"); free_tensor(transposed); return NULL; }
    memcpy(final_tensor->data, transposed->data, batch_size * seq_len * embed_dim * sizeof(float));
    free_tensor(transposed);
    return final_tensor;
}

void multihead_attention_forward(Tensor* out, Tensor* q_in, Tensor* k_in, Tensor* v_in, MultiHeadAttention* mha, Tensor* mask) {
    assert(out && q_in && k_in && v_in && mha);
    int batch_size = q_in->dims[0];
    int seq_len_q = q_in->dims[1];
    int seq_len_kv = k_in->dims[1];
    int embed_dim = mha->embed_dim;
    int n_heads = mha->n_heads;
    float scale = 1.0f / sqrtf((float)(embed_dim / n_heads));
    Tensor* q = create_tensor(3, (int[]){batch_size, seq_len_q, embed_dim}, TENSOR_TYPE_FLOAT);
    Tensor* k = create_tensor(3, (int[]){batch_size, seq_len_kv, embed_dim}, TENSOR_TYPE_FLOAT);
    Tensor* v = create_tensor(3, (int[]){batch_size, seq_len_kv, embed_dim}, TENSOR_TYPE_FLOAT);
    if (!q || !k || !v) { free_tensor(q); free_tensor(k); free_tensor(v); return; }
    matmul(q, q_in, mha->w_q);
    matmul(k, k_in, mha->w_k);
    matmul(v, v_in, mha->w_v);
    Tensor* q_split = split_heads(q, n_heads);
    Tensor* k_split = split_heads(k, n_heads);
    Tensor* v_split = split_heads(v, n_heads);
    if (!q_split || !k_split || !v_split) { free_tensor(q); free_tensor(k); free_tensor(v); free_tensor(q_split); free_tensor(k_split); free_tensor(v_split); return; }
    int k_t_dims[] = {k_split->dims[0], k_split->dims[1], k_split->dims[3], k_split->dims[2]};
    Tensor* k_t = create_tensor(4, k_t_dims, TENSOR_TYPE_FLOAT);
    if (!k_t) { free_tensor(q); free_tensor(k); free_tensor(v); free_tensor(q_split); free_tensor(k_split); free_tensor(v_split); return; }
    transpose(k_t, k_split, 2, 3);
    int scores_dims[] = {q_split->dims[0], q_split->dims[1], q_split->dims[2], k_t->dims[3]};
    Tensor* scores = create_tensor(4, scores_dims, TENSOR_TYPE_FLOAT);
    if (!scores) { free_tensor(q); free_tensor(k); free_tensor(v); free_tensor(q_split); free_tensor(k_split); free_tensor(v_split); free_tensor(k_t); return; }
    matmul(scores, q_split, k_t);
    // scale scores
    for (int i = 0; i < scores->dims[0] * scores->dims[1] * scores->dims[2] * scores->dims[3]; i++) {
        ((float*)scores->data)[i] *= scale;
    }
    if (mask) {
        assert(mask->n_dims == scores->n_dims);
        for (int i = 0; i < scores->n_dims; i++) assert(mask->dims[i] == scores->dims[i]);
        add(scores, scores, mask);
    }
    Tensor* attn_probs = create_tensor(4, scores->dims, TENSOR_TYPE_FLOAT);
    softmax(attn_probs, scores);
    Tensor* attn_out_split = create_tensor(4, q_split->dims, TENSOR_TYPE_FLOAT);
    if (!attn_out_split) { free_tensor(q); free_tensor(k); free_tensor(v); free_tensor(q_split); free_tensor(k_split); free_tensor(v_split); free_tensor(k_t); free_tensor(scores); return; }
    matmul(attn_out_split, attn_probs, v_split);
    Tensor* attn_out = combine_heads(attn_out_split);
    if (!attn_out) { free_tensor(q); free_tensor(k); free_tensor(v); free_tensor(q_split); free_tensor(k_split); free_tensor(v_split); free_tensor(k_t); free_tensor(scores); free_tensor(attn_out_split); return; }
    matmul(out, attn_out, mha->w_o);
    free_tensor(q);
    free_tensor(k);
    free_tensor(v);
    free_tensor(q_split);
    free_tensor(k_split);
    free_tensor(v_split);
    free_tensor(k_t);
    free_tensor(scores);
    free_tensor(attn_out_split);
    free_tensor(attn_out);
}

void backward_attention(Value* v) {
    (void)v; // Unused parameter
    // This is a composite operation. Its gradient is handled by the backward passes
    // of the elementary operations it's composed of (matmul, add, softmax, etc.).
    // This function is not strictly necessary if the forward pass builds the graph correctly.
}

// Helper op for splitting heads
Value* split_heads_ad(Value* in, int n_heads) {
    int batch_size = in->data->dims[0];
    int seq_len = in->data->dims[1];
    int embed_dim = in->data->dims[2];
    int head_dim = embed_dim / n_heads;

    // Reshape and transpose: (B, S, E) -> (B, S, H, D) -> (B, H, S, D)
    int reshaped_dims[] = {batch_size, seq_len, n_heads, head_dim};
    Value* reshaped = reshape_ad(in, 4, reshaped_dims);
    
    Value* transposed = transpose_ad(reshaped, 1, 2);
    // free_value(reshaped); // Defer freeing to free_graph

    return transposed;
}

// Helper op for combining heads
Value* combine_heads_ad(Value* in) {
    // Transpose and reshape: (B, H, S, D) -> (B, S, H, D) -> (B, S, E)
    Value* transposed = transpose_ad(in, 1, 2);

    int batch_size = transposed->data->dims[0];
    int seq_len = transposed->data->dims[1];
    int n_heads = transposed->data->dims[2];
    int head_dim = transposed->data->dims[3];
    int embed_dim = n_heads * head_dim;

    int final_dims[] = {batch_size, seq_len, embed_dim};
    Value* reshaped = reshape_ad(transposed, 3, final_dims);
    // free_value(transposed); // Defer freeing to free_graph

    return reshaped;
}


Value* multihead_attention_forward_ad(Value* q_in, Value* k_in, Value* v_in, MultiHeadAttention* mha, Tensor* mask) {
    int embed_dim = mha->embed_dim;
    int n_heads = mha->n_heads;
    float scale_factor = 1.0f / sqrtf(embed_dim / n_heads);

    Value* wq_val = create_value(mha->w_q, NULL, 0, NULL, NULL);
    Value* wk_val = create_value(mha->w_k, NULL, 0, NULL, NULL);
    Value* wv_val = create_value(mha->w_v, NULL, 0, NULL, NULL);
    Value* wo_val = create_value(mha->w_o, NULL, 0, NULL, NULL);

    // 1. Project to Q, K, V
    Value* q_proj = matmul_ad(q_in, wq_val);
    Value* k_proj = matmul_ad(k_in, wk_val);
    Value* v_proj = matmul_ad(v_in, wv_val);

    // 2. Split heads: (B, S, E) -> (B, H, S, D)
    Value* q = split_heads_ad(q_proj, n_heads);
    Value* k = split_heads_ad(k_proj, n_heads);
    Value* v = split_heads_ad(v_proj, n_heads);

    // 3. Transpose K for matmul: (B, H, S, D) -> (B, H, D, S)
    Value* k_t = transpose_ad(k, 2, 3);

    // 4. Q @ K^T -> scores
    Value* scores = matmul_ad(q, k_t); // (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)

    // 5. Scale scores
    Value* scaled_scores = scale_ad(scores, scale_factor);

    // 6. Mask (optional)
    if (mask) {
        Value* mask_val = create_value(mask, NULL, 0, NULL, NULL);
        scaled_scores = add_ad(scaled_scores, mask_val);
    }

    // 7. Softmax
    Value* attn_probs = softmax_ad(scaled_scores);

    // 8. Attn_probs @ V
    Value* context_split = matmul_ad(attn_probs, v); // (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)

    // 9. Combine heads: (B, H, S, D) -> (B, S, E)
    Value* context = combine_heads_ad(context_split);

    // 10. Final projection
    Value* out = matmul_ad(context, wo_val); // (B, S, E) @ (E, E) -> (B, S, E)
    
    // The graph of intermediate values will be freed later by calling free_graph()
    // on the final loss value.

    return out;
}

int save_multihead_attention(MultiHeadAttention* mha, FILE* fp) {
    if (!save_tensor(mha->w_q, fp)) return 0;
    if (!save_tensor(mha->w_k, fp)) return 0;
    if (!save_tensor(mha->w_v, fp)) return 0;
    if (!save_tensor(mha->w_o, fp)) return 0;
    return 1;
}

int load_multihead_attention(MultiHeadAttention* mha, FILE* fp) {
    // Free existing tensors before loading new ones
    free_tensor(mha->w_q);
    free_tensor(mha->w_k);
    free_tensor(mha->w_v);
    free_tensor(mha->w_o);

    mha->w_q = load_tensor(fp);
    mha->w_k = load_tensor(fp);
    mha->w_v = load_tensor(fp);
    mha->w_o = load_tensor(fp);

    if (!mha->w_q || !mha->w_k || !mha->w_v || !mha->w_o) {
        return 0; // Failure
    }
    return 1;
}
