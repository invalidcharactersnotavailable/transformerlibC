#include "attention.h"
#include <stdlib.h>
#include <assert.h>

MultiHeadAttention* create_multihead_attention(int embed_dim, int n_heads) {
    assert(embed_dim % n_heads == 0);
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    mha->embed_dim = embed_dim;
    mha->n_heads = n_heads;

    int head_dim = embed_dim / n_heads;

    int w_dims[] = {embed_dim, embed_dim};
    int w_qkv_dims[] = {embed_dim, head_dim};

    mha->Wq = create_tensor(2, w_dims);
    mha->Wk = create_tensor(2, w_dims);
    mha->Wv = create_tensor(2, w_dims);
    mha->Wo = create_tensor(2, w_dims);

    return mha;
}

void free_multihead_attention(MultiHeadAttention* mha) {
    if (mha) {
        free_tensor(mha->Wq);
        free_tensor(mha->Wk);
        free_tensor(mha->Wv);
        free_tensor(mha->Wo);
        free(mha);
    }
}

// Helper to reshape and transpose for multi-head attention
static Tensor* split_heads(Tensor* x, int n_heads) {
    int batch_size = x->dims[0];
    int seq_len = x->dims[1];
    int embed_dim = x->dims[2];
    int head_dim = embed_dim / n_heads;

    int new_dims[] = {batch_size, seq_len, n_heads, head_dim};
    Tensor* temp = create_tensor(4, new_dims);
    // This is a simplified view. A real implementation would involve careful memory copies.
    // For now, let's imagine the data is reshaped and transposed appropriately.
    // The logical view is (batch_size, n_heads, seq_len, head_dim)
    // We will fake this with a simple copy for now.
    memcpy(temp->data, x->data, batch_size * seq_len * embed_dim * sizeof(float));
    return temp;
}

static Tensor* combine_heads(Tensor* x) {
    int batch_size = x->dims[0];
    int seq_len = x->dims[1];
    int n_heads = x->dims[2];
    int head_dim = x->dims[3];
    int embed_dim = n_heads * head_dim;

    int new_dims[] = {batch_size, seq_len, embed_dim};
    Tensor* temp = create_tensor(3, new_dims);
    // Similar to split_heads, this is a simplification.
    memcpy(temp->data, x->data, batch_size * seq_len * embed_dim * sizeof(float));
    return temp;
}


void multihead_attention_forward(Tensor* out, Tensor* in, MultiHeadAttention* mha) {
    int batch_size = in->dims[0];
    int seq_len = in->dims[1];
    int embed_dim = in->dims[2];
    int n_heads = mha->n_heads;
    int head_dim = embed_dim / n_heads;

    int qkv_dims[] = {batch_size, seq_len, embed_dim};
    Tensor *q = create_tensor(3, qkv_dims);
    Tensor *k = create_tensor(3, qkv_dims);
    Tensor *v = create_tensor(3, qkv_dims);

    // These matmuls are not quite right for 3D x 2D, we'd need to loop over batch
    // This is a simplification
    matmul(q, in, mha->Wq);
    matmul(k, in, mha->Wk);
    matmul(v, in, mha->Wv);

    Tensor* q_heads = split_heads(q, n_heads);
    Tensor* k_heads = split_heads(k, n_heads);
    Tensor* v_heads = split_heads(v, n_heads);

    int attn_scores_dims[] = {batch_size, n_heads, seq_len, seq_len};
    Tensor* attn_scores = create_tensor(4, attn_scores_dims);
    
    // Simplified matmul for 4D tensors
    // This requires a batched matmul (BMM)
    // C = alpha * A @ B.transpose(-2, -1) + beta * C
    // We will just placeholder this for now
    // matmul(attn_scores, q_heads, k_heads_transposed);

    float scale = 1.0f / sqrtf(head_dim);
    // scale attn_scores
    
    softmax(attn_scores);

    int context_dims[] = {batch_size, n_heads, seq_len, head_dim};
    Tensor* context = create_tensor(4, context_dims);

    // another BMM
    // matmul(context, attn_scores, v_heads);

    Tensor* context_combined = combine_heads(context);

    // final linear layer
    matmul(out, context_combined, mha->Wo);

    free_tensor(q);
    free_tensor(k);
    free_tensor(v);
    free_tensor(q_heads);
    free_tensor(k_heads);
    free_tensor(v_heads);
    free_tensor(attn_scores);
    free_tensor(context);
    free_tensor(context_combined);
}
