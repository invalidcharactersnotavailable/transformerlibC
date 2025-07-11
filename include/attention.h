#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    int embed_dim, n_heads;
    Tensor *w_q, *w_k, *w_v, *w_o; // Weight matrices
} MultiHeadAttention;

MultiHeadAttention* create_multihead_attention(int embed_dim, int n_heads);
void free_multihead_attention(MultiHeadAttention* mha);
void multihead_attention_forward(Tensor* out, Tensor* q_in, Tensor* k_in, Tensor* v_in, MultiHeadAttention* mha, Tensor* mask);
Value* multihead_attention_forward_ad(Arena* arena, Value* q_in, Value* k_in, Value* v_in, MultiHeadAttention* mha, Tensor* mask);
int save_multihead_attention(MultiHeadAttention* mha, FILE* fp);
int load_multihead_attention(MultiHeadAttention* mha, FILE* fp);
Tensor* split_heads(Arena* arena, Tensor* x, int n_heads);
Tensor* combine_heads(Arena* arena, Tensor* x);

#endif // ATTENTION_H
