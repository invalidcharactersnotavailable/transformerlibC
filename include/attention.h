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
void multihead_attention_forward(Tensor* out, Tensor* in, MultiHeadAttention* mha, Tensor* mask, Arena* arena);
Value* multihead_attention_forward_ad(Value* in, MultiHeadAttention* mha, Tensor* mask, Arena* arena);
Tensor* split_heads(Tensor* x, int n_heads);
Tensor* combine_heads(Tensor* x);

#endif // ATTENTION_H
