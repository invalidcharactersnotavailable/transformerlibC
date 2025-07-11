#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

typedef struct {
    int embed_dim;
    int n_heads;
    Tensor *Wq, *Wk, *Wv, *Wo;
} MultiHeadAttention;

MultiHeadAttention* create_multihead_attention(int embed_dim, int n_heads);
void free_multihead_attention(MultiHeadAttention* mha);
void multihead_attention_forward(Tensor* out, Tensor* in, MultiHeadAttention* mha);

#endif // ATTENTION_H
