#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

typedef struct {
    int vocab_size;
    int embed_dim;
    Tensor* weights;
} TokenEmbedding;

TokenEmbedding* create_token_embedding(int vocab_size, int embed_dim);
void free_token_embedding(TokenEmbedding* emb);
void token_embedding_forward(Tensor* out, Tensor* in, TokenEmbedding* emb);

#endif // EMBEDDING_H 