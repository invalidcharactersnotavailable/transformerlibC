#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

typedef struct {
    int vocab_size;
    int embed_dim;
    Tensor* weights;
} TokenEmbedding;

TokenEmbedding* create_token_embedding(int vocab_size, int embed_dim);
void free_token_embedding(TokenEmbedding* te);
void token_embedding_forward(Tensor* out, Tensor* in, TokenEmbedding* te);
int save_token_embedding(TokenEmbedding* te, FILE* fp);
int load_token_embedding(TokenEmbedding* te, FILE* fp);

#endif // EMBEDDING_H 