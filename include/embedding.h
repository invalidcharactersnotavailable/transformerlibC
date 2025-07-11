#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

typedef struct {
    int vocab_size;
    int embed_dim;
    Tensor* weights;
} TokenEmbedding;

/**
 * create_token_embedding - allocate a token embedding layer
 * @vocab_size: vocabulary size
 * @embed_dim: embedding dimension
 * returns: pointer to token embedding or NULL on failure
 */
TokenEmbedding* create_token_embedding(int vocab_size, int embed_dim);

/**
 * free_token_embedding - free a token embedding layer
 * @te: pointer to token embedding
 */
void free_token_embedding(TokenEmbedding* te);

/**
 * token_embedding_forward - run forward pass of token embedding
 * @out: output tensor
 * @in: input tensor (int tokens)
 * @te: token embedding
 */
void token_embedding_forward(Tensor* out, Tensor* in, TokenEmbedding* te);

/**
 * save_token_embedding - write token embedding to file
 * @te: token embedding
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int save_token_embedding(TokenEmbedding* te, FILE* fp);

/**
 * load_token_embedding - read token embedding from file
 * @te: token embedding
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int load_token_embedding(TokenEmbedding* te, FILE* fp);

#endif // EMBEDDING_H 