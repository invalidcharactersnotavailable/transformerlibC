#include "embedding.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

TokenEmbedding* create_token_embedding(int vocab_size, int embed_dim) {
    TokenEmbedding* emb = (TokenEmbedding*)malloc(sizeof(TokenEmbedding));
    emb->vocab_size = vocab_size;
    emb->embed_dim = embed_dim;

    int weights_dims[] = {vocab_size, embed_dim};
    emb->weights = create_tensor(2, weights_dims, TENSOR_TYPE_FLOAT, NULL);

    // TODO: Initialize weights with some distribution (e.g., Xavier)
    return emb;
}

void free_token_embedding(TokenEmbedding* emb) {
    if (emb) {
        free_tensor(emb->weights);
        free(emb);
    }
}

void token_embedding_forward(Tensor* out, Tensor* in, TokenEmbedding* emb) {
    assert(out->dtype == TENSOR_TYPE_FLOAT);
    assert(in->dtype == TENSOR_TYPE_INT);
    int batch_size = in->dims[0];
    int seq_len = in->dims[1];
    int* in_data = (int*)in->data;
    float* out_data = (float*)out->data;
    float* weights_data = (float*)emb->weights->data;

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token_id = in_data[b * seq_len + s];
            memcpy(out_data + (b * seq_len + s) * emb->embed_dim, 
                   weights_data + token_id * emb->embed_dim, 
                   emb->embed_dim * sizeof(float));
        }
    }
} 