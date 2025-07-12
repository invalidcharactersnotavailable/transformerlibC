#include "embedding.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

TokenEmbedding* create_token_embedding(int vocab_size, int embed_dim) {
    TokenEmbedding* emb = (TokenEmbedding*)malloc(sizeof(TokenEmbedding));
    if (!emb) return NULL;
    emb->vocab_size = vocab_size;
    emb->embed_dim = embed_dim;

    int weights_dims[] = {vocab_size, embed_dim};
    emb->weights = create_tensor(2, weights_dims, TENSOR_TYPE_FLOAT);
    if (!emb->weights) {
        free(emb);
        return NULL;
    }

    // xavier uniform initialization
    float limit = sqrtf(6.0f / (vocab_size + embed_dim));
    float* data = (float*)emb->weights->data;
    size_t n = vocab_size * embed_dim;
    // seed random only once per program (should be in main, but fallback here)
    static int seeded = 0;
    if (!seeded) { srand((unsigned int)time(NULL)); seeded = 1; }
    for (size_t i = 0; i < n; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        data[i] = -limit + 2.0f * limit * r;
    }
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

    if (in->n_dims == 2) {
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
    } else {
        assert(in->n_dims == 1);
        int seq_len = in->dims[0];
        int* in_data = (int*)in->data;
        float* out_data = (float*)out->data;
        float* weights_data = (float*)emb->weights->data;

        for (int s = 0; s < seq_len; s++) {
            int token_id = in_data[s];
            memcpy(out_data + s * emb->embed_dim,
                   weights_data + token_id * emb->embed_dim, 
                   emb->embed_dim * sizeof(float));
        }
    }
}

int save_token_embedding(TokenEmbedding* te, FILE* fp) {
    return save_tensor(te->weights, fp);
}

int load_token_embedding(TokenEmbedding* te, FILE* fp) {
    free_tensor(te->weights);
    te->weights = load_tensor(fp);
    return te->weights != NULL;
} 