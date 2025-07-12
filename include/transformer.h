#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "encoder.h"
#include "decoder.h"
#include "tensor.h"
#include "embedding.h"
#include "autodiff.h"
#include <math.h>

typedef struct {
    int n_layers;
    int vocab_size;
    int embed_dim;
    int max_seq_len;
    TokenEmbedding* embedding;
    Tensor* pos_encoding;
    EncoderBlock** encoder_layers;
    DecoderBlock** decoder_layers;
    Tensor* output_layer;
    Tensor* look_ahead_mask;
} Transformer;

Transformer* create_transformer(int n_layers, int vocab_size, int max_seq_len, int embed_dim, int n_heads, int ff_hidden_dim);
void free_transformer(Transformer* t);

void transformer_forward(Tensor* out, Tensor* src_in, Tensor* tgt_in, Transformer* t, int training);
Value* transformer_forward_ad(Tensor* src, Tensor* tgt, Transformer* model, int is_training);
void create_positional_encoding(Tensor* pe, int max_seq_len, int embed_dim);
void create_look_ahead_mask(Tensor* mask, int seq_len);
int save_transformer(Transformer* t, const char* path);
int load_transformer(Transformer* t, const char* path);
long get_transformer_param_count(Transformer* t);

#endif // TRANSFORMER_H
