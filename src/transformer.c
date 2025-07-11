#include "transformer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h> // Added for assert

void create_positional_encoding(Tensor* pe, int max_seq_len, int embed_dim) {
    assert(pe->dtype == TENSOR_TYPE_FLOAT);
    float* pe_data = (float*)pe->data;
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < embed_dim / 2; i++) {
            float div_term = exp(i * -log(10000.0) / (embed_dim / 2));
            int base_idx = pos * embed_dim;
            pe_data[base_idx + 2 * i] = sin(pos * div_term);
            pe_data[base_idx + 2 * i + 1] = cos(pos * div_term);
        }
    }
}

Transformer* create_transformer(int n_layers, int vocab_size, int max_seq_len, int embed_dim, int n_heads, int ff_hidden_dim) {
    Transformer* t = (Transformer*)malloc(sizeof(Transformer));
    if (!t) return NULL;

    t->n_layers = n_layers;
    t->vocab_size = vocab_size;
    t->max_seq_len = max_seq_len;
    t->embed_dim = embed_dim;

    t->embedding = create_token_embedding(vocab_size, embed_dim);

    int pos_dims[] = {max_seq_len, embed_dim};
    t->pos_encoding = create_tensor(2, pos_dims, TENSOR_TYPE_FLOAT, NULL);
    create_positional_encoding(t->pos_encoding, max_seq_len, embed_dim);

    t->encoder_layers = (EncoderBlock**)malloc(n_layers * sizeof(EncoderBlock*));
    t->decoder_layers = (DecoderBlock**)malloc(n_layers * sizeof(DecoderBlock*));
    for (int i = 0; i < n_layers; i++) {
        t->encoder_layers[i] = create_encoder_block(embed_dim, n_heads, ff_hidden_dim);
        t->decoder_layers[i] = create_decoder_block(embed_dim, n_heads, ff_hidden_dim);
    }
    
    int out_dims[] = {embed_dim, vocab_size};
    t->output_layer = create_tensor(2, out_dims, TENSOR_TYPE_FLOAT, NULL);

    return t;
}

void free_transformer(Transformer* t) {
    free_token_embedding(t->embedding);
    free_tensor(t->pos_encoding);
    for (int i = 0; i < t->n_layers; i++) {
        free_encoder_block(t->encoder_layers[i]);
        free_decoder_block(t->decoder_layers[i]);
    }
    free(t->encoder_layers);
    free(t->decoder_layers);
    free_tensor(t->output_layer);
    free(t);
}

void transformer_forward(Tensor* out, Tensor* src_in, Tensor* tgt_in, Transformer* t, Arena* arena, int training) {
    // Encoder forward pass
    Tensor* encoder_out = create_tensor(src_in->n_dims, src_in->dims, TENSOR_TYPE_FLOAT, arena);
    // Note: This is a simplified forward pass for the encoder
    // A proper implementation would handle embeddings and sequential layer processing
    encoder_block_forward(encoder_out, src_in, t->encoder_layers[0], arena, training);
    for (int i = 1; i < t->n_layers; i++) {
        encoder_block_forward(encoder_out, encoder_out, t->encoder_layers[i], arena, training);
    }

    // Decoder forward pass
    Tensor* decoder_out = create_tensor(tgt_in->n_dims, tgt_in->dims, TENSOR_TYPE_FLOAT, arena);
    // Note: Simplified forward pass for the decoder
    decoder_block_forward(decoder_out, tgt_in, encoder_out, t->decoder_layers[0], arena, training);
    for (int i = 1; i < t->n_layers; i++) {
        decoder_block_forward(decoder_out, decoder_out, encoder_out, t->decoder_layers[i], arena, training);
    }
    
    // Final output layer
    matmul(out, decoder_out, t->output_layer);
}

Value* transformer_forward_ad(Tensor* src, Tensor* tgt, Transformer* model, Arena* arena, int is_training) {
    // Note: Embeddings and positional encoding are not part of the graph for now.
    Tensor* src_embedded = create_tensor(src->n_dims, src->dims, TENSOR_TYPE_FLOAT, arena);
    token_embedding_forward(src_embedded, src, model->embedding);
    // TODO: Positional encoding should be part of the graph if it's learned
    add(src_embedded, src_embedded, model->pos_encoding);
    Value* encoder_in_val = create_value(src_embedded, NULL, 0, NULL, arena, NULL);

    Value* encoder_out_val = encoder_in_val;
    for (int i = 0; i < model->n_layers; i++) {
        encoder_out_val = encoder_block_forward_ad(encoder_out_val, model->encoder_layers[i], arena, is_training);
    }

    Tensor* tgt_embedded = create_tensor(tgt->n_dims, tgt->dims, TENSOR_TYPE_FLOAT, arena);
    token_embedding_forward(tgt_embedded, tgt, model->embedding);
    add(tgt_embedded, tgt_embedded, model->pos_encoding);
    Value* decoder_in_val = create_value(tgt_embedded, NULL, 0, NULL, arena, NULL);

    Value* decoder_out_val = decoder_in_val;
    for (int i = 0; i < model->n_layers; i++) {
        decoder_out_val = decoder_block_forward_ad(decoder_out_val, encoder_out_val, model->decoder_layers[i], arena, is_training);
    }

    Value* out_layer_val = create_value(model->output_layer, NULL, 0, NULL, arena, NULL);
    Value* logits = matmul_ad(decoder_out_val, out_layer_val, arena);

    return logits;
}

int save_transformer(Transformer* t, const char* path) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    
    // Simplified: only saving the output layer for now.
    // A full implementation would iterate over all parameters.
    fwrite(t->output_layer->data, sizeof(float), t->output_layer->dims[0] * t->output_layer->dims[1], fp);

    fclose(fp);
    return 1;
}

int load_transformer(Transformer* t, const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;

    // Simplified: only loading the output layer for now.
    fread(t->output_layer->data, sizeof(float), t->output_layer->dims[0] * t->output_layer->dims[1], fp);

    fclose(fp);
    return 1;
}
