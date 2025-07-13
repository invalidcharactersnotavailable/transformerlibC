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
    if (!t) { fprintf(stderr, "[ERR] malloc failed for Transformer\n"); return NULL; }

    t->n_layers = n_layers;
    t->vocab_size = vocab_size;
    t->max_seq_len = max_seq_len;
    t->embed_dim = embed_dim;

    t->embedding = create_token_embedding(vocab_size, embed_dim);
    if (!t->embedding) { fprintf(stderr, "[ERR] create_token_embedding failed\n"); free(t); return NULL; }

    int pos_dims[] = {max_seq_len, embed_dim};
    t->pos_encoding = create_tensor(2, pos_dims, TENSOR_TYPE_FLOAT);
    if (!t->pos_encoding) { fprintf(stderr, "[ERR] create_tensor failed for pos_encoding\n"); free_token_embedding(t->embedding); free(t); return NULL; }
    create_positional_encoding(t->pos_encoding, max_seq_len, embed_dim);

    t->encoder_layers = (EncoderBlock**)malloc(n_layers * sizeof(EncoderBlock*));
    if (!t->encoder_layers) { fprintf(stderr, "[ERR] malloc failed for encoder_layers\n"); free_tensor(t->pos_encoding); free_token_embedding(t->embedding); free(t); return NULL; }
    t->decoder_layers = (DecoderBlock**)malloc(n_layers * sizeof(DecoderBlock*));
    if (!t->decoder_layers) { fprintf(stderr, "[ERR] malloc failed for decoder_layers\n"); free(t->encoder_layers); free_tensor(t->pos_encoding); free_token_embedding(t->embedding); free(t); return NULL; }
    
    for (int i = 0; i < n_layers; i++) {
        t->encoder_layers[i] = create_encoder_block(embed_dim, n_heads, ff_hidden_dim);
        if (!t->encoder_layers[i]) { fprintf(stderr, "[ERR] create_encoder_block failed for layer %d\n", i); goto cleanup; }
        t->decoder_layers[i] = create_decoder_block(embed_dim, n_heads, ff_hidden_dim);
        if (!t->decoder_layers[i]) { fprintf(stderr, "[ERR] create_decoder_block failed for layer %d\n", i); goto cleanup; }
    }
    
    int out_dims[] = {embed_dim, vocab_size};
    t->output_layer = create_tensor(2, out_dims, TENSOR_TYPE_FLOAT);
    if (!t->output_layer) { fprintf(stderr, "[ERR] create_tensor failed for output_layer\n"); goto cleanup; }

    int mask_dims[] = {max_seq_len, max_seq_len};
    t->look_ahead_mask = create_tensor(2, mask_dims, TENSOR_TYPE_FLOAT);
    if (!t->look_ahead_mask) { fprintf(stderr, "[ERR] create_tensor failed for look_ahead_mask\n"); free_tensor(t->output_layer); goto cleanup; }
    create_look_ahead_mask(t->look_ahead_mask, max_seq_len);

    return t;

cleanup:
    for (int j = 0; j < i; j++) {
        if (t->encoder_layers[j]) free_encoder_block(t->encoder_layers[j]);
        if (t->decoder_layers[j]) free_decoder_block(t->decoder_layers[j]);
    }
    free(t->encoder_layers);
    free(t->decoder_layers);
    free_tensor(t->pos_encoding);
    free_token_embedding(t->embedding);
    free(t);
    return NULL;
}

void create_look_ahead_mask(Tensor* mask, int seq_len) {
    float* mask_data = (float*)mask->data;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                mask_data[i * seq_len + j] = -1e9; // Use a large negative value for masking
            } else {
                mask_data[i * seq_len + j] = 0.0f;
            }
        }
    }
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
    free_tensor(t->look_ahead_mask);
    free(t);
}

void transformer_forward(Tensor* out, Tensor* src_in, Tensor* tgt_in, Transformer* t, int training) {
    // embeddings
    Tensor* src_embed = create_tensor(3, (int[]){src_in->dims[0], src_in->dims[1], t->embed_dim}, TENSOR_TYPE_FLOAT);
    token_embedding_forward(src_embed, src_in, t->embedding);
    Tensor* tgt_embed = create_tensor(3, (int[]){tgt_in->dims[0], tgt_in->dims[1], t->embed_dim}, TENSOR_TYPE_FLOAT);
    token_embedding_forward(tgt_embed, tgt_in, t->embedding);

    // encoder
    Tensor* encoder_in = src_embed;
    Tensor* encoder_out = NULL;
    for (int i = 0; i < t->n_layers; i++) {
        encoder_out = create_tensor(encoder_in->n_dims, encoder_in->dims, TENSOR_TYPE_FLOAT);
        encoder_block_forward(encoder_out, encoder_in, t->encoder_layers[i], training);
        if (i > 0) free_tensor(encoder_in);
        encoder_in = encoder_out;
    }

    // decoder
    Tensor* decoder_in = tgt_embed;
    Tensor* decoder_out = NULL;
    for (int i = 0; i < t->n_layers; i++) {
        decoder_out = create_tensor(decoder_in->n_dims, decoder_in->dims, TENSOR_TYPE_FLOAT);
        decoder_block_forward(decoder_out, decoder_in, encoder_out, t->decoder_layers[i], training);
        if (i > 0) free_tensor(decoder_in);
        decoder_in = decoder_out;
    }

    // output projection (e.g., linear layer)
    matmul(out, decoder_out, t->output_layer);

    // free intermediates
    free_tensor(src_embed);
    free_tensor(tgt_embed);
    free_tensor(encoder_out);
    free_tensor(decoder_out);
}

Value* transformer_forward_ad(Tensor* src, Tensor* tgt, Transformer* model, int is_training) {
    // Note: Embeddings and positional encoding are not part of the graph for now.
    int* src_embedded_dims = (int[]){src->dims[0], src->dims[1], model->embed_dim};
    Tensor* src_embedded = create_tensor(3, src_embedded_dims, TENSOR_TYPE_FLOAT);
    token_embedding_forward(src_embedded, src, model->embedding);

    Tensor* pos_encoding_broadcast = create_tensor(3, src_embedded_dims, TENSOR_TYPE_FLOAT);
    size_t src_numel = tensor_numel(src_embedded);
    size_t pos_numel = tensor_numel(model->pos_encoding);
    for (size_t i = 0; i < src_numel / pos_numel; i++) {
        memcpy((char*)pos_encoding_broadcast->data + i * pos_numel * sizeof(float), model->pos_encoding->data, pos_numel * sizeof(float));
    }
    add(src_embedded, src_embedded, pos_encoding_broadcast);
    Value* encoder_in_val = create_value(src_embedded, NULL, 0, NULL, NULL);

    Value* encoder_out_val = encoder_in_val;
    for (int i = 0; i < model->n_layers; i++) {
        encoder_out_val = encoder_block_forward_ad(encoder_out_val, model->encoder_layers[i], is_training);
    }

    int* tgt_embedded_dims = (int[]){tgt->dims[0], tgt->dims[1], model->embed_dim};
    Tensor* tgt_embedded = create_tensor(3, tgt_embedded_dims, TENSOR_TYPE_FLOAT);
    token_embedding_forward(tgt_embedded, tgt, model->embedding);

    Tensor* pos_encoding_broadcast_tgt = create_tensor(3, tgt_embedded_dims, TENSOR_TYPE_FLOAT);
    size_t tgt_numel = tensor_numel(tgt_embedded);
    size_t pos_numel_tgt = tensor_numel(model->pos_encoding);
    for (size_t i = 0; i < tgt_numel / pos_numel_tgt; i++) {
        memcpy((char*)pos_encoding_broadcast_tgt->data + i * pos_numel_tgt * sizeof(float), model->pos_encoding->data, pos_numel_tgt * sizeof(float));
    }
    add(tgt_embedded, tgt_embedded, pos_encoding_broadcast_tgt);
    Value* decoder_in_val = create_value(tgt_embedded, NULL, 0, NULL, NULL);

    Value* decoder_out_val = decoder_in_val;
    for (int i = 0; i < model->n_layers; i++) {
        decoder_out_val = decoder_block_forward_ad(decoder_out_val, encoder_out_val, model->decoder_layers[i], is_training, model->look_ahead_mask);
    }

    Value* out_layer_val = create_value(model->output_layer, NULL, 0, NULL, NULL);
    Value* logits = matmul_ad(decoder_out_val, out_layer_val);

    return logits;
}

int save_transformer(Transformer* t, const char* path) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;

    int success = 1;
    if (!save_token_embedding(t->embedding, fp)) success = 0;
    
    for (int i = 0; i < t->n_layers && success; i++) {
        if (!save_encoder_block(t->encoder_layers[i], fp)) success = 0;
    }
    for (int i = 0; i < t->n_layers && success; i++) {
        if (!save_decoder_block(t->decoder_layers[i], fp)) success = 0;
    }

    if (success && !save_tensor(t->output_layer, fp)) success = 0;

    fclose(fp);
    return success;
}

long get_transformer_param_count(Transformer* t) {
    long count = 0;
    count += tensor_numel(t->embedding->weights);
    for (int i = 0; i < t->n_layers; i++) {
        // Encoder
        count += tensor_numel(t->encoder_layers[i]->attention->w_q);
        count += tensor_numel(t->encoder_layers[i]->attention->w_k);
        count += tensor_numel(t->encoder_layers[i]->attention->w_v);
        count += tensor_numel(t->encoder_layers[i]->attention->w_o);
        count += tensor_numel(t->encoder_layers[i]->ff->w1);
        count += tensor_numel(t->encoder_layers[i]->ff->w2);
        count += tensor_numel(t->encoder_layers[i]->ln1->gamma);
        count += tensor_numel(t->encoder_layers[i]->ln1->beta);
        count += tensor_numel(t->encoder_layers[i]->ln2->gamma);
        count += tensor_numel(t->encoder_layers[i]->ln2->beta);
        // Decoder
        count += tensor_numel(t->decoder_layers[i]->masked_attention->w_q);
        count += tensor_numel(t->decoder_layers[i]->masked_attention->w_k);
        count += tensor_numel(t->decoder_layers[i]->masked_attention->w_v);
        count += tensor_numel(t->decoder_layers[i]->masked_attention->w_o);
        count += tensor_numel(t->decoder_layers[i]->cross_attention->w_q);
        count += tensor_numel(t->decoder_layers[i]->cross_attention->w_k);
        count += tensor_numel(t->decoder_layers[i]->cross_attention->w_v);
        count += tensor_numel(t->decoder_layers[i]->cross_attention->w_o);
        count += tensor_numel(t->decoder_layers[i]->ff->w1);
        count += tensor_numel(t->decoder_layers[i]->ff->w2);
        count += tensor_numel(t->decoder_layers[i]->ln1->gamma);
        count += tensor_numel(t->decoder_layers[i]->ln1->beta);
        count += tensor_numel(t->decoder_layers[i]->ln2->gamma);
        count += tensor_numel(t->decoder_layers[i]->ln2->beta);
        count += tensor_numel(t->decoder_layers[i]->ln3->gamma);
        count += tensor_numel(t->decoder_layers[i]->ln3->beta);
    }
    count += tensor_numel(t->output_layer);
    return count;
}

int load_transformer(Transformer* t, const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;

    int success = 1;
    if (!load_token_embedding(t->embedding, fp)) success = 0;

    for (int i = 0; i < t->n_layers && success; i++) {
        if (!load_encoder_block(t->encoder_layers[i], fp)) success = 0;
    }
    for (int i = 0; i < t->n_layers && success; i++) {
        if (!load_decoder_block(t->decoder_layers[i], fp)) success = 0;
    }

    if (success) {
        free_tensor(t->output_layer);
        t->output_layer = load_tensor(fp);
        if (!t->output_layer) success = 0;
    }

    fclose(fp);
    return success;
}
