#include "layernorm.h"
#include "tensor.h"
#include "autodiff.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

LayerNorm* create_layernorm(int embed_dim) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    if (!ln) { fprintf(stderr, "[ERR] malloc failed for LayerNorm\n"); return NULL; }
    ln->embed_dim = embed_dim;
    int dims[] = {embed_dim};
    ln->gamma = create_tensor(1, dims, TENSOR_TYPE_FLOAT);
    if (!ln->gamma) { fprintf(stderr, "[ERR] create_tensor failed for gamma in LayerNorm\n"); free(ln); return NULL; }
    ln->beta = create_tensor(1, dims, TENSOR_TYPE_FLOAT);
    if (!ln->beta) { fprintf(stderr, "[ERR] create_tensor failed for beta in LayerNorm\n"); free_tensor(ln->gamma); free(ln); return NULL; }
    for (int i = 0; i < embed_dim; i++) {
        ((float*)ln->gamma->data)[i] = 1.0f;
        ((float*)ln->beta->data)[i] = 0.0f;
    }
    return ln;
}

void free_layernorm(LayerNorm* ln) {
    if (ln) {
        free_tensor(ln->gamma);
        free_tensor(ln->beta);
        free(ln);
    }
}

void layernorm_forward(Tensor* out, Tensor* in, LayerNorm* ln) {
    assert(in->dtype == TENSOR_TYPE_FLOAT);
    assert(out->dtype == TENSOR_TYPE_FLOAT);
    assert(in->n_dims > 0 && out->n_dims == in->n_dims);
    assert(in->dims[in->n_dims - 1] == ln->embed_dim);
    size_t outer_size = 1;
    for (int i = 0; i < in->n_dims - 1; i++) {
        outer_size *= in->dims[i];
    }
    int feature_size = ln->embed_dim;
    float eps = 1e-5f;
    for (size_t i = 0; i < outer_size; i++) {
        float* in_row = (float*)in->data + i * feature_size;
        float* out_row = (float*)out->data + i * feature_size;
        float mean = 0.0f;
        for (int j = 0; j < feature_size; j++) {
            mean += in_row[j];
        }
        mean /= feature_size;
        float variance = 0.0f;
        for (int j = 0; j < feature_size; j++) {
            variance += (in_row[j] - mean) * (in_row[j] - mean);
        }
        variance /= feature_size;
        float inv_std = 1.0f / sqrtf(variance + eps);
        for (int j = 0; j < feature_size; j++) {
            out_row[j] = (in_row[j] - mean) * inv_std * ((float*)ln->gamma->data)[j] + ((float*)ln->beta->data)[j];
        }
    }
}

typedef struct {
    Tensor* x;
    Tensor* gamma;
    Tensor* mean;
    Tensor* inv_std;
} LayernormBackwardContext;


void backward_layernorm(Value* v) {
    LayernormBackwardContext* ctx = (LayernormBackwardContext*)v->op_context;
    Value* x_val = v->prev[0];
    Value* gamma_val = v->prev[1];
    Value* beta_val = v->prev[2];

    int feature_size = x_val->data->dims[x_val->data->n_dims - 1];
    size_t outer_size = 1;
    for(int i = 0; i < x_val->data->n_dims - 1; i++) {
        outer_size *= x_val->data->dims[i];
    }

    float* d_out = (float*)v->grad->data;
    float* d_x = (float*)x_val->grad->data;
    float* x = (float*)ctx->x->data;
    float* d_gamma = (float*)gamma_val->grad->data;
    float* d_beta = (float*)beta_val->grad->data;
    float* gamma = (float*)ctx->gamma->data;
    float* mean = (float*)ctx->mean->data;
    float* inv_std = (float*)ctx->inv_std->data;

    for(size_t i=0; i < outer_size; i++) {
        float* d_out_row = d_out + i * feature_size;
        float* x_row = x + i * feature_size;
        float* d_x_row = d_x + i * feature_size;
        float mean_row = mean[i];
        float inv_std_row = inv_std[i];
        
        float d_beta_sum = 0.f;
        float d_gamma_sum = 0.f;

        float d_norm_sum = 0.f;
        for(int j=0; j<feature_size; j++){
            d_beta_sum += d_out_row[j];
            d_gamma_sum += d_out_row[j] * (x_row[j] - mean_row) * inv_std_row;
        }
        d_beta[i] += d_beta_sum;
        d_gamma[i] += d_gamma_sum;
        
        for(int j=0; j<feature_size; j++){
            d_norm_sum += d_out_row[j] * gamma[j];
        }

        for(int j=0; j<feature_size; j++){
            float dx_norm = d_out_row[j] * gamma[j];
            float d_var = -0.5f * powf(inv_std_row, 3) * (x_row[j] - mean_row) * d_norm_sum;
            float d_mean = -inv_std_row * dx_norm + d_var * (-2.f/feature_size) * (x_row[j] - mean_row);
            
            d_x_row[j] += dx_norm * inv_std_row + d_var * (2.f/feature_size)*(x_row[j] - mean_row) + d_mean/feature_size;
        }
    }
}

Value* layernorm_forward_ad(Value* in, LayerNorm* ln) {
    Tensor* out_data = create_tensor(in->data->n_dims, in->data->dims, TENSOR_TYPE_FLOAT);
    
    size_t outer_size = 1;
    for(int i = 0; i < in->data->n_dims - 1; i++) {
        outer_size *= in->data->dims[i];
    }
    
    Tensor* mean = create_tensor(1, (int[]){outer_size}, TENSOR_TYPE_FLOAT);
    Tensor* inv_std = create_tensor(1, (int[]){outer_size}, TENSOR_TYPE_FLOAT);

    layernorm_forward(out_data, in->data, ln);

    LayernormBackwardContext* ctx = (LayernormBackwardContext*)malloc(sizeof(LayernormBackwardContext));
    ctx->x = in->data;
    ctx->gamma = ln->gamma;
    ctx->mean = mean;
    ctx->inv_std = inv_std;

    Value** prev = (Value**)malloc(3 * sizeof(Value*));
    prev[0] = in;
    prev[1] = create_value(ln->gamma, NULL, 0, NULL, NULL);
    prev[2] = create_value(ln->beta, NULL, 0, NULL, NULL);
    
    return create_value(out_data, prev, 3, ctx, backward_layernorm);
}

int save_layernorm(LayerNorm* ln, FILE* fp) {
    if (!save_tensor(ln->gamma, fp)) return 0;
    if (!save_tensor(ln->beta, fp)) return 0;
    return 1;
}

int load_layernorm(LayerNorm* ln, FILE* fp) {
    free_tensor(ln->gamma);
    free_tensor(ln->beta);

    ln->gamma = load_tensor(fp);
    ln->beta = load_tensor(fp);

    if (!ln->gamma || !ln->beta) {
        return 0;
    }
    return 1;
}
