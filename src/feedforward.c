#include "feedforward.h"
#include "autodiff.h"
#include <stdlib.h>
#include <assert.h>

static void relu_ad_inplace(Value* v) {
    // This is just a placeholder for a potential future op
}

FeedForward* create_feedforward(int embed_dim, int hidden_dim) {
    FeedForward* ff = (FeedForward*)malloc(sizeof(FeedForward));
    int w1_dims[] = {embed_dim, hidden_dim};
    int b1_dims[] = {hidden_dim};
    int w2_dims[] = {hidden_dim, embed_dim};
    int b2_dims[] = {embed_dim};
    ff->w1 = create_tensor(2, w1_dims, TENSOR_TYPE_FLOAT);
    ff->b1 = create_tensor(1, b1_dims, TENSOR_TYPE_FLOAT);
    ff->w2 = create_tensor(2, w2_dims, TENSOR_TYPE_FLOAT);
    ff->b2 = create_tensor(1, b2_dims, TENSOR_TYPE_FLOAT);
    return ff;
}

void free_feedforward(FeedForward* ff) {
    if (ff) {
        free_tensor(ff->w1);
        free_tensor(ff->b1);
        free_tensor(ff->w2);
        free_tensor(ff->b2);
        free(ff);
    }
}

void feedforward_forward(Tensor* out, Tensor* in, FeedForward* ff) {
    Tensor* hidden = create_tensor(NULL, 3, (int[]){in->dims[0], in->dims[1], ff->hidden_dim}, TENSOR_TYPE_FLOAT);
    matmul(hidden, in, ff->w1);
    // relu is in-place
    float* hidden_data = (float*)hidden->data;
    for (int i = 0; i < in->dims[0] * in->dims[1] * ff->hidden_dim; i++) {
        if (hidden_data[i] < 0) hidden_data[i] = 0;
    }
    matmul(out, hidden, ff->w2);
    free_tensor(hidden);
}

void backward_feedforward(Value* v) {
    // Placeholder for feedforward backward pass
}

Value* feedforward_forward_ad(Arena* arena, Value* in, FeedForward* ff) {
    Value* w1_val = create_value(arena, ff->w1, NULL, 0, NULL, NULL);
    Value* w2_val = create_value(arena, ff->w2, NULL, 0, NULL, NULL);
    Value* b1_val = create_value(arena, ff->b1, NULL, 0, NULL, NULL);
    Value* b2_val = create_value(arena, ff->b2, NULL, 0, NULL, NULL);

    Value* hidden = matmul_ad(arena, in, w1_val);
    hidden = add_ad(arena, hidden, b1_val);
    hidden = relu_ad(arena, hidden);
    Value* out = matmul_ad(arena, hidden, w2_val);
    out = add_ad(arena, out, b2_val);
    
    return out;
}

int save_feedforward(FeedForward* ff, FILE* fp) {
    if (!save_tensor(ff->w1, fp)) return 0;
    if (!save_tensor(ff->b1, fp)) return 0;
    if (!save_tensor(ff->w2, fp)) return 0;
    if (!save_tensor(ff->b2, fp)) return 0;
    return 1;
}

int load_feedforward(FeedForward* ff, FILE* fp) {
    free_tensor(ff->w1);
    free_tensor(ff->b1);
    free_tensor(ff->w2);
    free_tensor(ff->b2);

    ff->w1 = load_tensor(fp, NULL);
    ff->b1 = load_tensor(fp, NULL);
    ff->w2 = load_tensor(fp, NULL);
    ff->b2 = load_tensor(fp, NULL);

    if (!ff->w1 || !ff->b1 || !ff->w2 || !ff->b2) {
        return 0;
    }
    return 1;
}
