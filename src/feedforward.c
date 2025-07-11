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

    ff->w1 = create_tensor(2, w1_dims, TENSOR_TYPE_FLOAT, NULL);
    ff->b1 = create_tensor(1, b1_dims, TENSOR_TYPE_FLOAT, NULL);
    ff->w2 = create_tensor(2, w2_dims, TENSOR_TYPE_FLOAT, NULL);
    ff->b2 = create_tensor(1, b2_dims, TENSOR_TYPE_FLOAT, NULL);

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

void feedforward_forward(Tensor* out, Tensor* in, FeedForward* ff, Arena* arena) {
    int batch_size = in->dims[0];
    int seq_len = in->dims[1];
    
    int hidden_dims[] = {batch_size, seq_len, ff->w1->dims[1]};
    Tensor* hidden = create_tensor(3, hidden_dims, TENSOR_TYPE_FLOAT, arena);

    matmul(hidden, in, ff->w1);
    add(hidden, hidden, ff->b1); // Broadcasting add
    
    // Non-in-place relu
    float* hidden_data = (float*)hidden->data;
    size_t size = batch_size * seq_len * ff->w1->dims[1];
    for(size_t i=0; i<size; i++) {
        hidden_data[i] = hidden_data[i] > 0 ? hidden_data[i] : 0;
    }
    
    // 2. Second linear layer + bias
    matmul(out, hidden, ff->w2);
    add(out, out, ff->b2);
}

void backward_feedforward(Value* v) {
    // Placeholder for feedforward backward pass
}

Value* feedforward_forward_ad(Value* in, FeedForward* ff, Arena* arena) {
    Value* w1_val = create_value(ff->w1, NULL, 0, NULL, arena, NULL); 
    Value* b1_val = create_value(ff->b1, NULL, 0, NULL, arena, NULL); 
    Value* w2_val = create_value(ff->w2, NULL, 0, NULL, arena, NULL); 
    Value* b2_val = create_value(ff->b2, NULL, 0, NULL, arena, NULL);

    Value* hidden = add_ad(matmul_ad(in, w1_val, arena), b1_val, arena);
    Value* relu_out = relu_ad(hidden, arena);
    Value* out = add_ad(matmul_ad(relu_out, w2_val, arena), b2_val, arena);
    
    return out;
}
