#include "loss.h"
#include "tensor.h"
#include "autodiff.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void cross_entropy_loss(float* loss, Tensor* logits, Tensor* targets) {
    // This is a simplified implementation assuming batch_size = 1 and 2D logits (seq_len, vocab_size)
    assert(logits->dtype == TENSOR_TYPE_FLOAT);
    assert(targets->dtype == TENSOR_TYPE_INT);
    float* logits_data = (float*)logits->data;
    int* targets_data = (int*)targets->data;

    int seq_len = logits->dims[0];
    int vocab_size = logits->dims[1];
    float loss_val = 0.0f;

    for (int i = 0; i < seq_len; i++) {
        float max_logit = logits_data[i * vocab_size];
        for (int j = 1; j < vocab_size; j++) {
            if (logits_data[i * vocab_size + j] > max_logit) {
                max_logit = logits_data[i * vocab_size + j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            sum_exp += expf(logits_data[i * vocab_size + j] - max_logit);
        }
        
        int target_idx = targets_data[i];
        loss_val += -logf(expf(logits_data[i * vocab_size + target_idx] - max_logit) / sum_exp);
    }
    *loss = loss_val / seq_len;
}

void backward_cross_entropy(Value* v) {
    CrossEntropyContext* ctx = (CrossEntropyContext*)v->op_context;
    Value* y_val = v->prev[0]; // These are the probabilities after softmax
    Tensor* y = y_val->data;
    Tensor* t = ctx->targets;
    Tensor* d_loss = v->grad;

    float* y_data = (float*)y->data;
    int* t_data = (int*)t->data;
    float* d_y = (float*)y_val->grad->data;
    float d_loss_val = ((float*)d_loss->data)[0];

    int batch_size = y->dims[0];
    int seq_len = y->dims[1];
    int vocab_size = y->dims[2];

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int target_idx = t_data[b * seq_len + s];
            for (int v_idx = 0; v_idx < vocab_size; v_idx++) {
                float y_hat = y_data[b * seq_len * vocab_size + s * vocab_size + v_idx];
                if (v_idx == target_idx) {
                    d_y[b * seq_len * vocab_size + s * vocab_size + v_idx] += d_loss_val * (-1.0f / y_hat);
                }
            }
        }
    }
}

Value* cross_entropy_loss_ad(Arena* arena, Value* logits, Tensor* targets) {
    Value* probs = softmax_ad(arena, logits);

    int batch_size = probs->data->dims[0];
    int seq_len = probs->data->dims[1];
    int vocab_size = probs->data->dims[2];

    float loss_val = 0.0f;
    float* probs_data = (float*)probs->data->data;
    int* targets_data = (int*)targets->data;

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int target_idx = targets_data[b * seq_len + s];
            float prob_of_correct_class = probs_data[b * vocab_size * seq_len + s * vocab_size + target_idx];
            loss_val += -logf(prob_of_correct_class);
        }
    }
    loss_val /= (batch_size * seq_len);

    Tensor* loss_tensor = create_tensor(arena, 1, (int[]){1}, TENSOR_TYPE_FLOAT);
    ((float*)loss_tensor->data)[0] = loss_val;
    
    Value** prev = (Value**)arena_alloc(arena, 1 * sizeof(Value*));
    prev[0] = probs;
    
    CrossEntropyContext* ctx = (CrossEntropyContext*)arena_alloc(arena, sizeof(CrossEntropyContext));
    ctx->targets = targets;

    return create_value(arena, loss_tensor, prev, 1, ctx, backward_cross_entropy);
} 