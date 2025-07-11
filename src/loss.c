#include "loss.h"
#include "tensor.h"
#include "autodiff.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void cross_entropy_loss(float* loss, Tensor* logits, Tensor* targets) {
    // logits: (batch_size, seq_len, vocab_size) or (seq_len, vocab_size) if batch_size==1
    // targets: (batch_size, seq_len) or (seq_len) if batch_size==1
    assert(logits->dtype == TENSOR_TYPE_FLOAT);
    assert(targets->dtype == TENSOR_TYPE_INT);
    int n_dims = logits->n_dims;
    int batch_size, seq_len, vocab_size;
    if (n_dims == 3) {
        batch_size = logits->dims[0];
        seq_len = logits->dims[1];
        vocab_size = logits->dims[2];
        assert(targets->n_dims == 2);
        assert(targets->dims[0] == batch_size && targets->dims[1] == seq_len);
    } else if (n_dims == 2) {
        batch_size = 1;
        seq_len = logits->dims[0];
        vocab_size = logits->dims[1];
        assert(targets->n_dims == 1);
        assert(targets->dims[0] == seq_len);
    } else {
        assert(0 && "logits must be 2D or 3D");
    }
    float* logits_data = (float*)logits->data;
    int* targets_data = (int*)targets->data;
    float loss_val = 0.0f;
    int total = batch_size * seq_len;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_len; i++) {
            int logit_offset = (batch_size == 1) ? (i * vocab_size) : (b * seq_len * vocab_size + i * vocab_size);
            int target_offset = (batch_size == 1) ? i : (b * seq_len + i);
            float max_logit = logits_data[logit_offset];
            for (int j = 1; j < vocab_size; j++) {
                if (logits_data[logit_offset + j] > max_logit) {
                    max_logit = logits_data[logit_offset + j];
                }
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < vocab_size; j++) {
                sum_exp += expf(logits_data[logit_offset + j] - max_logit);
            }
            int target_idx = targets_data[target_offset];
            assert(target_idx >= 0 && target_idx < vocab_size);
            float log_prob = logits_data[logit_offset + target_idx] - max_logit - logf(sum_exp);
            loss_val += -log_prob;
        }
    }
    *loss = loss_val / total;
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