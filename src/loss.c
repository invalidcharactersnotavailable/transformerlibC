#include "loss.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h> // Added for assertions

float cross_entropy_loss(Tensor* logits, Tensor* targets) {
    // This is a simplified implementation assuming batch_size = 1 and 2D logits (seq_len, vocab_size)
    assert(logits->dtype == TENSOR_TYPE_FLOAT);
    assert(targets->dtype == TENSOR_TYPE_INT);
    float* logits_data = (float*)logits->data;
    int* targets_data = (int*)targets->data;

    int seq_len = logits->dims[0];
    int vocab_size = logits->dims[1];
    float loss = 0.0f;

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
        loss += -logf(expf(logits_data[i * vocab_size + target_idx] - max_logit) / sum_exp);
    }
    return loss / seq_len;
}

void backward_cross_entropy(Value* v) {
    // This backward pass is complex. For now, we will leave it as a placeholder.
    // It would calculate the gradient of the loss with respect to the logits.
    // The gradient is essentially (softmax_probabilities - one_hot_targets).
}

Value* cross_entropy_loss_ad(Value* logits, Tensor* targets, Arena* arena) {
    // The forward pass is the same as the regular function.
    float loss_val = cross_entropy_loss(logits->data, targets);

    // Create a scalar tensor for the loss value.
    Tensor* loss_tensor = create_tensor(1, (int[]){1}, TENSOR_TYPE_FLOAT, arena);
    ((float*)loss_tensor->data)[0] = loss_val;
    
    Value** prev = (Value**)arena_alloc(arena, sizeof(Value*));
    prev[0] = logits;

    return create_value(loss_tensor, prev, 1, NULL, arena, backward_cross_entropy);
} 