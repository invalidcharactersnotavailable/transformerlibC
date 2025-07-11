#include <stdio.h>
#include <stdlib.h>
#include "transformer.h"
#include "tokenizer.h"
#include "optimizer.h"
#include "loss.h"
#include <string.h>
#include "memory.h"

#define ARENA_SIZE 256 * 1024 * 1024 // 256 MB

int main() {
    // Hyperparameters
    int n_layers = 1; // Simplified for demonstration
    int embed_dim = 128;
    int n_heads = 4;
    int ff_hidden_dim = 512;
    int batch_size = 1;
    int seq_len = 10;
    int vocab_size = 256; // ASCII
    int max_seq_len = 512;
    float learning_rate = 0.001f;
    int epochs = 10;

    // Create a transformer model
    Transformer* model = create_transformer(n_layers, vocab_size, max_seq_len, embed_dim, n_heads, ff_hidden_dim);
    Optimizer* optimizer = create_optimizer(model, learning_rate);

    // Create a memory arena for temporary allocations during training
    Arena* training_arena = create_arena(ARENA_SIZE);

    // Dummy data
    const char* sample_text = "hello world";
    int token_ids[seq_len];
    int num_tokens = encode(sample_text, token_ids, seq_len);
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* src_in = create_tensor(2, (int[]){batch_size, num_tokens}, TENSOR_TYPE_INT, training_arena);
        Tensor* tgt_in = create_tensor(2, (int[]){batch_size, num_tokens}, TENSOR_TYPE_INT, training_arena);
        // In a real scenario, src and tgt would be different (e.g., tgt is src shifted by one)
        memcpy(src_in->data, token_ids, num_tokens * sizeof(int));
        memcpy(tgt_in->data, token_ids, num_tokens * sizeof(int));

        // Forward pass
        Value* logits = transformer_forward_ad(src_in, tgt_in, model, training_arena, 1 /* training */);

        // Compute loss
        Value* loss = cross_entropy_loss_ad(logits, tgt_in, training_arena);
        printf("Epoch %d, Loss: %f\\n", epoch, ((float*)loss->data->data)[0]);

        // Backward pass
        backward(loss, training_arena);

        // Update weights
        optimizer_step(optimizer);
        
        // Zero gradients
        zero_grad(optimizer);

        // Reset the arena for the next iteration
        arena_reset(training_arena);
    }

    // Clean up
    destroy_arena(training_arena);
    free_optimizer(optimizer);
    free_transformer(model);

    printf("Training complete.\\n");
    return 0;
}
