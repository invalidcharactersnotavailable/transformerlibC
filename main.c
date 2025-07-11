#include <stdio.h>
#include <stdlib.h>
#include "transformer.h"

int main() {
    // Hyperparameters
    int n_layers = 6;
    int embed_dim = 512;
    int n_heads = 8;
    int ff_hidden_dim = 2048;
    int batch_size = 1;
    int seq_len = 10;

    // Create a transformer model
    Transformer* model = create_transformer(n_layers, embed_dim, n_heads, ff_hidden_dim);
    if (!model) {
        fprintf(stderr, "Failed to create transformer model.\\n");
        return 1;
    }

    // Create input and output tensors
    int in_dims[] = {batch_size, seq_len, embed_dim};
    Tensor* in = create_tensor(3, in_dims);
    Tensor* out = create_tensor(3, in_dims);

    // Initialize input tensor with some data (e.g., random)
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        in->data[i] = (float)rand() / RAND_MAX;
    }

    // Perform a forward pass
    printf("Performing forward pass...\\n");
    transformer_forward(out, in, model);
    printf("Forward pass completed.\\n");

    // Clean up
    free_tensor(in);
    free_tensor(out);
    free_transformer(model);

    printf("Cleaned up resources.\\n");
    return 0;
}
