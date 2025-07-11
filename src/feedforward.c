#include "feedforward.h"
#include <stdlib.h>
#include <assert.h>

static void relu(Tensor* t) {
    size_t total_elements = 1;
    for (int i = 0; i < t->n_dims; i++) {
        total_elements *= t->dims[i];
    }
    for (size_t i = 0; i < total_elements; i++) {
        t->data[i] = t->data[i] > 0 ? t->data[i] : 0;
    }
}

FeedForward* create_feedforward(int dim, int hidden_dim) {
    FeedForward* ff = (FeedForward*)malloc(sizeof(FeedForward));
    ff->dim = dim;
    ff->hidden_dim = hidden_dim;

    int w1_dims[] = {dim, hidden_dim};
    int b1_dims[] = {hidden_dim};
    int w2_dims[] = {hidden_dim, dim};
    int b2_dims[] = {dim};

    ff->W1 = create_tensor(2, w1_dims);
    ff->b1 = create_tensor(1, b1_dims);
    ff->W2 = create_tensor(2, w2_dims);
    ff->b2 = create_tensor(1, b2_dims);
    
    return ff;
}

void free_feedforward(FeedForward* ff) {
    if (ff) {
        free_tensor(ff->W1);
        free_tensor(ff->b1);
        free_tensor(ff->W2);
        free_tensor(ff->b2);
        free(ff);
    }
}

void feedforward_forward(Tensor* out, Tensor* in, FeedForward* ff) {
    assert(in->n_dims == 2);
    int batch_size = in->dims[0];
    int dim = in->dims[1];
    assert(dim == ff->dim);

    int hidden_dims[] = {batch_size, ff->hidden_dim};
    Tensor* hidden = create_tensor(2, hidden_dims);

    matmul(hidden, in, ff->W1);
    add(hidden, hidden, ff->b1);
    relu(hidden);
    matmul(out, hidden, ff->W2);
    add(out, out, ff->b2);

    free_tensor(hidden);
}
