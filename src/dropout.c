#include "dropout.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h> // Added for assertions

Dropout* create_dropout(float p) {
    Dropout* d = (Dropout*)malloc(sizeof(Dropout));
    d->p = p;
    return d;
}

void free_dropout(Dropout* d) {
    free(d);
}

void dropout_forward(Tensor* out, Tensor* in, Dropout* d, int training) {
    if (training) {
        assert(in->dtype == TENSOR_TYPE_FLOAT);
        assert(out->dtype == TENSOR_TYPE_FLOAT);
        float* in_data = (float*)in->data;
        float* out_data = (float*)out->data;
        int size = in->dims[0] * in->dims[1] * in->dims[2];
        float scale = 1.0f / (1.0f - d->p);
        for (int i = 0; i < size; i++) {
            if ((float)rand() / RAND_MAX > d->p) {
                out_data[i] = in_data[i] * scale;
            } else {
                out_data[i] = 0.0f;
            }
        }
    } else {
        // Just copy data during inference
        memcpy(out->data, in->data, in->dims[0] * in->dims[1] * in->dims[2] * sizeof(float));
    }
} 