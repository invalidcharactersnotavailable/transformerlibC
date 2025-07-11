#include "autodiff.h"
#include "tensor.h"
#include "memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

Value* create_value(Arena* arena, Tensor* data, Value** prev, int n_prev, void* op_context, void (*_backward)(struct Value*)) {
    Value* v;
    if (arena) {
        v = (Value*)arena_alloc(arena, sizeof(Value));
        if (!v) { fprintf(stderr, "[ERR] arena_alloc failed for Value\n"); return NULL; }
        v->prev = n_prev > 0 ? (Value**)arena_alloc(arena, n_prev * sizeof(Value*)) : NULL;
        if (n_prev > 0 && !v->prev) { fprintf(stderr, "[ERR] arena_alloc failed for Value prev\n"); return NULL; }
    } else {
        v = (Value*)malloc(sizeof(Value));
        if (!v) { fprintf(stderr, "[ERR] malloc failed for Value\n"); return NULL; }
        v->prev = n_prev > 0 ? (Value**)malloc(n_prev * sizeof(Value*)) : NULL;
        if (n_prev > 0 && !v->prev) { fprintf(stderr, "[ERR] malloc failed for Value prev\n"); free(v); return NULL; }
    }
    
    v->data = data;
    v->grad = NULL;
    v->n_prev = n_prev;
    if (prev && v->prev) {
        memcpy(v->prev, prev, n_prev * sizeof(Value*));
    }
    v->_backward = _backward;
    v->op_context = op_context;
    v->arena = arena;
    return v;
}

void free_value(Value* v) {
    if (v) {
        free_tensor(v->data);
        free_tensor(v->grad);
        if (v->op_context) free(v->op_context);
        free(v);
    }
}

void backward_add(Value* v) {
    assert(v->n_prev == 2);
    Value* a = v->prev[0];
    Value* b = v->prev[1];
    // Gradient of add is 1, so we just add the output grad to both inputs.
    // Note: This needs to handle broadcasting correctly for a robust solution,
    // but for now, we assume gradients have the correct shapes.
    add(a->grad, a->grad, v->grad);
    add(b->grad, b->grad, v->grad);
}

Value* add_ad(Arena* arena, Value* a, Value* b) {
    // Determine output shape, considering broadcasting
    assert(a->data->n_dims >= b->data->n_dims);
    int* out_dims = (int*)arena_alloc(arena, a->data->n_dims * sizeof(int));
    memcpy(out_dims, a->data->dims, a->data->n_dims * sizeof(int));
    Tensor* out_data = create_tensor(arena, a->data->n_dims, out_dims, TENSOR_TYPE_FLOAT);
    free(out_dims);

    add(out_data, a->data, b->data);
    Value** prev = (Value**)arena_alloc(arena, 2 * sizeof(Value*));
    prev[0] = a;
    prev[1] = b;
    return create_value(arena, out_data, prev, 2, NULL, backward_add);
}

void backward_matmul(Value* v) {
    assert(v->n_prev == 2);
    Value* a = v->prev[0];
    Value* b = v->prev[1];
    Tensor* dC = v->grad; // Gradient of the output

    // Create temporary tensors for the gradients to be computed
    Tensor* dA = create_tensor(a->data->n_dims, a->data->dims, TENSOR_TYPE_FLOAT);
    Tensor* dB = create_tensor(b->data->n_dims, b->data->dims, TENSOR_TYPE_FLOAT);

    // Case 1: Simple 2D x 2D -> 2D
    if (a->data->n_dims == 2 && b->data->n_dims == 2) {
        // dA = dC @ B.T
        Tensor* b_t = create_tensor(2, (int[]){b->data->dims[1], b->data->dims[0]}, TENSOR_TYPE_FLOAT);
        transpose(b_t, b->data, 0, 1);
        matmul(dA, dC, b_t);
        free_tensor(b_t);

        // dB = A.T @ dC
        Tensor* a_t = create_tensor(2, (int[]){a->data->dims[1], a->data->dims[0]}, TENSOR_TYPE_FLOAT);
        transpose(a_t, a->data, 0, 1);
        matmul(dB, a_t, dC);
        free_tensor(a_t);
    }
    // Case 2: Batched 3D x 2D -> 3D (e.g. (B, S, E) @ (E, F) -> (B, S, F))
    else if (a->data->n_dims == 3 && b->data->n_dims == 2) {
        // dA = dC @ B.T = (B, S, F) @ (F, E) -> (B, S, E)
        Tensor* b_t = create_tensor(2, (int[]){b->data->dims[1], b->data->dims[0]}, TENSOR_TYPE_FLOAT);
        transpose(b_t, b->data, 0, 1);
        matmul(dA, dC, b_t);
        free_tensor(b_t);

        // dB = A.T @ dC. This requires reshaping and summation over the batch dimension.
        // Reshape A from (B, S, E) to (B*S, E)
        // Reshape dC from (B, S, F) to (B*S, F)
        // A.T becomes (E, B*S)
        // Then (E, B*S) @ (B*S, F) -> (E, F)
        int B = a->data->dims[0];
        int S = a->data->dims[1];
        int E = a->data->dims[2];
        int F = b->data->dims[1];
        
        Tensor* a_reshaped = create_tensor(2, (int[]){B * S, E}, TENSOR_TYPE_FLOAT);
        memcpy(a_reshaped->data, a->data->data, (size_t)B * S * E * sizeof(float));
        
        Tensor* dC_reshaped = create_tensor(2, (int[]){B * S, F}, TENSOR_TYPE_FLOAT);
        memcpy(dC_reshaped->data, dC->data, (size_t)B * S * F * sizeof(float));

        Tensor* a_reshaped_t = create_tensor(2, (int[]){E, B * S}, TENSOR_TYPE_FLOAT);
        transpose(a_reshaped_t, a_reshaped, 0, 1);
        
        matmul(dB, a_reshaped_t, dC_reshaped); // dB is (E, F)

        free_tensor(a_reshaped);
        free_tensor(dC_reshaped);
        free_tensor(a_reshaped_t);
    }
    // Case 3: Batched 4D x 4D -> 4D (e.g. (B, H, S, D) @ (B, H, D, S')) -> (B, H, S, S'))
    else if (a->data->n_dims == 4 && b->data->n_dims == 4) {
        // dA = dC @ B.T -> (B,H,S,S') @ (B,H,S',D) -> (B,H,S,D)
        Tensor* b_t = create_tensor(4, (int[]){b->data->dims[0], b->data->dims[1], b->data->dims[3], b->data->dims[2]}, TENSOR_TYPE_FLOAT);
        transpose(b_t, b->data, 2, 3);
        matmul(dA, dC, b_t);
        free_tensor(b_t);

        // dB = A.T @ dC -> (B,H,D,S) @ (B,H,S,S') -> (B,H,D,S')
        Tensor* a_t = create_tensor(4, (int[]){a->data->dims[0], a->data->dims[1], a->data->dims[3], a->data->dims[2]}, TENSOR_TYPE_FLOAT);
        transpose(a_t, a->data, 2, 3);
        matmul(dB, a_t, dC);
        free_tensor(a_t);
    }
     // Case 4: Batched 3D x 3D -> 3D (e.g. (B, S, E) @ (B, E, T) -> (B, S, T))
    else if (a->data->n_dims == 3 && b->data->n_dims == 3) {
        // dA = dC @ B.T = (B, S, T) @ (B, T, E) -> (B, S, E)
        Tensor* b_t = create_tensor(3, (int[]){b->data->dims[0], b->data->dims[2], b->data->dims[1]}, TENSOR_TYPE_FLOAT);
        transpose(b_t, b->data, 1, 2);
        matmul(dA, dC, b_t);
        free_tensor(b_t);

        // dB = A.T @ dC = (B, E, S) @ (B, S, T) -> (B, E, T)
        Tensor* a_t = create_tensor(3, (int[]){a->data->dims[0], a->data->dims[2], a->data->dims[1]}, TENSOR_TYPE_FLOAT);
        transpose(a_t, a->data, 1, 2);
        matmul(dB, a_t, dC);
        free_tensor(a_t);
    }
    else {
        fprintf(stderr, "Unsupported matmul dimensions for backward pass: a_dims=%d, b_dims=%d\n", a->data->n_dims, b->data->n_dims);
        assert(0);
    }
    
    // Accumulate gradients
    add(a->grad, a->grad, dA);
    add(b->grad, b->grad, dB);

    // Free the temporary tensors
    free_tensor(dA);
    free_tensor(dB);
}

Value* matmul_ad(Arena* arena, Value* a, Value* b) {
    // This function needs to correctly determine the output shape based on the input shapes
    int* out_dims;
    int out_n_dims;

    if (a->data->n_dims == 2 && b->data->n_dims == 2) {
        out_n_dims = 2;
        out_dims = (int*)arena_alloc(arena, out_n_dims * sizeof(int));
        out_dims[0] = a->data->dims[0];
        out_dims[1] = b->data->dims[1];
    } else if (a->data->n_dims == 3 && b->data->n_dims == 2) {
        out_n_dims = 3;
        out_dims = (int*)arena_alloc(arena, out_n_dims * sizeof(int));
        out_dims[0] = a->data->dims[0];
        out_dims[1] = a->data->dims[1];
        out_dims[2] = b->data->dims[1];
    } else if (a->data->n_dims == 4 && b->data->n_dims == 4) {
        out_n_dims = 4;
        out_dims = (int*)arena_alloc(arena, out_n_dims * sizeof(int));
        out_dims[0] = a->data->dims[0];
        out_dims[1] = a->data->dims[1];
        out_dims[2] = a->data->dims[2];
        out_dims[3] = b->data->dims[3];
    } else if (a->data->n_dims == 3 && b->data->n_dims == 3) {
        out_n_dims = 3;
        out_dims = (int*)arena_alloc(arena, out_n_dims * sizeof(int));
        out_dims[0] = a->data->dims[0];
        out_dims[1] = a->data->dims[1];
        out_dims[2] = b->data->dims[2];
    } else {
        fprintf(stderr, "Unsupported matmul dimensions for forward pass ad: a_dims=%d, b_dims=%d\n", a->data->n_dims, b->data->n_dims);
        assert(0);
    }

    Tensor* out_data = create_tensor(arena, out_n_dims, out_dims, TENSOR_TYPE_FLOAT);
    free(out_dims);
    matmul(out_data, a->data, b->data);
    Value** prev = (Value**)arena_alloc(arena, 2 * sizeof(Value*));
    prev[0] = a;
    prev[1] = b;
    return create_value(arena, out_data, prev, 2, NULL, backward_matmul);
}

void backward_softmax(Value* v) {
    assert(v->n_prev == 1);
    Value* a = v->prev[0];

    float* y = (float*)v->data->data;
    float* dL_dy = (float*)v->grad->data;
    
    int last_dim = v->data->dims[v->data->n_dims - 1];
    size_t outer_size = 1;
    for (int i = 0; i < v->data->n_dims - 1; i++) {
        outer_size *= v->data->dims[i];
    }
    
    // In-place add to gradient of a
    float* a_grad_data = (float*)a->grad->data;

    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        float* y_row = y + i * last_dim;
        float* dL_dy_row = dL_dy + i * last_dim;
        float* a_grad_row = a_grad_data + i * last_dim;

        float dL_dy_dot_y = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            dL_dy_dot_y += dL_dy_row[j] * y_row[j];
        }

        for (int j = 0; j < last_dim; j++) {
            // This is dL/dx_j, which we add to the existing gradient
            a_grad_row[j] += y_row[j] * (dL_dy_row[j] - dL_dy_dot_y);
        }
    }
}

Value* softmax_ad(Arena* arena, Value* v) {
    Tensor* out_data = create_tensor(arena, v->data->n_dims, v->data->dims, TENSOR_TYPE_FLOAT);
    size_t total_elements = 1;
    for (int i = 0; i < v->data->n_dims; i++) {
        total_elements *= v->data->dims[i];
    }
    memcpy(out_data->data, v->data->data, total_elements * sizeof(float));
    softmax(out_data->data); // softmax is in-place in tensor.c, should be fixed
    Value** prev = (Value**)arena_alloc(arena, 1 * sizeof(Value*));
    prev[0] = v;
    return create_value(arena, out_data, prev, 1, NULL, backward_softmax);
}

void backward_scale(Value* v) {
    assert(v->n_prev == 1);
    Value* a = v->prev[0];
    float scalar = *(float*)v->op_context;
    Tensor* temp_grad = create_tensor(v->grad->n_dims, v->grad->dims, TENSOR_TYPE_FLOAT);
    scale(temp_grad, v->grad, scalar);
    add(a->grad, a->grad, temp_grad);
    free_tensor(temp_grad);
}

Value* scale_ad(Arena* arena, Value* a, float scalar) {
    Tensor* out_data = create_tensor(arena, a->data->n_dims, a->data->dims, TENSOR_TYPE_FLOAT);
    scale(out_data, a->data, scalar);
    Value** prev = (Value**)arena_alloc(arena, 1 * sizeof(Value*));
    prev[0] = a;
    float* context = (float*)arena_alloc(arena, sizeof(float));
    *context = scalar;
    return create_value(arena, out_data, prev, 1, context, backward_scale);
}

void backward_transpose(Value* v) {
    assert(v->n_prev == 1);
    Value* a = v->prev[0];
    int* dims = (int*)v->op_context;
    int dim1 = dims[0];
    int dim2 = dims[1];
    Tensor* temp_grad = create_tensor(v->grad->n_dims, a->grad->dims, TENSOR_TYPE_FLOAT);
    transpose(temp_grad, v->grad, dim1, dim2);
    add(a->grad, a->grad, temp_grad);
}

Value* transpose_ad(Arena* arena, Value* a, int dim1, int dim2) {
    int* new_dims = (int*)arena_alloc(arena, a->data->n_dims * sizeof(int));
    memcpy(new_dims, a->data->dims, a->data->n_dims * sizeof(int));
    int temp = new_dims[dim1];
    new_dims[dim1] = new_dims[dim2];
    new_dims[dim2] = temp;
    Tensor* out_data = create_tensor(arena, a->data->n_dims, new_dims, TENSOR_TYPE_FLOAT);
    transpose(out_data, a->data, dim1, dim2);
    Value** prev = (Value**)arena_alloc(arena, 1 * sizeof(Value*));
    prev[0] = a;
    int* context = (int*)arena_alloc(arena, 2 * sizeof(int));
    context[0] = dim1;
    context[1] = dim2;
    return create_value(arena, out_data, prev, 1, context, backward_transpose);
}

void backward_reshape(Value* v) {
    assert(v->n_prev == 1);
    Value* a = v->prev[0];
    // The gradient is just reshaped back to the original shape
    Tensor* temp_grad = create_tensor(a->data->n_dims, a->data->dims, TENSOR_TYPE_FLOAT);
    // Reshape is just a view, so we can copy the memory.
    // The total number of elements must be the same.
    size_t total_elements = 1;
    for (int i = 0; i < v->grad->n_dims; i++) total_elements *= v->grad->dims[i];
    memcpy(temp_grad->data, v->grad->data, total_elements * sizeof(float));
    add(a->grad, a->grad, temp_grad);
    free_tensor(temp_grad);
}

Value* reshape_ad(Arena* arena, Value* a, int n_dims, int* dims) {
    Tensor* out_data = create_tensor(arena, n_dims, dims, TENSOR_TYPE_FLOAT);
    // Reshape is a no-op on data for row-major tensors, just changes metadata.
    // We need to copy the data.
    size_t total_elements = 1;
    for (int i = 0; i < a->data->n_dims; i++) total_elements *= a->data->dims[i];
    memcpy(out_data->data, a->data->data, total_elements * sizeof(float));
    
    Value** prev = (Value**)arena_alloc(arena, 1 * sizeof(Value*));
    prev[0] = a;
    // No context needed for backward, original shape is in prev->data->dims
    return create_value(arena, out_data, prev, 1, NULL, backward_reshape);
}

void backward_relu(Value* v) {
    assert(v->n_prev == 1);
    Value* a = v->prev[0];
    float* a_data = (float*)a->data->data;
    float* grad_data = (float*)v->grad->data;
    float* a_grad_data = (float*)a->grad->data;
    
    size_t size = 1;
    for (int i = 0; i < a->data->n_dims; i++) {
        size *= a->data->dims[i];
    }

    for (size_t i = 0; i < size; i++) {
        if (a_data[i] > 0) {
            a_grad_data[i] += grad_data[i];
        }
    }
}

Value* relu_ad(Arena* arena, Value* a) {
    Tensor* out_data = create_tensor(arena, a->data->n_dims, a->data->dims, TENSOR_TYPE_FLOAT);
    float* in_data_p = (float*)a->data->data;
    float* out_data_p = (float*)out_data->data;
    size_t size = 1;
    for (int i = 0; i < a->data->n_dims; i++) {
        size *= a->data->dims[i];
    }
    for (size_t i = 0; i < size; i++) {
        out_data_p[i] = in_data_p[i] > 0 ? in_data_p[i] : 0;
    }
    Value** prev = (Value**)arena_alloc(arena, 1 * sizeof(Value*));
    prev[0] = a;
    return create_value(arena, out_data, prev, 1, NULL, backward_relu);
}


// hash set for visited nodes (simple open addressing)
#define HASHSET_SIZE 1048576
static Value** hashset_create() {
    Value** set = (Value**)calloc(HASHSET_SIZE, sizeof(Value*));
    return set;
}
static void hashset_free(Value** set) { free(set); }
static size_t hash_ptr(Value* v) { return ((uintptr_t)v) % HASHSET_SIZE; }
static bool hashset_contains(Value** set, Value* v) {
    size_t idx = hash_ptr(v);
    for (int i = 0; i < 16; i++) {
        size_t j = (idx + i) % HASHSET_SIZE;
        if (!set[j]) return false;
        if (set[j] == v) return true;
    }
    return false;
}
static void hashset_insert(Value** set, Value* v) {
    size_t idx = hash_ptr(v);
    for (int i = 0; i < 16; i++) {
        size_t j = (idx + i) % HASHSET_SIZE;
        if (!set[j]) { set[j] = v; return; }
        if (set[j] == v) return;
    }
    fprintf(stderr, "Hashset insert failed, consider increasing collision limit or table size\n");
}

// iterative topo sort
static Value** topo_sort_iter(Value* v, int* out_count) {
    Value** stack = (Value**)malloc(1024 * sizeof(Value*));
    int stack_size = 0, stack_cap = 1024;
    Value** sorted = (Value**)malloc(1024 * sizeof(Value*));
    int sorted_size = 0, sorted_cap = 1024;
    Value** visited = hashset_create();
    stack[stack_size++] = v;
    hashset_insert(visited, v); // Mark root as visited

    while (stack_size > 0) {
        Value* cur = stack[--stack_size];
        
        if (sorted_size >= sorted_cap) { sorted_cap *= 2; sorted = realloc(sorted, sorted_cap * sizeof(Value*)); }
        sorted[sorted_size++] = cur;

        for (int i = 0; i < cur->n_prev; i++) {
            if (!hashset_contains(visited, cur->prev[i])) {
                 if (stack_size >= stack_cap) { stack_cap *= 2; stack = realloc(stack, stack_cap * sizeof(Value*)); }
                stack[stack_size++] = cur->prev[i];
                hashset_insert(visited, cur->prev[i]);
            }
        }
    }
    hashset_free(visited);
    free(stack);
    
    // Reverse the order to get a correct topological sort
    for (int i = 0; i < sorted_size / 2; i++) {
        Value* temp = sorted[i];
        sorted[i] = sorted[sorted_size - 1 - i];
        sorted[sorted_size - 1 - i] = temp;
    }

    *out_count = sorted_size;
    return sorted;
}

void backward(Value* v) {
    int count = 0;
    Value** sorted_nodes = topo_sort_iter(v, &count);
    // Initialize gradient of the final value to 1.0
    free_tensor(v->grad);
    v->grad = create_tensor(v->data->n_dims, v->data->dims, TENSOR_TYPE_FLOAT);
    // Assuming the loss is a scalar, set the first element of its gradient to 1.
    if (v->data->n_dims > 0) {
        size_t total_elements = 1;
        for(int i=0; i<v->data->n_dims; i++) total_elements *= v->data->dims[i];
        for(size_t i=0; i<total_elements; i++) ((float*)v->grad->data)[i] = 1.0f / total_elements;
    } else {
        ((float*)v->grad->data)[0] = 1.0f;
    }
    
    for (int i = count - 1; i >= 0; i--) {
        if (sorted_nodes[i]->_backward) {
            sorted_nodes[i]->_backward(sorted_nodes[i]);
        }
    }
    free(sorted_nodes);
} 

// Frees the entire computation graph starting from a root node.
// It frees all intermediate Value structs, their data tensors, grad tensors, and op contexts.
// It does NOT free leaf nodes (inputs, weights) as they are managed elsewhere.
void free_graph(Value* v) {
    int count = 0;
    Value** sorted_nodes = topo_sort_iter(v, &count);
    for (int i = 0; i < count; i++) {
        Value* node = sorted_nodes[i];
        // Only free intermediate nodes, not leaves (weights/inputs)
        if (node->n_prev > 0) {
            free_value(node);
        } else {
            // For leaves, we only free the Value struct itself, not the data tensor
            // which is managed by the model or the user.
            free_tensor(node->grad);
            if (node->op_context) free(node->op_context);
            if (node->prev) free(node->prev);
            free(node);
        }
    }
    free(sorted_nodes);
}