#include "autodiff.h"
#include "tensor.h"
#include "memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

Value* create_value(Tensor* data, Value** prev, int n_prev, void* op_context, Arena* arena, void (*_backward)(struct Value*)) {
    Value* v;
    if (arena) {
        v = (Value*)arena_alloc(arena, sizeof(Value));
    } else {
        v = (Value*)malloc(sizeof(Value));
    }
    v->data = data;
    // Gradients are persistent, not in the temporary arena
    v->grad = create_tensor(data->n_dims, data->dims, TENSOR_TYPE_FLOAT, NULL); 
    v->_backward = _backward;
    v->prev = prev;
    v->n_prev = n_prev;
    v->op_context = op_context;
    v->arena = arena;
    return v;
}

void free_value(Value* v) {
    if (v && v->arena == NULL) {
        free_tensor(v->data);
        free_tensor(v->grad);
        // prev is now arena allocated if the value is, so don't free it here.
        // if (v->prev) free(v->prev); 
        if (v->op_context) free(v->op_context);
        free(v);
    }
}

void backward_add(Value* v) {
    assert(v->n_prev == 2);
    Value* a = v->prev[0];
    Value* b = v->prev[1];
    // Gradient of add is 1, so we just add the output grad to both inputs.
    add(a->grad, a->grad, v->grad);
    add(b->grad, b->grad, v->grad);
}

Value* add_ad(Value* a, Value* b, Arena* arena) {
    Tensor* out_data = create_tensor(a->data->n_dims, a->data->dims, TENSOR_TYPE_FLOAT, arena);
    add(out_data, a->data, b->data);
    Value** prev = (Value**)arena_alloc(arena, 2 * sizeof(Value*));
    prev[0] = a;
    prev[1] = b;
    return create_value(out_data, prev, 2, NULL, arena, backward_add);
}

void backward_matmul(Value* v) {
    assert(v->n_prev == 2);
    Value* a = v->prev[0];
    Value* b = v->prev[1];
    Arena* arena = v->arena;

    // Grad with respect to a: grad * b^T
    int b_t_dims[] = {b->data->dims[1], b->data->dims[0]};
    Tensor* b_t = create_tensor(b->data->n_dims, b_t_dims, TENSOR_TYPE_FLOAT, arena);
    transpose(b_t, b->data, 0, 1);
    
    Tensor* da = create_tensor(a->grad->n_dims, a->grad->dims, TENSOR_TYPE_FLOAT, arena);
    matmul(da, v->grad, b_t);
    add(a->grad, a->grad, da);

    // Grad with respect to b: a^T * grad
    int a_t_dims[] = {a->data->dims[1], a->data->dims[0]};
    Tensor* a_t = create_tensor(a->data->n_dims, a_t_dims, TENSOR_TYPE_FLOAT, arena);
    transpose(a_t, a->data, 0, 1);
    
    Tensor* db = create_tensor(b->grad->n_dims, b->grad->dims, TENSOR_TYPE_FLOAT, arena);
    matmul(db, a_t, v->grad);
    add(b->grad, b->grad, db);
}

Value* matmul_ad(Value* a, Value* b, Arena* arena) {
    // Simplified matmul for 2D tensors
    int out_dims[] = {a->data->dims[0], b->data->dims[1]};
    Tensor* out_data = create_tensor(2, out_dims, TENSOR_TYPE_FLOAT, arena);
    matmul(out_data, a->data, b->data);
    Value** prev = (Value**)arena_alloc(arena, 2 * sizeof(Value*));
    prev[0] = a;
    prev[1] = b;
    return create_value(out_data, prev, 2, NULL, arena, backward_matmul);
}

void backward_softmax(Value* v) {
    // The backward pass for softmax is complex and not implemented yet.
}

Value* softmax_ad(Value* v, Arena* arena) {
    Tensor* out_data = create_tensor(v->data->n_dims, v->data->dims, TENSOR_TYPE_FLOAT, arena);
    size_t total_elements = 1;
    for (int i = 0; i < v->data->n_dims; i++) {
        total_elements *= v->data->dims[i];
    }
    memcpy(out_data->data, v->data->data, total_elements * sizeof(float));
    softmax(out_data);
    Value** prev = (Value**)arena_alloc(arena, sizeof(Value*));
    prev[0] = v;
    return create_value(out_data, prev, 1, NULL, arena, backward_softmax);
}

void backward_scale(Value* v) {
    assert(v->n_prev == 1);
    Value* a = v->prev[0];
    float scalar = *(float*)v->op_context;
    Tensor* temp_grad = create_tensor(v->grad->n_dims, v->grad->dims, TENSOR_TYPE_FLOAT, v->arena);
    scale(temp_grad, v->grad, scalar);
    add(a->grad, a->grad, temp_grad);
}

Value* scale_ad(Value* a, float scalar, Arena* arena) {
    Tensor* out_data = create_tensor(a->data->n_dims, a->data->dims, TENSOR_TYPE_FLOAT, arena);
    scale(out_data, a->data, scalar);
    Value** prev = (Value**)arena_alloc(arena, sizeof(Value*));
    prev[0] = a;
    float* context = (float*)arena_alloc(arena, sizeof(float));
    *context = scalar;
    return create_value(out_data, prev, 1, context, arena, backward_scale);
}

void backward_transpose(Value* v) {
    assert(v->n_prev == 1);
    Value* a = v->prev[0];
    int* dims = (int*)v->op_context;
    int dim1 = dims[0];
    int dim2 = dims[1];
    Tensor* temp_grad = create_tensor(v->grad->n_dims, a->grad->dims, TENSOR_TYPE_FLOAT, v->arena);
    transpose(temp_grad, v->grad, dim1, dim2);
    add(a->grad, a->grad, temp_grad);
}

Value* transpose_ad(Value* a, int dim1, int dim2, Arena* arena) {
    int* new_dims = (int*)arena_alloc(arena, a->data->n_dims * sizeof(int));
    memcpy(new_dims, a->data->dims, a->data->n_dims * sizeof(int));
    int temp = new_dims[dim1];
    new_dims[dim1] = new_dims[dim2];
    new_dims[dim2] = temp;
    Tensor* out_data = create_tensor(a->data->n_dims, new_dims, TENSOR_TYPE_FLOAT, arena);
    transpose(out_data, a->data, dim1, dim2);
    Value** prev = (Value**)arena_alloc(arena, sizeof(Value*));
    prev[0] = a;
    int* context = (int*)arena_alloc(arena, 2 * sizeof(int));
    context[0] = dim1;
    context[1] = dim2;
    return create_value(out_data, prev, 1, context, arena, backward_transpose);
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

Value* relu_ad(Value* a, Arena* arena) {
    Tensor* out_data = create_tensor(a->data->n_dims, a->data->dims, TENSOR_TYPE_FLOAT, arena);
    float* in_data_p = (float*)a->data->data;
    float* out_data_p = (float*)out_data->data;
    size_t size = 1;
    for (int i = 0; i < a->data->n_dims; i++) {
        size *= a->data->dims[i];
    }
    for (size_t i = 0; i < size; i++) {
        out_data_p[i] = in_data_p[i] > 0 ? in_data_p[i] : 0;
    }

    Value** prev = (Value**)arena_alloc(arena, sizeof(Value*));
    prev[0] = a;
    return create_value(out_data, prev, 1, NULL, arena, backward_relu);
}


// NOTE: This recursive implementation of topological sort can be inefficient for deep graphs
// and may lead to stack overflow. An iterative version would be more robust.
// Also, the visited check is missing, which is needed for DAGs, not just trees.
static void topological_sort(Value* v, Value** sorted, int* count) {
    if (v->arena == NULL) { // Parameters don't have an arena, consider them visited.
      return;
    }
    // A proper implementation would use a hash set to mark visited nodes.
    // This is a simplified version.
    for (int i = 0; i < v->n_prev; i++) {
        topological_sort(v->prev[i], sorted, count);
    }
    sorted[(*count)++] = v;
}


// Simplified topological sort and backward pass
void backward(Value* v, Arena* arena) {
    // This function performs backpropagation starting from a given value (usually the loss).
    // It requires a topological sort of the computation graph.
    
    // NOTE: The size of this array is a major limitation. For a 100M model, the
    // computation graph can have millions of nodes. This must be dynamically sized.
    // For now, we allocate a large-ish buffer from the arena.
    int max_nodes = 1000000; // Still a guess, might not be enough
    Value** sorted_nodes = (Value**)arena_alloc(arena, max_nodes * sizeof(Value*));

    int count = 0;
    topological_sort(v, sorted_nodes, &count);

    // Filter duplicates from sorted_nodes (a consequence of the simple topo sort)
    // A hash set would be much more efficient here.
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (sorted_nodes[i] == sorted_nodes[j]) {
                // Shift elements to the left
                for (int k = j; k < count - 1; k++) {
                    sorted_nodes[k] = sorted_nodes[k+1];
                }
                count--;
                j--;
            }
        }
    }

    // 2. Initialize gradient of the output value to 1.
    v->grad = create_tensor(v->data->n_dims, v->data->dims, TENSOR_TYPE_FLOAT, NULL);
    ((float*)v->grad->data)[0] = 1.0f;

    // 3. Propagate gradients backwards through the sorted list.
    for (int i = count - 1; i >= 0; i--) {
        if (sorted_nodes[i]->_backward) {
            sorted_nodes[i]->_backward(sorted_nodes[i]);
        }
    }
} 