#include <stdio.h>
#include <stdlib.h>
#include "transformer.h"
#include "tokenizer.h"
#include "optimizer.h"
#include "loss.h"
#include <string.h>
#include "memory.h"
#include "autodiff.h"
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>

#define DEFAULT_ARENA_SIZE (256 * 1024 * 1024) // 256 MB

#define LOG_INFO(fmt, ...) fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) fprintf(stderr, "[ERR] " fmt "\n", ##__VA_ARGS__)

#define MAX_DATASET_SIZE 1000000
#define MAX_LINE_LEN 1024

// simple dataset loader: loads lines from a text file
typedef struct {
    char** lines;
    int num_lines;
} TextDataset;

TextDataset* load_text_dataset(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;
    TextDataset* ds = (TextDataset*)malloc(sizeof(TextDataset));
    ds->lines = (char**)malloc(MAX_DATASET_SIZE * sizeof(char*));
    ds->num_lines = 0;
    char buf[MAX_LINE_LEN];
    while (fgets(buf, sizeof(buf), f) && ds->num_lines < MAX_DATASET_SIZE) {
        size_t len = strlen(buf);
        if (len > 0 && buf[len-1] == '\n') buf[len-1] = 0;
        ds->lines[ds->num_lines] = strdup(buf);
        ds->num_lines++;
    }
    fclose(f);
    return ds;
}
void free_text_dataset(TextDataset* ds) {
    for (int i = 0; i < ds->num_lines; i++) free(ds->lines[i]);
    free(ds->lines);
    free(ds);
}
void shuffle_dataset(TextDataset* ds) {
    srand(time(NULL));
    for (int i = ds->num_lines - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char* tmp = ds->lines[i];
        ds->lines[i] = ds->lines[j];
        ds->lines[j] = tmp;
    }
}

void test_transformer(Transformer* model) {
    LOG_INFO("running transformer test...");
    int batch_size = 2, seq_len = 8;
    // Create a temporary arena for the test
    Arena* test_arena = create_arena(1024 * 1024 * 32); // 32MB should be enough
    Tensor* src = create_tensor(test_arena, 2, (int[]){batch_size, seq_len}, TENSOR_TYPE_INT);
    Tensor* tgt = create_tensor(test_arena, 2, (int[]){batch_size, seq_len}, TENSOR_TYPE_INT);
    for (int i = 0; i < batch_size * seq_len; i++) ((int*)src->data)[i] = i % 256;
    for (int i = 0; i < batch_size * seq_len; i++) ((int*)tgt->data)[i] = (i + 1) % 256;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    Value* logits = transformer_forward_ad(test_arena, src, tgt, model, 1);
    Value* loss = cross_entropy_loss_ad(test_arena, logits, tgt);
    backward(loss);
    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-6 * (t1.tv_usec - t0.tv_usec);
    LOG_INFO("test forward+backward time: %.3fs", elapsed);
    LOG_INFO("test loss: %f", ((float*)loss->data->data)[0]);
    // No need to free tensors, just the arena
    destroy_arena(test_arena);
}

int main(int argc, char** argv) {
    size_t arena_size = DEFAULT_ARENA_SIZE;
    char* arena_size_env = getenv("ARENA_SIZE_MB");
    if (arena_size_env) {
        arena_size = atol(arena_size_env) * 1024 * 1024;
    }
    Arena* training_arena = create_arena(arena_size);
    if (!training_arena) {
        LOG_ERR("Failed to create memory arena of size %zu bytes", arena_size);
        return 1;
    }
    LOG_INFO("Memory arena created: %zu MB", arena_size / 1024 / 1024);

    // CLI args for checkpointing
    const char* save_path = NULL;
    const char* load_path = NULL;
    const char* data_path = NULL;
    int run_test = 0;
    int opt;
    while ((opt = getopt(argc, argv, "")) != -1) {}
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) save_path = argv[++i];
        if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) load_path = argv[++i];
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) data_path = argv[++i];
        if (strcmp(argv[i], "--test") == 0) run_test = 1;
    }
    // Hyperparameters
    int n_layers = 1;
    int embed_dim = 128;
    int n_heads = 4;
    int ff_hidden_dim = 512;
    int batch_size = 1;
    int seq_len = 10;
    int vocab_size = 256;
    int max_seq_len = 512;
    float learning_rate = 0.001f;
    int epochs = 10;
    int gradient_accumulation_steps = 4; // accumulate gradients over 4 batches
    if (n_layers <= 0 || embed_dim <= 0 || n_heads <= 0 || ff_hidden_dim <= 0 || batch_size <= 0 || seq_len <= 0 || vocab_size <= 0 || max_seq_len <= 0 || learning_rate <= 0) {
        LOG_ERR("invalid hyperparameters");
        return 1;
    }
    if (embed_dim % n_heads != 0) {
        LOG_ERR("embed_dim must be divisible by n_heads");
        return 1;
    }
    Transformer* model = create_transformer(n_layers, vocab_size, max_seq_len, embed_dim, n_heads, ff_hidden_dim);
    if (!model) {
        LOG_ERR("failed to create transformer model");
        return 1;
    }
    Optimizer* optimizer = create_optimizer(model, learning_rate);
    if (!optimizer) {
        LOG_ERR("failed to create optimizer");
        free_transformer(model);
        return 1;
    }
    TextDataset* dataset = NULL;
    if (data_path) {
        dataset = load_text_dataset(data_path);
        if (!dataset) {
            LOG_ERR("failed to load dataset from %s", data_path);
            free_optimizer(optimizer);
            free_transformer(model);
            return 1;
        }
        shuffle_dataset(dataset);
        LOG_INFO("loaded %d lines from dataset", dataset->num_lines);
    }
    if (load_path) {
        if (!load_transformer(model, load_path)) {
            LOG_ERR("failed to load checkpoint from %s", load_path);
        } else {
            LOG_INFO("loaded checkpoint from %s", load_path);
        }
    }
    if (run_test) {
        test_transformer(model);
        free_optimizer(optimizer);
        free_transformer(model);
        return 0;
    }
    for (int epoch = 0; epoch < epochs; epoch++) {
        int batch_count = 0;
        for (int batch = 0; batch < (dataset ? dataset->num_lines : 1); batch += batch_size) {
            int this_batch = dataset ? (batch + batch_size <= dataset->num_lines ? batch_size : dataset->num_lines - batch) : 1;
            Tensor* src_in = create_tensor(training_arena, 2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
            Tensor* tgt_in = create_tensor(training_arena, 2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
            if (!src_in || !tgt_in) {
                LOG_ERR("failed to allocate input tensors");
                break;
            }
            for (int b = 0; b < this_batch; b++) {
                const char* text = dataset ? dataset->lines[batch + b] : "hello world";
                int token_ids[seq_len];
                int num_tokens = encode(text, token_ids, seq_len);
                memcpy((int*)src_in->data + b * seq_len, token_ids, num_tokens * sizeof(int));
                memcpy((int*)tgt_in->data + b * seq_len, token_ids, num_tokens * sizeof(int));
            }
            Value* logits = transformer_forward_ad(training_arena, src_in, tgt_in, model, 1);
            if (!logits) { LOG_ERR("forward pass failed"); break; }
            Value* loss = cross_entropy_loss_ad(training_arena, logits, tgt_in);
            if (!loss) { LOG_ERR("loss computation failed"); break; }
            LOG_INFO("Epoch %d Batch %d Loss: %f", epoch, batch / batch_size, ((float*)loss->data->data)[0]);
            backward(loss);

            if ((batch_count + 1) % gradient_accumulation_steps == 0) {
                optimizer_step(optimizer);
                zero_grad(optimizer);
            }
            
            arena_reset(training_arena);
            batch_count++;
        }
    }
    if (save_path) {
        if (!save_transformer(model, save_path)) {
            LOG_ERR("failed to save checkpoint to %s", save_path);
        } else {
            LOG_INFO("saved checkpoint to %s", save_path);
        }
    }
    if (dataset) free_text_dataset(dataset);
    free_optimizer(optimizer);
    free_transformer(model);
    destroy_arena(training_arena);
    LOG_INFO("Training complete.");
    return 0;
}
