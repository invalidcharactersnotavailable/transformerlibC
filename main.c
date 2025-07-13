#include <stdio.h>
#include <stdlib.h>
#include "transformer.h"
#include "tokenizer.h"
#include "optimizer.h"
#include "loss.h"
#include <string.h>
#include "autodiff.h"
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>

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
    Tensor* src = create_tensor(2, (int[]){batch_size, seq_len}, TENSOR_TYPE_INT);
    Tensor* tgt = create_tensor(2, (int[]){batch_size, seq_len}, TENSOR_TYPE_INT);
    for (int i = 0; i < batch_size * seq_len; i++) {
        ((int*)src->data)[i] = i % model->vocab_size;
        ((int*)tgt->data)[i] = (i + 1) % model->vocab_size;
    }
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    Value* logits = transformer_forward_ad(src, tgt, model, 1);
    Value* loss = cross_entropy_loss_ad(logits, tgt);
    backward(loss);
    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-6 * (t1.tv_usec - t0.tv_usec);
    LOG_INFO("test forward+backward time: %.3fs", elapsed);
    LOG_INFO("test loss: %f", ((float*)loss->data->data)[0]);
    free_tensor(src);
    free_tensor(tgt);
    free_graph(loss);
}

int main(int argc, char** argv) {

    // CLI args
    const char* save_path = NULL;
    const char* load_path = NULL;
    const char* data_path = NULL;
    int run_test = 0;
    int n_layers = 2;
    int embed_dim = 512;
    int n_heads = 8;
    int ff_hidden_dim = 2048;
    int batch_size = 1;
    int seq_len = 10;
    int vocab_size = 10000;
    int max_seq_len = 512;
    float learning_rate = 0.001f;
    int epochs = 10;
    int gradient_accumulation_steps = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) save_path = argv[++i];
        else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) load_path = argv[++i];
        else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) data_path = argv[++i];
        else if (strcmp(argv[i], "--test") == 0) run_test = 1;
        else if (strcmp(argv[i], "--n_layers") == 0 && i + 1 < argc) n_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--embed_dim") == 0 && i + 1 < argc) embed_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n_heads") == 0 && i + 1 < argc) n_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ff_hidden_dim") == 0 && i + 1 < argc) ff_hidden_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch_size") == 0 && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq_len") == 0 && i + 1 < argc) seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vocab_size") == 0 && i + 1 < argc) vocab_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max_seq_len") == 0 && i + 1 < argc) max_seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--learning_rate") == 0 && i + 1 < argc) learning_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gradient_accumulation_steps") == 0 && i + 1 < argc) gradient_accumulation_steps = atoi(argv[++i]);
    }
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
    LOG_INFO("Transformer model created with %ld parameters", get_transformer_param_count(model));
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
            Tensor* src_in = create_tensor(2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
            Tensor* tgt_in = create_tensor(2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
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
            Value* logits = transformer_forward_ad(src_in, tgt_in, model, 1);
            if (!logits) { LOG_ERR("forward pass failed"); break; }
            Value* loss = cross_entropy_loss_ad(logits, tgt_in);
            if (!loss) { LOG_ERR("loss computation failed"); break; }
            LOG_INFO("Epoch %d Batch %d Loss: %f", epoch, batch / batch_size, ((float*)loss->data->data)[0]);
            backward(loss);

            if ((batch_count + 1) % gradient_accumulation_steps == 0) {
                optimizer_step(optimizer);
                zero_grad(optimizer);
            }
            
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
    if (dataset) {
        // Split dataset into training and validation sets
        TextDataset* train_ds = (TextDataset*)malloc(sizeof(TextDataset));
        TextDataset* val_ds = (TextDataset*)malloc(sizeof(TextDataset));
        int val_size = dataset->num_lines / 10; // 10% for validation
        int train_size = dataset->num_lines - val_size;

        train_ds->lines = (char**)malloc(train_size * sizeof(char*));
        train_ds->num_lines = train_size;
        memcpy(train_ds->lines, dataset->lines, train_size * sizeof(char*));

        val_ds->lines = (char**)malloc(val_size * sizeof(char*));
        val_ds->num_lines = val_size;
        memcpy(val_ds->lines, dataset->lines + train_size, val_size * sizeof(char*));

        free_text_dataset(dataset);

        for (int epoch = 0; epoch < epochs; epoch++) {
            int batch_count = 0;
            for (int batch = 0; batch < train_ds->num_lines; batch += batch_size) {
                int this_batch = (batch + batch_size <= train_ds->num_lines) ? batch_size : train_ds->num_lines - batch;
                Tensor* src_in = create_tensor(2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
                Tensor* tgt_in = create_tensor(2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
                if (!src_in || !tgt_in) {
                    LOG_ERR("failed to allocate input tensors");
                    break;
                }
                for (int b = 0; b < this_batch; b++) {
                    const char* text = train_ds->lines[batch + b];
                    int token_ids[seq_len];
                    int num_tokens = encode(text, token_ids, seq_len);
                    memcpy((int*)src_in->data + b * seq_len, token_ids, num_tokens * sizeof(int));
                    memcpy((int*)tgt_in->data + b * seq_len, token_ids, num_tokens * sizeof(int));
                }
                Value* logits = transformer_forward_ad(src_in, tgt_in, model, 1);
                if (!logits) { LOG_ERR("forward pass failed"); break; }
                Value* loss = cross_entropy_loss_ad(logits, tgt_in);
                if (!loss) { LOG_ERR("loss computation failed"); break; }

                // Progress bar
                int bar_width = 50;
                float progress = (float)(batch + this_batch) / train_ds->num_lines;
                int pos = bar_width * progress;
                printf("[");
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) printf("=");
                    else if (i == pos) printf(">");
                    else printf(" ");
                }
                printf("] %d/%d Loss: %f\r", batch + this_batch, train_ds->num_lines, ((float*)loss->data->data)[0]);
                fflush(stdout);

                backward(loss);

                if ((batch_count + 1) % gradient_accumulation_steps == 0) {
                    optimizer_step(optimizer);
                    zero_grad(optimizer);
                }

                batch_count++;
            }

            // Validation
            float val_loss = 0.0f;
            int val_batches = 0;
            for (int batch = 0; batch < val_ds->num_lines; batch += batch_size) {
                int this_batch = (batch + batch_size <= val_ds->num_lines) ? batch_size : val_ds->num_lines - batch;
                Tensor* src_in = create_tensor(2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
                Tensor* tgt_in = create_tensor(2, (int[]){this_batch, seq_len}, TENSOR_TYPE_INT);
                if (!src_in || !tgt_in) {
                    LOG_ERR("failed to allocate validation input tensors");
                    break;
                }
                for (int b = 0; b < this_batch; b++) {
                    const char* text = val_ds->lines[batch + b];
                    int token_ids[seq_len];
                    int num_tokens = encode(text, token_ids, seq_len);
                    memcpy((int*)src_in->data + b * seq_len, token_ids, num_tokens * sizeof(int));
                    memcpy((int*)tgt_in->data + b * seq_len, token_ids, num_tokens * sizeof(int));
                }
                Value* logits = transformer_forward_ad(src_in, tgt_in, model, 0); // No training
                if (!logits) { LOG_ERR("validation forward pass failed"); break; }
                Value* loss = cross_entropy_loss_ad(logits, tgt_in);
                if (!loss) { LOG_ERR("validation loss computation failed"); break; }
                val_loss += ((float*)loss->data->data)[0];
                val_batches++;
                free_graph(loss);
            }
            if (val_batches > 0) {
                LOG_INFO("Epoch %d Validation Loss: %f", epoch, val_loss / val_batches);
            }
        }

        free_text_dataset(train_ds);
        free_text_dataset(val_ds);
    }

    if (save_path) {
        if (!save_transformer(model, save_path)) {
            LOG_ERR("failed to save checkpoint to %s", save_path);
        } else {
            LOG_INFO("saved checkpoint to %s", save_path);
        }
    }

    free_optimizer(optimizer);
    free_transformer(model);
    LOG_INFO("Training complete.");
    return 0;
}
