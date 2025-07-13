#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <signal.h>
#include <errno.h>

#include "transformer.h"
#include "tokenizer.h"
#include "optimizer.h"
#include "loss.h"
#include "memory.h"
#include "autodiff.h"
#include "dataset.h"
#include "inference.h"

#define LOG_INFO(fmt, ...) fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) fprintf(stderr, "[ERR] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)

// Global variables for signal handling
static volatile int should_stop = 0;
static Transformer* global_model = NULL;
static Optimizer* global_optimizer = NULL;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    LOG_INFO("Received signal %d, shutting down gracefully...", sig);
    should_stop = 1;
}

// Training configuration
typedef struct {
    // Model architecture
    int n_layers;
    int embed_dim;
    int n_heads;
    int ff_hidden_dim;
    int vocab_size;
    int max_seq_len;
    
    // Training parameters
    int batch_size;
    int seq_len;
    float learning_rate;
    int epochs;
    int gradient_accumulation_steps;
    char optimizer_type[16];
    
    // Data parameters
    char data_path[1024];
    char data_dir[1024];
    char finetune_dir[1024];
    char vocab_file[1024];
    
    // I/O parameters
    char save_path[1024];
    char load_path[1024];
    
    // Inference parameters
    int inference_mode;
    int interactive_mode;
    char prompt[1024];
    int max_length;
    float temperature;
    int top_k;
    
    // System parameters
    int test_mode;
    int verbose;
    int seed;
    size_t arena_size_mb;
} Config;

void set_default_config(Config* config) {
    if (!config) return;
    
    // Model defaults
    config->n_layers = 2;
    config->embed_dim = 512;
    config->n_heads = 8;
    config->ff_hidden_dim = 2048;
    config->vocab_size = 10000;
    config->max_seq_len = 512;
    
    // Training defaults
    config->batch_size = 1;
    config->seq_len = 10;
    config->learning_rate = 0.001f;
    config->epochs = 10;
    config->gradient_accumulation_steps = 4;
    strcpy(config->optimizer_type, "adam");
    
    // Data defaults
    config->data_path[0] = '\0';
    config->data_dir[0] = '\0';
    config->finetune_dir[0] = '\0';
    config->vocab_file[0] = '\0';
    
    // I/O defaults
    config->save_path[0] = '\0';
    config->load_path[0] = '\0';
    
    // Inference defaults
    config->inference_mode = 0;
    config->interactive_mode = 0;
    config->prompt[0] = '\0';
    config->max_length = 100;
    config->temperature = 1.0f;
    config->top_k = 40;
    
    // System defaults
    config->test_mode = 0;
    config->verbose = 0;
    config->seed = 42;
    config->arena_size_mb = 4096;
}

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("OPTIONS:\n");
    printf("  Model Architecture:\n");
    printf("    --n_layers <int>           Number of transformer layers (default: 2)\n");
    printf("    --embed_dim <int>          Embedding dimension (default: 512)\n");
    printf("    --n_heads <int>            Number of attention heads (default: 8)\n");
    printf("    --ff_hidden_dim <int>      Feed-forward hidden dimension (default: 2048)\n");
    printf("    --vocab_size <int>         Vocabulary size (default: 10000)\n");
    printf("    --max_seq_len <int>        Maximum sequence length (default: 512)\n\n");
    
    printf("  Training Parameters:\n");
    printf("    --batch_size <int>         Batch size (default: 1)\n");
    printf("    --seq_len <int>            Training sequence length (default: 10)\n");
    printf("    --learning_rate <float>    Learning rate (default: 0.001)\n");
    printf("    --epochs <int>             Number of training epochs (default: 10)\n");
    printf("    --gradient_accumulation_steps <int>  Gradient accumulation steps (default: 4)\n");
    printf("    --optimizer <string>       Optimizer type: sgd or adam (default: adam)\n\n");
    
    printf("  Data & I/O:\n");
    printf("    --data <file>              Single text file for training\n");
    printf("    --data_dir <dir>           Directory containing text files for training\n");
    printf("    --finetune_dir <dir>       Directory with JSON files for finetuning\n");
    printf("    --vocab_file <file>        Custom vocabulary file\n");
    printf("    --save <file>              Save model checkpoint\n");
    printf("    --load <file>              Load model checkpoint\n\n");
    
    printf("  Inference:\n");
    printf("    --inference                Enable inference mode\n");
    printf("    --prompt <text>            Input prompt for generation\n");
    printf("    --max_length <int>         Maximum generation length (default: 100)\n");
    printf("    --temperature <float>      Sampling temperature (default: 1.0)\n");
    printf("    --top_k <int>              Top-k sampling (default: 40)\n");
    printf("    --interactive              Interactive generation mode\n\n");
    
    printf("  System:\n");
    printf("    --test                     Run performance test and exit\n");
    printf("    --verbose                  Enable verbose logging\n");
    printf("    --seed <int>               Random seed (default: 42)\n");
    printf("    --arena_size_mb <int>      Training arena size in MB (default: 4096)\n");
    printf("    --help                     Show this help message\n\n");
    
    printf("EXAMPLES:\n");
    printf("  # Train on a text file\n");
    printf("  %s --data data.txt --save model.ckpt --epochs 10\n\n", program_name);
    
    printf("  # Train on a directory of text files\n");
    printf("  %s --data_dir ./corpus/ --save model.ckpt --epochs 20\n\n", program_name);
    
    printf("  # Finetune on JSON dataset\n");
    printf("  %s --load base_model.ckpt --finetune_dir ./finetune_data/ --save finetuned.ckpt\n\n", program_name);
    
    printf("  # Run inference\n");
    printf("  %s --load model.ckpt --inference --prompt \"Hello world\" --max_length 50\n\n", program_name);
    
    printf("  # Interactive generation\n");
    printf("  %s --load model.ckpt --interactive\n\n", program_name);
}

int parse_arguments(int argc, char** argv, Config* config) {
    int opt;
    const char* short_options = "";
    struct option long_options[] = {
        // Model architecture
        {"n_layers", required_argument, 0, 0},
        {"embed_dim", required_argument, 0, 0},
        {"n_heads", required_argument, 0, 0},
        {"ff_hidden_dim", required_argument, 0, 0},
        {"vocab_size", required_argument, 0, 0},
        {"max_seq_len", required_argument, 0, 0},
        
        // Training parameters
        {"batch_size", required_argument, 0, 0},
        {"seq_len", required_argument, 0, 0},
        {"learning_rate", required_argument, 0, 0},
        {"epochs", required_argument, 0, 0},
        {"gradient_accumulation_steps", required_argument, 0, 0},
        {"optimizer", required_argument, 0, 0},
        
        // Data & I/O
        {"data", required_argument, 0, 0},
        {"data_dir", required_argument, 0, 0},
        {"finetune_dir", required_argument, 0, 0},
        {"vocab_file", required_argument, 0, 0},
        {"save", required_argument, 0, 0},
        {"load", required_argument, 0, 0},
        
        // Inference
        {"inference", no_argument, 0, 0},
        {"prompt", required_argument, 0, 0},
        {"max_length", required_argument, 0, 0},
        {"temperature", required_argument, 0, 0},
        {"top_k", required_argument, 0, 0},
        {"interactive", no_argument, 0, 0},
        
        // System
        {"test", no_argument, 0, 0},
        {"verbose", no_argument, 0, 0},
        {"seed", required_argument, 0, 0},
        {"arena_size_mb", required_argument, 0, 0},
        {"help", no_argument, 0, 0},
        
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
        switch (opt) {
            case 0:
                if (strcmp(long_options[option_index].name, "n_layers") == 0) {
                    config->n_layers = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "embed_dim") == 0) {
                    config->embed_dim = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "n_heads") == 0) {
                    config->n_heads = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ff_hidden_dim") == 0) {
                    config->ff_hidden_dim = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "vocab_size") == 0) {
                    config->vocab_size = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "max_seq_len") == 0) {
                    config->max_seq_len = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "batch_size") == 0) {
                    config->batch_size = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "seq_len") == 0) {
                    config->seq_len = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "learning_rate") == 0) {
                    config->learning_rate = atof(optarg);
                } else if (strcmp(long_options[option_index].name, "epochs") == 0) {
                    config->epochs = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "gradient_accumulation_steps") == 0) {
                    config->gradient_accumulation_steps = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "optimizer") == 0) {
                    strncpy(config->optimizer_type, optarg, sizeof(config->optimizer_type) - 1);
                } else if (strcmp(long_options[option_index].name, "data") == 0) {
                    strncpy(config->data_path, optarg, sizeof(config->data_path) - 1);
                } else if (strcmp(long_options[option_index].name, "data_dir") == 0) {
                    strncpy(config->data_dir, optarg, sizeof(config->data_dir) - 1);
                } else if (strcmp(long_options[option_index].name, "finetune_dir") == 0) {
                    strncpy(config->finetune_dir, optarg, sizeof(config->finetune_dir) - 1);
                } else if (strcmp(long_options[option_index].name, "vocab_file") == 0) {
                    strncpy(config->vocab_file, optarg, sizeof(config->vocab_file) - 1);
                } else if (strcmp(long_options[option_index].name, "save") == 0) {
                    strncpy(config->save_path, optarg, sizeof(config->save_path) - 1);
                } else if (strcmp(long_options[option_index].name, "load") == 0) {
                    strncpy(config->load_path, optarg, sizeof(config->load_path) - 1);
                } else if (strcmp(long_options[option_index].name, "inference") == 0) {
                    config->inference_mode = 1;
                } else if (strcmp(long_options[option_index].name, "prompt") == 0) {
                    strncpy(config->prompt, optarg, sizeof(config->prompt) - 1);
                } else if (strcmp(long_options[option_index].name, "max_length") == 0) {
                    config->max_length = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "temperature") == 0) {
                    config->temperature = atof(optarg);
                } else if (strcmp(long_options[option_index].name, "top_k") == 0) {
                    config->top_k = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "interactive") == 0) {
                    config->interactive_mode = 1;
                } else if (strcmp(long_options[option_index].name, "test") == 0) {
                    config->test_mode = 1;
                } else if (strcmp(long_options[option_index].name, "verbose") == 0) {
                    config->verbose = 1;
                } else if (strcmp(long_options[option_index].name, "seed") == 0) {
                    config->seed = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "arena_size_mb") == 0) {
                    config->arena_size_mb = atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "help") == 0) {
                    print_usage(argv[0]);
                    return 1;
                }
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    return 0;
}

int validate_config(const Config* config) {
    if (!config) return 0;
    
    // Validate model parameters
    if (config->n_layers <= 0 || config->embed_dim <= 0 || 
        config->n_heads <= 0 || config->ff_hidden_dim <= 0 ||
        config->vocab_size <= 0 || config->max_seq_len <= 0) {
        LOG_ERR("Invalid model parameters");
        return 0;
    }
    
    if (config->embed_dim % config->n_heads != 0) {
        LOG_ERR("embed_dim must be divisible by n_heads");
        return 0;
    }
    
    // Validate training parameters
    if (config->batch_size <= 0 || config->seq_len <= 0 || 
        config->learning_rate <= 0 || config->epochs <= 0 ||
        config->gradient_accumulation_steps <= 0) {
        LOG_ERR("Invalid training parameters");
        return 0;
    }
    
    // Validate inference parameters
    if (config->inference_mode && config->max_length <= 0) {
        LOG_ERR("Invalid inference parameters");
        return 0;
    }
    
    // Check that we have some data source
    if (!config->inference_mode && !config->test_mode &&
        config->data_path[0] == '\0' && config->data_dir[0] == '\0' && 
        config->finetune_dir[0] == '\0') {
        LOG_ERR("No data source specified (use --data, --data_dir, or --finetune_dir)");
        return 0;
    }
    
    return 1;
}

void test_transformer(Transformer* model) {
    LOG_INFO("Running transformer test...");
    
    int batch_size = 2, seq_len = 8;
    Tensor* src = create_tensor(2, (int[]){batch_size, seq_len}, TENSOR_TYPE_INT);
    Tensor* tgt = create_tensor(2, (int[]){batch_size, seq_len}, TENSOR_TYPE_INT);
    
    if (!src || !tgt) {
        LOG_ERR("Failed to create test tensors");
        return;
    }
    
    // Initialize with test data
    for (int i = 0; i < batch_size * seq_len; i++) {
        ((int*)src->data)[i] = i % model->vocab_size;
        ((int*)tgt->data)[i] = (i + 1) % model->vocab_size;
    }
    
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    
    Value* logits = transformer_forward_ad(src, tgt, model, 1);
    if (!logits) {
        LOG_ERR("Forward pass failed");
        free_tensor(src);
        free_tensor(tgt);
        return;
    }
    
    Value* loss = cross_entropy_loss_ad(logits, tgt);
    if (!loss) {
        LOG_ERR("Loss computation failed");
        free_tensor(src);
        free_tensor(tgt);
        free_graph(logits);
        return;
    }
    
    backward(loss);
    
    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-6 * (t1.tv_usec - t0.tv_usec);
    
    LOG_INFO("Test forward+backward time: %.3fs", elapsed);
    LOG_INFO("Test loss: %f", ((float*)loss->data->data)[0]);
    
    free_tensor(src);
    free_tensor(tgt);
    free_graph(loss);
}

Dataset* load_training_data(const Config* config) {
    Dataset* dataset = NULL;
    
    if (config->finetune_dir[0] != '\0') {
        LOG_INFO("Loading finetuning dataset from %s", config->finetune_dir);
        dataset = load_finetune_dataset(config->finetune_dir);
    } else if (config->data_dir[0] != '\0') {
        LOG_INFO("Loading dataset from directory %s", config->data_dir);
        dataset = load_dataset_from_directory(config->data_dir, DATASET_TYPE_TEXT);
    } else if (config->data_path[0] != '\0') {
        LOG_INFO("Loading dataset from file %s", config->data_path);
        dataset = load_dataset_from_file(config->data_path);
    }
    
    if (!dataset) {
        LOG_ERR("Failed to load dataset");
        return NULL;
    }
    
    LOG_INFO("Loaded dataset with %d samples", get_dataset_size(dataset));
    shuffle_dataset(dataset);
    
    return dataset;
}

Tokenizer* create_and_build_tokenizer(const Config* config, Dataset* dataset) {
    Tokenizer* tokenizer = create_tokenizer(config->vocab_size);
    if (!tokenizer) {
        LOG_ERR("Failed to create tokenizer");
        return NULL;
    }
    
    // Load vocabulary from file if specified
    if (config->vocab_file[0] != '\0') {
        LOG_INFO("Loading vocabulary from %s", config->vocab_file);
        if (!load_vocab(tokenizer, config->vocab_file)) {
            LOG_WARN("Failed to load vocabulary file, building from data");
        } else {
            LOG_INFO("Loaded vocabulary with %d tokens", tokenizer->vocab_size);
            return tokenizer;
        }
    }
    
    // Build vocabulary from dataset
    LOG_INFO("Building vocabulary from dataset...");
    
    if (dataset->text_data) {
        for (int i = 0; i < dataset->text_data->num_lines; i++) {
            build_vocab_from_text(tokenizer, dataset->text_data->lines[i]);
        }
    }
    
    if (dataset->json_data) {
        for (int i = 0; i < dataset->json_data->num_samples; i++) {
            build_vocab_from_text(tokenizer, dataset->json_data->inputs[i]);
            build_vocab_from_text(tokenizer, dataset->json_data->outputs[i]);
        }
    }
    
    finalize_vocab(tokenizer);
    LOG_INFO("Built vocabulary with %d tokens", tokenizer->vocab_size);
    
    return tokenizer;
}

void run_training(const Config* config, Transformer* model, Dataset* dataset, Tokenizer* tokenizer) {
    LOG_INFO("Starting training...");
    
    // Create optimizer
    Optimizer* optimizer = create_optimizer(model, config->learning_rate);
    if (!optimizer) {
        LOG_ERR("Failed to create optimizer");
        return;
    }
    
    // Set optimizer type
    if (strcmp(config->optimizer_type, "sgd") == 0) {
        optimizer_set_type(optimizer, OPTIMIZER_SGD);
    } else {
        optimizer_set_type(optimizer, OPTIMIZER_ADAM);
    }
    
    // Create training arena
    Arena* training_arena = create_arena(config->arena_size_mb * 1024 * 1024);
    if (!training_arena) {
        LOG_ERR("Failed to create training arena");
        free_optimizer(optimizer);
        return;
    }
    
    int dataset_size = get_dataset_size(dataset);
    int total_batches = (dataset_size + config->batch_size - 1) / config->batch_size;
    
    LOG_INFO("Training for %d epochs, %d batches per epoch", config->epochs, total_batches);
    
    for (int epoch = 0; epoch < config->epochs && !should_stop; epoch++) {
        LOG_INFO("Epoch %d/%d", epoch + 1, config->epochs);
        
        float epoch_loss = 0.0f;
        int batch_count = 0;
        
        for (int batch_start = 0; batch_start < dataset_size && !should_stop; 
             batch_start += config->batch_size) {
            
            // Create batch
            Batch* batch = create_batch(dataset, batch_start, config->batch_size, 
                                       config->seq_len, config->vocab_size);
            if (!batch) {
                LOG_ERR("Failed to create batch");
                continue;
            }
            
            // Create input tensors
            Tensor* src_in = create_tensor(2, (int[]){batch->actual_batch_size, config->seq_len}, TENSOR_TYPE_INT);
            Tensor* tgt_in = create_tensor(2, (int[]){batch->actual_batch_size, config->seq_len}, TENSOR_TYPE_INT);
            
            if (!src_in || !tgt_in) {
                LOG_ERR("Failed to allocate input tensors");
                free_batch(batch);
                continue;
            }
            
            // Copy batch data
            memcpy(src_in->data, batch->input_ids, batch->actual_batch_size * config->seq_len * sizeof(int));
            memcpy(tgt_in->data, batch->target_ids, batch->actual_batch_size * config->seq_len * sizeof(int));
            
            // Forward pass
            Value* logits = transformer_forward_ad(training_arena, src_in, tgt_in, model, 1);
            if (!logits) {
                LOG_ERR("Forward pass failed");
                free_tensor(src_in);
                free_tensor(tgt_in);
                free_batch(batch);
                continue;
            }
            
            // Compute loss
            Value* loss = cross_entropy_loss_ad(training_arena, logits, tgt_in);
            if (!loss) {
                LOG_ERR("Loss computation failed");
                free_tensor(src_in);
                free_tensor(tgt_in);
                free_batch(batch);
                free_graph(logits);
                continue;
            }
            
            float current_loss = ((float*)loss->data->data)[0];
            epoch_loss += current_loss;
            
            // Backward pass
            backward(loss);
            
            // Optimizer step
            if ((batch_count + 1) % config->gradient_accumulation_steps == 0) {
                optimizer_step(optimizer);
                zero_grad(optimizer);
            }
            
            if (config->verbose || batch_count % 100 == 0) {
                LOG_INFO("Epoch %d, Batch %d/%d, Loss: %.4f", 
                        epoch + 1, batch_count + 1, total_batches, current_loss);
            }
            
            // Cleanup
            free_tensor(src_in);
            free_tensor(tgt_in);
            free_batch(batch);
            free_graph(loss);
            
            // Reset arena for next batch
            arena_reset(training_arena);
            batch_count++;
        }
        
        float avg_loss = epoch_loss / batch_count;
        LOG_INFO("Epoch %d completed, average loss: %.4f", epoch + 1, avg_loss);
        
        // Save checkpoint if specified
        if (config->save_path[0] != '\0' && (epoch + 1) % 5 == 0) {
            char checkpoint_path[1024];
            snprintf(checkpoint_path, sizeof(checkpoint_path), "%s.epoch%d", config->save_path, epoch + 1);
            if (save_transformer(model, checkpoint_path)) {
                LOG_INFO("Saved checkpoint to %s", checkpoint_path);
            } else {
                LOG_ERR("Failed to save checkpoint to %s", checkpoint_path);
            }
        }
    }
    
    // Save final model
    if (config->save_path[0] != '\0') {
        if (save_transformer(model, config->save_path)) {
            LOG_INFO("Saved final model to %s", config->save_path);
        } else {
            LOG_ERR("Failed to save final model to %s", config->save_path);
        }
    }
    
    // Cleanup
    destroy_arena(training_arena);
    free_optimizer(optimizer);
}

void run_inference(const Config* config, Transformer* model, Tokenizer* tokenizer) {
    LOG_INFO("Starting inference...");
    
    // Create inference context
    InferenceContext* ctx = create_inference_context(model, tokenizer);
    if (!ctx) {
        LOG_ERR("Failed to create inference context");
        return;
    }
    
    // Set generation parameters
    ctx->config.max_length = config->max_length;
    ctx->config.temperature = config->temperature;
    ctx->config.top_k = config->top_k;
    
    if (config->verbose) {
        print_generation_config(&ctx->config);
    }
    
    if (config->interactive_mode) {
        // Interactive mode
        char prompt[1024];
        printf("Enter prompts (type 'quit' to exit):\n");
        
        while (1) {
            printf("> ");
            if (!fgets(prompt, sizeof(prompt), stdin)) break;
            
            // Remove newline
            prompt[strcspn(prompt, "\n")] = '\0';
            
            if (strcmp(prompt, "quit") == 0) break;
            if (strlen(prompt) == 0) continue;
            
            char* generated = generate_text_interactive(ctx, prompt);
            if (generated) {
                printf("\nGenerated: %s\n\n", generated);
                free(generated);
            }
        }
    } else {
        // Single generation
        const char* prompt_text = config->prompt[0] != '\0' ? config->prompt : "Hello world";
        char* generated = generate_text(ctx, prompt_text);
        
        if (generated) {
            printf("Prompt: %s\n", prompt_text);
            printf("Generated: %s\n", generated);
            
            if (config->verbose) {
                log_inference_stats(ctx, prompt_text, generated);
            }
            
            free(generated);
        } else {
            LOG_ERR("Failed to generate text");
        }
    }
    
    free_inference_context(ctx);
}

int main(int argc, char** argv) {
    // Set up signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parse configuration
    Config config;
    set_default_config(&config);
    
    if (parse_arguments(argc, argv, &config)) {
        return 1;
    }
    
    if (!validate_config(&config)) {
        return 1;
    }
    
    // Set random seed
    srand(config.seed);
    
    // Set up logging
    if (config.verbose) {
        LOG_INFO("Starting transformer training/inference");
        LOG_INFO("Configuration: %d layers, %d dim, %d heads, vocab_size=%d", 
                config.n_layers, config.embed_dim, config.n_heads, config.vocab_size);
    }
    
    // Create model
    Transformer* model = create_transformer(config.n_layers, config.vocab_size, 
                                           config.max_seq_len, config.embed_dim, 
                                           config.n_heads, config.ff_hidden_dim);
    if (!model) {
        LOG_ERR("Failed to create transformer model");
        return 1;
    }
    
    LOG_INFO("Created transformer model with %ld parameters", get_transformer_param_count(model));
    
    // Load model if specified
    if (config.load_path[0] != '\0') {
        LOG_INFO("Loading model from %s", config.load_path);
        if (!load_transformer(model, config.load_path)) {
            LOG_ERR("Failed to load model from %s", config.load_path);
            free_transformer(model);
            return 1;
        }
        LOG_INFO("Successfully loaded model");
    }
    
    // Run test mode
    if (config.test_mode) {
        test_transformer(model);
        free_transformer(model);
        return 0;
    }
    
    // Run inference mode
    if (config.inference_mode) {
        // Create a simple tokenizer for inference
        Tokenizer* tokenizer = create_tokenizer(config.vocab_size);
        if (!tokenizer) {
            LOG_ERR("Failed to create tokenizer for inference");
            free_transformer(model);
            return 1;
        }
        
        run_inference(&config, model, tokenizer);
        
        free_tokenizer(tokenizer);
        free_transformer(model);
        return 0;
    }
    
    // Training mode
    Dataset* dataset = load_training_data(&config);
    if (!dataset) {
        free_transformer(model);
        return 1;
    }
    
    Tokenizer* tokenizer = create_and_build_tokenizer(&config, dataset);
    if (!tokenizer) {
        free_dataset(dataset);
        free_transformer(model);
        return 1;
    }
    
    // Run training
    run_training(&config, model, dataset, tokenizer);
    
    // Cleanup
    free_tokenizer(tokenizer);
    free_dataset(dataset);
    free_transformer(model);
    
    LOG_INFO("Completed successfully");
    return 0;
}
