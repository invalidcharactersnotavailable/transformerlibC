#include "inference.h"
#include "autodiff.h"
#include "loss.h"
#include <sys/time.h>

// Default generation configuration
void set_default_generation_config(GenerationConfig* config) {
    if (!config) return;
    
    config->temperature = 1.0f;
    config->top_k = 40;
    config->top_p = 0.9f;
    config->max_length = 100;
    config->min_length = 1;
    config->repetition_penalty = 1.0f;
    config->do_sample = 1;
    config->num_beams = 1;
    config->pad_token_id = 0;
    config->eos_token_id = 3;
    config->bos_token_id = 2;
}

void print_generation_config(const GenerationConfig* config) {
    if (!config) return;
    
    printf("Generation Config:\n");
    printf("  Temperature: %.2f\n", config->temperature);
    printf("  Top-k: %d\n", config->top_k);
    printf("  Top-p: %.2f\n", config->top_p);
    printf("  Max length: %d\n", config->max_length);
    printf("  Min length: %d\n", config->min_length);
    printf("  Repetition penalty: %.2f\n", config->repetition_penalty);
    printf("  Do sample: %s\n", config->do_sample ? "yes" : "no");
    printf("  Num beams: %d\n", config->num_beams);
    printf("  Special tokens: PAD=%d, BOS=%d, EOS=%d\n", 
           config->pad_token_id, config->bos_token_id, config->eos_token_id);
}

Arena* create_inference_arena(size_t size_mb) {
    size_t size_bytes = size_mb * 1024 * 1024;
    return create_arena(size_bytes);
}

InferenceContext* create_inference_context(Transformer* model, Tokenizer* tokenizer) {
    if (!model || !tokenizer) return NULL;
    
    InferenceContext* ctx = malloc(sizeof(InferenceContext));
    if (!ctx) return NULL;
    
    ctx->model = model;
    ctx->tokenizer = tokenizer;
    set_default_generation_config(&ctx->config);
    
    // Create inference arena (default 1GB)
    ctx->inference_arena = create_inference_arena(1024);
    if (!ctx->inference_arena) {
        free(ctx);
        return NULL;
    }
    
    // Allocate token buffer
    ctx->max_generated = ctx->config.max_length;
    ctx->generated_tokens = malloc(ctx->max_generated * sizeof(int));
    ctx->num_generated = 0;
    
    if (!ctx->generated_tokens) {
        destroy_arena(ctx->inference_arena);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

void free_inference_context(InferenceContext* ctx) {
    if (!ctx) return;
    
    if (ctx->inference_arena) {
        destroy_arena(ctx->inference_arena);
    }
    if (ctx->generated_tokens) {
        free(ctx->generated_tokens);
    }
    free(ctx);
}

void reset_inference_arena(InferenceContext* ctx) {
    if (ctx && ctx->inference_arena) {
        arena_reset(ctx->inference_arena);
    }
}

Transformer* load_model_for_inference(const char* model_path) {
    if (!model_path) return NULL;
    
    // Create a minimal transformer for loading
    Transformer* model = create_transformer(2, 10000, 512, 512, 8, 2048);
    if (!model) return NULL;
    
    // Load the model weights
    if (!load_transformer(model, model_path)) {
        free_transformer(model);
        return NULL;
    }
    
    return model;
}

int validate_model_for_inference(Transformer* model) {
    if (!model) return 0;
    
    // Check basic model parameters
    if (model->n_layers <= 0 || model->embed_dim <= 0 || 
        model->n_heads <= 0 || model->ff_hidden_dim <= 0 ||
        model->vocab_size <= 0 || model->max_seq_len <= 0) {
        return 0;
    }
    
    // Check that embed_dim is divisible by n_heads
    if (model->embed_dim % model->n_heads != 0) {
        return 0;
    }
    
    return 1;
}

int check_model_compatibility(Transformer* model, Tokenizer* tokenizer) {
    if (!model || !tokenizer) return 0;
    
    // Check vocabulary size compatibility
    if (tokenizer->vocab_size > model->vocab_size) {
        return 0;
    }
    
    return 1;
}

float* compute_attention_mask(int seq_len, int max_seq_len) {
    float* mask = malloc(max_seq_len * max_seq_len * sizeof(float));
    if (!mask) return NULL;
    
    // Create causal mask (lower triangular)
    for (int i = 0; i < max_seq_len; i++) {
        for (int j = 0; j < max_seq_len; j++) {
            if (i < seq_len && j < seq_len && j <= i) {
                mask[i * max_seq_len + j] = 0.0f;  // Allow attention
            } else {
                mask[i * max_seq_len + j] = -1e9f; // Mask attention
            }
        }
    }
    
    return mask;
}

int* create_position_ids(int seq_len) {
    int* position_ids = malloc(seq_len * sizeof(int));
    if (!position_ids) return NULL;
    
    for (int i = 0; i < seq_len; i++) {
        position_ids[i] = i;
    }
    
    return position_ids;
}

int greedy_next_token(float* logits, int vocab_size) {
    if (!logits || vocab_size <= 0) return 0;
    
    int best_token = 0;
    float best_score = logits[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_token = i;
        }
    }
    
    return best_token;
}

int top_k_sampling(float* logits, int vocab_size, int top_k, float temperature) {
    if (!logits || vocab_size <= 0 || top_k <= 0) return 0;
    
    // Create a copy of logits for sorting
    float* logits_copy = malloc(vocab_size * sizeof(float));
    if (!logits_copy) return 0;
    
    memcpy(logits_copy, logits, vocab_size * sizeof(float));
    
    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits_copy[i] /= temperature;
    }
    
    // Find top-k indices
    int* indices = malloc(vocab_size * sizeof(int));
    if (!indices) {
        free(logits_copy);
        return 0;
    }
    
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    // Sort by logits (descending)
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits_copy[indices[i]] < logits_copy[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Sample from top-k
    int selected_idx = rand() % top_k;
    int selected_token = indices[selected_idx];
    
    free(logits_copy);
    free(indices);
    
    return selected_token;
}

int top_p_sampling(float* logits, int vocab_size, float top_p, float temperature) {
    if (!logits || vocab_size <= 0 || top_p <= 0.0f || top_p > 1.0f) return 0;
    
    // Apply temperature and softmax
    float* probs = malloc(vocab_size * sizeof(float));
    if (!probs) return 0;
    
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
    }
    
    // Sort by probability (descending)
    int* indices = malloc(vocab_size * sizeof(int));
    if (!indices) {
        free(probs);
        return 0;
    }
    
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs[indices[i]] < probs[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Find cumulative probability threshold
    float cumulative_prob = 0.0f;
    int cutoff_idx = vocab_size - 1;
    
    for (int i = 0; i < vocab_size; i++) {
        cumulative_prob += probs[indices[i]];
        if (cumulative_prob >= top_p) {
            cutoff_idx = i;
            break;
        }
    }
    
    // Sample from top-p tokens
    int selected_idx = rand() % (cutoff_idx + 1);
    int selected_token = indices[selected_idx];
    
    free(probs);
    free(indices);
    
    return selected_token;
}

int sample_next_token(InferenceContext* ctx, float* logits, int vocab_size) {
    if (!ctx || !logits || vocab_size <= 0) return 0;
    
    if (!ctx->config.do_sample) {
        return greedy_next_token(logits, vocab_size);
    }
    
    // Apply repetition penalty if needed
    if (ctx->config.repetition_penalty != 1.0f && ctx->num_generated > 0) {
        for (int i = 0; i < ctx->num_generated; i++) {
            int token_id = ctx->generated_tokens[i];
            if (token_id >= 0 && token_id < vocab_size) {
                logits[token_id] /= ctx->config.repetition_penalty;
            }
        }
    }
    
    // Choose sampling method
    if (ctx->config.top_p > 0.0f && ctx->config.top_p < 1.0f) {
        return top_p_sampling(logits, vocab_size, ctx->config.top_p, ctx->config.temperature);
    } else if (ctx->config.top_k > 0 && ctx->config.top_k < vocab_size) {
        return top_k_sampling(logits, vocab_size, ctx->config.top_k, ctx->config.temperature);
    } else {
        return greedy_next_token(logits, vocab_size);
    }
}

int generate_tokens(InferenceContext* ctx, const char* prompt, int* output_tokens, int max_output_tokens) {
    if (!ctx || !output_tokens || max_output_tokens <= 0) return 0;
    
    reset_inference_arena(ctx);
    
    // Encode prompt
    int prompt_tokens[ctx->model->max_seq_len];
    int prompt_length = encode_text_with_special_tokens(ctx->tokenizer, prompt, prompt_tokens, ctx->model->max_seq_len);
    
    if (prompt_length <= 0) return 0;
    
    // Initialize generation
    ctx->num_generated = 0;
    int total_length = prompt_length;
    
    // Copy prompt tokens to output
    for (int i = 0; i < prompt_length && i < max_output_tokens; i++) {
        output_tokens[i] = prompt_tokens[i];
    }
    
    // Generate tokens
    while (total_length < ctx->config.max_length && ctx->num_generated < max_output_tokens) {
        // Create input tensor
        Tensor* input = create_tensor(2, (int[]){1, total_length}, TENSOR_TYPE_INT);
        if (!input) break;
        
        // Copy current sequence
        memcpy(input->data, output_tokens, total_length * sizeof(int));
        
        // Forward pass
        Value* logits_value = transformer_forward_ad(ctx->inference_arena, input, NULL, ctx->model, 0);
        if (!logits_value) {
            free_tensor(input);
            break;
        }
        
        // Get logits for the last position
        float* logits = (float*)logits_value->data->data;
        int vocab_size = ctx->model->vocab_size;
        float* last_logits = logits + (total_length - 1) * vocab_size;
        
        // Sample next token
        int next_token = sample_next_token(ctx, last_logits, vocab_size);
        
        // Add to output
        if (total_length < max_output_tokens) {
            output_tokens[total_length] = next_token;
            ctx->generated_tokens[ctx->num_generated++] = next_token;
            total_length++;
        }
        
        // Check for EOS
        if (next_token == ctx->config.eos_token_id) {
            break;
        }
        
        // Clean up
        free_tensor(input);
        free_graph(logits_value);
    }
    
    return total_length;
}

char* generate_text(InferenceContext* ctx, const char* prompt) {
    if (!ctx || !prompt) return NULL;
    
    int max_tokens = ctx->config.max_length;
    int* tokens = malloc(max_tokens * sizeof(int));
    if (!tokens) return NULL;
    
    int num_tokens = generate_tokens(ctx, prompt, tokens, max_tokens);
    if (num_tokens <= 0) {
        free(tokens);
        return NULL;
    }
    
    // Decode tokens to text
    char* result = decode_tokens(ctx->tokenizer, tokens, num_tokens);
    free(tokens);
    
    return result;
}

char* generate_text_interactive(InferenceContext* ctx, const char* prompt) {
    if (!ctx || !prompt) return NULL;
    
    printf("Generating text for prompt: '%s'\n", prompt);
    printf("Press Enter to continue generation, 'q' to quit\n");
    
    int max_tokens = ctx->config.max_length;
    int* tokens = malloc(max_tokens * sizeof(int));
    if (!tokens) return NULL;
    
    // Encode prompt
    int prompt_tokens[ctx->model->max_seq_len];
    int prompt_length = encode_text_with_special_tokens(ctx->tokenizer, prompt, prompt_tokens, ctx->model->max_seq_len);
    
    if (prompt_length <= 0) {
        free(tokens);
        return NULL;
    }
    
    // Copy prompt tokens
    int total_length = prompt_length;
    for (int i = 0; i < prompt_length; i++) {
        tokens[i] = prompt_tokens[i];
    }
    
    // Print initial prompt
    char* current_text = decode_tokens(ctx->tokenizer, tokens, total_length);
    printf("%s", current_text);
    fflush(stdout);
    free(current_text);
    
    // Interactive generation
    while (total_length < max_tokens) {
        // Generate next token
        reset_inference_arena(ctx);
        
        Tensor* input = create_tensor(2, (int[]){1, total_length}, TENSOR_TYPE_INT);
        if (!input) break;
        
        memcpy(input->data, tokens, total_length * sizeof(int));
        
        Value* logits_value = transformer_forward_ad(ctx->inference_arena, input, NULL, ctx->model, 0);
        if (!logits_value) {
            free_tensor(input);
            break;
        }
        
        float* logits = (float*)logits_value->data->data;
        int vocab_size = ctx->model->vocab_size;
        float* last_logits = logits + (total_length - 1) * vocab_size;
        
        int next_token = sample_next_token(ctx, last_logits, vocab_size);
        
        // Add token
        tokens[total_length++] = next_token;
        
        // Decode and print new token
        char* new_text = decode_tokens(ctx->tokenizer, tokens + total_length - 1, 1);
        printf("%s", new_text);
        fflush(stdout);
        free(new_text);
        
        // Check for EOS
        if (next_token == ctx->config.eos_token_id) {
            break;
        }
        
        // Check for user input
        if (total_length % 10 == 0) {  // Check every 10 tokens
            struct timeval tv;
            gettimeofday(&tv, NULL);
            
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(0, &fds);
            
            tv.tv_sec = 0;
            tv.tv_usec = 0;
            
            if (select(1, &fds, NULL, NULL, &tv) > 0) {
                char input_char;
                if (read(0, &input_char, 1) > 0) {
                    if (input_char == 'q' || input_char == 'Q') {
                        break;
                    }
                }
            }
        }
        
        free_tensor(input);
        free_graph(logits_value);
    }
    
    printf("\n");
    
    // Return full generated text
    char* result = decode_tokens(ctx->tokenizer, tokens, total_length);
    free(tokens);
    
    return result;
}

void log_inference_stats(const InferenceContext* ctx, const char* prompt, const char* generated) {
    if (!ctx) return;
    
    printf("=== Inference Stats ===\n");
    printf("Prompt length: %zu characters\n", strlen(prompt));
    printf("Generated length: %zu characters\n", strlen(generated));
    printf("Tokens generated: %d\n", ctx->num_generated);
    printf("Model: %d layers, %d dim, %d heads\n", 
           ctx->model->n_layers, ctx->model->embed_dim, ctx->model->n_heads);
    printf("Vocabulary size: %d\n", ctx->model->vocab_size);
    printf("=====================\n");
}

void print_token_distribution(float* logits, int vocab_size, int top_k) {
    if (!logits || vocab_size <= 0) return;
    
    printf("Top-%d token probabilities:\n", top_k);
    
    // Create index array
    int* indices = malloc(vocab_size * sizeof(int));
    if (!indices) return;
    
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    // Sort by logits (descending)
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[indices[i]] < logits[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Print top-k
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        printf("  [%d] %.4f\n", indices[i], logits[indices[i]]);
    }
    
    free(indices);
}