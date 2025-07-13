#ifndef INFERENCE_H
#define INFERENCE_H

#include "transformer.h"
#include "tokenizer.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Generation parameters
typedef struct {
    float temperature;      // Sampling temperature (0.1 to 2.0)
    int top_k;             // Top-k sampling (1 to vocab_size)
    int top_p;             // Top-p (nucleus) sampling (0.0 to 1.0)
    int max_length;        // Maximum generation length
    int min_length;        // Minimum generation length
    float repetition_penalty; // Penalty for repetition (1.0 to 2.0)
    int do_sample;         // Whether to use sampling vs greedy
    int num_beams;         // Number of beams for beam search
    int pad_token_id;      // Padding token ID
    int eos_token_id;      // End-of-sequence token ID
    int bos_token_id;      // Beginning-of-sequence token ID
} GenerationConfig;

// Inference context
typedef struct {
    Transformer* model;
    Tokenizer* tokenizer;
    GenerationConfig config;
    int* generated_tokens;
    int num_generated;
    int max_generated;
} InferenceContext;

// Inference functions
InferenceContext* create_inference_context(Transformer* model, Tokenizer* tokenizer);
void free_inference_context(InferenceContext* ctx);

// Generation functions
char* generate_text(InferenceContext* ctx, const char* prompt);
char* generate_text_interactive(InferenceContext* ctx, const char* prompt);
int generate_tokens(InferenceContext* ctx, const char* prompt, int* output_tokens, int max_output_tokens);

// Sampling functions
int sample_next_token(InferenceContext* ctx, float* logits, int vocab_size);
int greedy_next_token(float* logits, int vocab_size);
int top_k_sampling(float* logits, int vocab_size, int top_k, float temperature);
int top_p_sampling(float* logits, int vocab_size, float top_p, float temperature);

// Model loading and validation
Transformer* load_model_for_inference(const char* model_path);
int validate_model_for_inference(Transformer* model);
int check_model_compatibility(Transformer* model, Tokenizer* tokenizer);

// Utility functions
void set_default_generation_config(GenerationConfig* config);
void print_generation_config(const GenerationConfig* config);
float* compute_attention_mask(int seq_len, int max_seq_len);
int* create_position_ids(int seq_len);

// Memory management for inference
void reset_inference_context(InferenceContext* ctx);

// Logging and debugging
void log_inference_stats(const InferenceContext* ctx, const char* prompt, const char* generated);
void print_token_distribution(float* logits, int vocab_size, int top_k);

#endif // INFERENCE_H