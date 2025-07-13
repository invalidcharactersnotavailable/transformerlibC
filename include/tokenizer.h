#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_VOCAB_SIZE 50000
#define MAX_TOKEN_LEN 256
#define UNK_TOKEN_ID 1
#define PAD_TOKEN_ID 0
#define BOS_TOKEN_ID 2
#define EOS_TOKEN_ID 3

// Vocabulary entry
typedef struct {
    char* token;
    int frequency;
    int id;
} VocabEntry;

// Tokenizer structure
typedef struct {
    VocabEntry* vocab;
    int vocab_size;
    int max_vocab_size;
    char** id_to_token;
    int* token_to_id;
    int max_token_id;
} Tokenizer;

// Tokenizer functions
Tokenizer* create_tokenizer(int max_vocab_size);
void free_tokenizer(Tokenizer* tokenizer);

// Vocabulary building
int build_vocab_from_text(Tokenizer* tokenizer, const char* text);
int build_vocab_from_file(Tokenizer* tokenizer, const char* file_path);
int build_vocab_from_directory(Tokenizer* tokenizer, const char* dir_path);
void finalize_vocab(Tokenizer* tokenizer);

// Tokenization
int encode_text(Tokenizer* tokenizer, const char* text, int* tokens, int max_tokens);
int encode_text_with_special_tokens(Tokenizer* tokenizer, const char* text, int* tokens, int max_tokens);
char* decode_tokens(Tokenizer* tokenizer, const int* tokens, int num_tokens);

// Character-level tokenization (fallback)
int encode_char_level(const char* text, int* tokens, int max_tokens, int vocab_size);
char* decode_char_level(const int* tokens, int num_tokens);

// Vocabulary management
int add_token_to_vocab(Tokenizer* tokenizer, const char* token);
int get_token_id(Tokenizer* tokenizer, const char* token);
const char* get_token_by_id(Tokenizer* tokenizer, int id);

// Save/load vocabulary
int save_vocab(Tokenizer* tokenizer, const char* file_path);
int load_vocab(Tokenizer* tokenizer, const char* file_path);

// Utility functions
void normalize_text(char* text);
int is_special_token(const char* token);
int count_words(const char* text);

#endif // TOKENIZER_H 