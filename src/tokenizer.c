#include "tokenizer.h"
#include <dirent.h>
#include <sys/stat.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

// Define DT_REG if not available
#ifndef DT_REG
#define DT_REG 8
#endif

// Define strdup if not available (for older systems)
#ifndef _GNU_SOURCE
char* strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* dup = malloc(len);
    if (dup) {
        memcpy(dup, s, len);
    }
    return dup;
}
#endif

// Comparison function for qsort
static int compare_vocab_entries(const void* a, const void* b) {
    const VocabEntry* va = (const VocabEntry*)a;
    const VocabEntry* vb = (const VocabEntry*)b;
    
    // Sort by frequency (descending), then by token (ascending)
    if (vb->frequency != va->frequency) {
        return vb->frequency - va->frequency;
    }
    return strcmp(va->token, vb->token);
}



Tokenizer* create_tokenizer(int max_vocab_size) {
    Tokenizer* tokenizer = malloc(sizeof(Tokenizer));
    if (!tokenizer) return NULL;
    
    tokenizer->max_vocab_size = max_vocab_size > 0 ? max_vocab_size : MAX_VOCAB_SIZE;
    tokenizer->vocab_size = 0;
    tokenizer->max_token_id = 0;
    
    // Allocate vocabulary array
    tokenizer->vocab = malloc(tokenizer->max_vocab_size * sizeof(VocabEntry));
    if (!tokenizer->vocab) {
        free(tokenizer);
        return NULL;
    }
    
    // Initialize mapping arrays
    tokenizer->id_to_token = malloc(tokenizer->max_vocab_size * sizeof(char*));
    tokenizer->token_to_id = malloc(tokenizer->max_vocab_size * sizeof(int));
    
    if (!tokenizer->id_to_token || !tokenizer->token_to_id) {
        if (tokenizer->id_to_token) free(tokenizer->id_to_token);
        if (tokenizer->token_to_id) free(tokenizer->token_to_id);
        free(tokenizer->vocab);
        free(tokenizer);
        return NULL;
    }
    
    // Initialize all entries
    for (int i = 0; i < tokenizer->max_vocab_size; i++) {
        tokenizer->vocab[i].token = NULL;
        tokenizer->vocab[i].frequency = 0;
        tokenizer->vocab[i].id = -1;
        tokenizer->id_to_token[i] = NULL;
        tokenizer->token_to_id[i] = -1;
    }
    
    // Add special tokens
    add_token_to_vocab(tokenizer, "<PAD>");
    add_token_to_vocab(tokenizer, "<UNK>");
    add_token_to_vocab(tokenizer, "<BOS>");
    add_token_to_vocab(tokenizer, "<EOS>");
    
    return tokenizer;
}

void free_tokenizer(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    // Free vocabulary entries
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        if (tokenizer->vocab[i].token) {
            free(tokenizer->vocab[i].token);
        }
    }
    
    // Free mapping arrays
    for (int i = 0; i < tokenizer->max_vocab_size; i++) {
        if (tokenizer->id_to_token[i]) {
            free(tokenizer->id_to_token[i]);
        }
    }
    
    if (tokenizer->vocab) free(tokenizer->vocab);
    if (tokenizer->id_to_token) free(tokenizer->id_to_token);
    if (tokenizer->token_to_id) free(tokenizer->token_to_id);
    
    free(tokenizer);
}

int add_token_to_vocab(Tokenizer* tokenizer, const char* token) {
    if (!tokenizer || !token) return -1;
    
    // Check if token already exists
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        if (tokenizer->vocab[i].token && strcmp(tokenizer->vocab[i].token, token) == 0) {
            tokenizer->vocab[i].frequency++;
            return tokenizer->vocab[i].id;
        }
    }
    
    // Add new token if there's space
    if (tokenizer->vocab_size >= tokenizer->max_vocab_size) {
        return -1;
    }
    
    int id = tokenizer->vocab_size;
    tokenizer->vocab[id].token = strdup(token);
    tokenizer->vocab[id].frequency = 1;
    tokenizer->vocab[id].id = id;
    
    if (!tokenizer->vocab[id].token) {
        return -1;
    }
    
    tokenizer->vocab_size++;
    if (id > tokenizer->max_token_id) {
        tokenizer->max_token_id = id;
    }
    
    return id;
}

int get_token_id(Tokenizer* tokenizer, const char* token) {
    if (!tokenizer || !token) return UNK_TOKEN_ID;
    
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        if (tokenizer->vocab[i].token && strcmp(tokenizer->vocab[i].token, token) == 0) {
            return tokenizer->vocab[i].id;
        }
    }
    
    return UNK_TOKEN_ID;
}

const char* get_token_by_id(Tokenizer* tokenizer, int id) {
    if (!tokenizer || id < 0 || id >= tokenizer->vocab_size) {
        return "<UNK>";
    }
    
    return tokenizer->vocab[id].token;
}

void normalize_text(char* text) {
    if (!text) return;
    
    // Convert to lowercase and normalize whitespace
    int j = 0;
    int prev_space = 1;
    
    for (int i = 0; text[i]; i++) {
        char c = text[i];
        
        if (isspace(c)) {
            if (!prev_space) {
                text[j++] = ' ';
                prev_space = 1;
            }
        } else {
            text[j++] = tolower(c);
            prev_space = 0;
        }
    }
    
    // Remove trailing space
    if (j > 0 && text[j-1] == ' ') {
        j--;
    }
    
    text[j] = '\0';
}

int is_special_token(const char* token) {
    if (!token) return 0;
    return (strcmp(token, "<PAD>") == 0 || strcmp(token, "<UNK>") == 0 ||
            strcmp(token, "<BOS>") == 0 || strcmp(token, "<EOS>") == 0);
}

int count_words(const char* text) {
    if (!text) return 0;
    
    int count = 0;
    int in_word = 0;
    
    for (int i = 0; text[i]; i++) {
        if (isspace(text[i])) {
            in_word = 0;
        } else if (!in_word) {
            count++;
            in_word = 1;
        }
    }
    
    return count;
}

int build_vocab_from_text(Tokenizer* tokenizer, const char* text) {
    if (!tokenizer || !text) return 0;
    
    char* text_copy = strdup(text);
    if (!text_copy) return 0;
    
    normalize_text(text_copy);
    
    // Simple word-based tokenization
    char* word = strtok(text_copy, " \t\n\r.,!?;:()[]{}\"'");
    while (word && tokenizer->vocab_size < tokenizer->max_vocab_size) {
        if (strlen(word) > 0 && !is_special_token(word)) {
            add_token_to_vocab(tokenizer, word);
        }
        word = strtok(NULL, " \t\n\r.,!?;:()[]{}\"'");
    }
    
    free(text_copy);
    return 1;
}

int build_vocab_from_file(Tokenizer* tokenizer, const char* file_path) {
    FILE* file = fopen(file_path, "r");
    if (!file) return 0;
    
    char line[4096];
    int success = 1;
    
    while (fgets(line, sizeof(line), file) && success) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        
        if (len > 1) {
            success = build_vocab_from_text(tokenizer, line);
        }
    }
    
    fclose(file);
    return success;
}

int build_vocab_from_directory(Tokenizer* tokenizer, const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (!dir) return 0;
    
    struct dirent* entry;
    int success = 1;
    
    while ((entry = readdir(dir)) != NULL && success) {
        if (entry->d_type == DT_REG) {
            char full_path[1024];
            snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
            
            // Only process text files
            if (strstr(entry->d_name, ".txt") || strstr(entry->d_name, ".md") ||
                strstr(entry->d_name, ".py") || strstr(entry->d_name, ".c") ||
                strstr(entry->d_name, ".cpp") || strstr(entry->d_name, ".h")) {
                success = build_vocab_from_file(tokenizer, full_path);
            }
        }
    }
    
    closedir(dir);
    return success;
}

void finalize_vocab(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    // Sort vocabulary by frequency
    qsort(tokenizer->vocab, tokenizer->vocab_size, sizeof(VocabEntry), compare_vocab_entries);
    
    // Rebuild ID mappings
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        tokenizer->vocab[i].id = i;
        tokenizer->id_to_token[i] = strdup(tokenizer->vocab[i].token);
        tokenizer->token_to_id[i] = i;
    }
    
    tokenizer->max_token_id = tokenizer->vocab_size - 1;
}

int encode_text(Tokenizer* tokenizer, const char* text, int* tokens, int max_tokens) {
    if (!tokenizer || !text || !tokens || max_tokens <= 0) return 0;
    
    char* text_copy = strdup(text);
    if (!text_copy) return 0;
    
    normalize_text(text_copy);
    
    int token_count = 0;
    char* word = strtok(text_copy, " \t\n\r.,!?;:()[]{}\"'");
    
    while (word && token_count < max_tokens) {
        int token_id = get_token_id(tokenizer, word);
        tokens[token_count++] = token_id;
        word = strtok(NULL, " \t\n\r.,!?;:()[]{}\"'");
    }
    
    free(text_copy);
    return token_count;
}

int encode_text_with_special_tokens(Tokenizer* tokenizer, const char* text, int* tokens, int max_tokens) {
    if (!tokenizer || !text || !tokens || max_tokens <= 2) return 0;
    
    // Add BOS token
    tokens[0] = BOS_TOKEN_ID;
    
    // Encode the text
    int text_tokens = encode_text(tokenizer, text, tokens + 1, max_tokens - 2);
    
    // Add EOS token
    tokens[text_tokens + 1] = EOS_TOKEN_ID;
    
    return text_tokens + 2;
}

char* decode_tokens(Tokenizer* tokenizer, const int* tokens, int num_tokens) {
    if (!tokenizer || !tokens || num_tokens <= 0) return NULL;
    
    // Calculate total length needed
    int total_length = 0;
    for (int i = 0; i < num_tokens; i++) {
        const char* token = get_token_by_id(tokenizer, tokens[i]);
        if (token) {
            total_length += strlen(token) + 1; // +1 for space
        }
    }
    
    char* result = malloc(total_length + 1);
    if (!result) return NULL;
    
    result[0] = '\0';
    int pos = 0;
    
    for (int i = 0; i < num_tokens; i++) {
        const char* token = get_token_by_id(tokenizer, tokens[i]);
        if (token && !is_special_token(token)) {
            int len = strlen(token);
            strcpy(result + pos, token);
            pos += len;
            result[pos++] = ' ';
        }
    }
    
    // Remove trailing space
    if (pos > 0 && result[pos-1] == ' ') {
        pos--;
    }
    result[pos] = '\0';
    
    return result;
}

// Character-level tokenization (fallback)
int encode_char_level(const char* text, int* tokens, int max_tokens, int vocab_size) {
    if (!text || !tokens || max_tokens <= 0) return 0;
    
    int count = 0;
    for (int i = 0; text[i] && count < max_tokens; i++) {
        tokens[count++] = (int)text[i] % vocab_size;
    }
    
    return count;
}

char* decode_char_level(const int* tokens, int num_tokens) {
    if (!tokens || num_tokens <= 0) return NULL;
    
    char* result = malloc(num_tokens + 1);
    if (!result) return NULL;
    
    for (int i = 0; i < num_tokens; i++) {
        result[i] = (char)tokens[i];
    }
    result[num_tokens] = '\0';
    
    return result;
}

int save_vocab(Tokenizer* tokenizer, const char* file_path) {
    if (!tokenizer || !file_path) return 0;
    
    FILE* file = fopen(file_path, "w");
    if (!file) return 0;
    
    // Write vocabulary size
    fprintf(file, "%d\n", tokenizer->vocab_size);
    
    // Write each token and its frequency
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        fprintf(file, "%s\t%d\n", tokenizer->vocab[i].token, tokenizer->vocab[i].frequency);
    }
    
    fclose(file);
    return 1;
}

int load_vocab(Tokenizer* tokenizer, const char* file_path) {
    if (!tokenizer || !file_path) return 0;
    
    FILE* file = fopen(file_path, "r");
    if (!file) return 0;
    
    // Read vocabulary size
    int vocab_size;
    if (fscanf(file, "%d\n", &vocab_size) != 1) {
        fclose(file);
        return 0;
    }
    
    // Clear existing vocabulary
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        if (tokenizer->vocab[i].token) {
            free(tokenizer->vocab[i].token);
        }
    }
    tokenizer->vocab_size = 0;
    
    // Read vocabulary entries
    char token[256];
    int frequency;
    
    for (int i = 0; i < vocab_size && i < tokenizer->max_vocab_size; i++) {
        if (fscanf(file, "%255s\t%d\n", token, &frequency) == 2) {
            tokenizer->vocab[i].token = strdup(token);
            tokenizer->vocab[i].frequency = frequency;
            tokenizer->vocab[i].id = i;
            tokenizer->vocab_size++;
        }
    }
    
    fclose(file);
    
    // Rebuild mappings
    finalize_vocab(tokenizer);
    
    return 1;
}

// Legacy function for backward compatibility
int encode(const char* text, int* tokens, int max_tokens) {
    // Create a simple character-level tokenizer
    return encode_char_level(text, tokens, max_tokens, 10000);
}

char* decode(const int* tokens, int num_tokens) {
    return decode_char_level(tokens, num_tokens);
}
