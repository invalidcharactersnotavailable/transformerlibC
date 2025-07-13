#define _GNU_SOURCE
#include "dataset.h"
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

// Define DT_REG if not available
#ifndef DT_REG
#define DT_REG 8
#endif

// Utility functions
int is_text_file(const char* filename) {
    if (!filename) return 0;
    const char* ext = strrchr(filename, '.');
    if (!ext) return 1; // Assume text if no extension
    ext++; // Skip the dot
    return (strcmp(ext, "txt") == 0 || strcmp(ext, "md") == 0 || 
            strcmp(ext, "py") == 0 || strcmp(ext, "c") == 0 || 
            strcmp(ext, "cpp") == 0 || strcmp(ext, "h") == 0 ||
            strcmp(ext, "js") == 0 || strcmp(ext, "html") == 0 ||
            strcmp(ext, "css") == 0 || strcmp(ext, "json") == 0);
}

int is_json_file(const char* filename) {
    if (!filename) return 0;
    const char* ext = strrchr(filename, '.');
    return ext && strcmp(ext, ".json") == 0;
}

char* read_file_content(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* content = malloc(file_size + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }
    
    size_t bytes_read = fread(content, 1, file_size, file);
    content[bytes_read] = '\0';
    fclose(file);
    return content;
}

int count_lines_in_file(const char* file_path) {
    FILE* file = fopen(file_path, "r");
    if (!file) return 0;
    
    int count = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') count++;
    }
    fclose(file);
    return count;
}

int count_files_in_directory(const char* dir_path, const char* extension) {
    DIR* dir = opendir(dir_path);
    if (!dir) return 0;
    
    int count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) { // Regular file
            if (!extension || strstr(entry->d_name, extension)) {
                count++;
            }
        }
    }
    closedir(dir);
    return count;
}

// Text dataset functions
TextDataset* create_text_dataset(int initial_capacity) {
    TextDataset* ds = malloc(sizeof(TextDataset));
    if (!ds) return NULL;
    
    ds->capacity = initial_capacity > 0 ? initial_capacity : 1000;
    ds->lines = malloc(ds->capacity * sizeof(char*));
    ds->num_lines = 0;
    ds->source_file = NULL;
    
    if (!ds->lines) {
        free(ds);
        return NULL;
    }
    
    return ds;
}

void free_text_dataset(TextDataset* ds) {
    if (!ds) return;
    
    for (int i = 0; i < ds->num_lines; i++) {
        if (ds->lines[i]) free(ds->lines[i]);
    }
    if (ds->lines) free(ds->lines);
    if (ds->source_file) free(ds->source_file);
    free(ds);
}

int add_line_to_text_dataset(TextDataset* ds, const char* line) {
    if (!ds || !line) return 0;
    
    if (ds->num_lines >= ds->capacity) {
        int new_capacity = ds->capacity * 2;
        char** new_lines = realloc(ds->lines, new_capacity * sizeof(char*));
        if (!new_lines) return 0;
        ds->lines = new_lines;
        ds->capacity = new_capacity;
    }
    
    ds->lines[ds->num_lines] = strdup(line);
    if (!ds->lines[ds->num_lines]) return 0;
    
    ds->num_lines++;
    return 1;
}

TextDataset* load_text_dataset_from_file(const char* file_path) {
    FILE* file = fopen(file_path, "r");
    if (!file) return NULL;
    
    TextDataset* ds = create_text_dataset(1000);
    if (!ds) {
        fclose(file);
        return NULL;
    }
    
    ds->source_file = strdup(file_path);
    char line[MAX_LINE_LEN];
    
    while (fgets(line, sizeof(line), file) && ds->num_lines < MAX_DATASET_SIZE) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (len > 1) { // Skip empty lines
            if (!add_line_to_text_dataset(ds, line)) {
                free_text_dataset(ds);
                fclose(file);
                return NULL;
            }
        }
    }
    
    fclose(file);
    return ds;
}

// JSON dataset functions
JsonDataset* create_json_dataset(int initial_capacity) {
    JsonDataset* ds = malloc(sizeof(JsonDataset));
    if (!ds) return NULL;
    
    ds->capacity = initial_capacity > 0 ? initial_capacity : 1000;
    ds->inputs = malloc(ds->capacity * sizeof(char*));
    ds->outputs = malloc(ds->capacity * sizeof(char*));
    ds->num_samples = 0;
    ds->source_file = NULL;
    
    if (!ds->inputs || !ds->outputs) {
        if (ds->inputs) free(ds->inputs);
        if (ds->outputs) free(ds->outputs);
        free(ds);
        return NULL;
    }
    
    return ds;
}

void free_json_dataset(JsonDataset* ds) {
    if (!ds) return;
    
    for (int i = 0; i < ds->num_samples; i++) {
        if (ds->inputs[i]) free(ds->inputs[i]);
        if (ds->outputs[i]) free(ds->outputs[i]);
    }
    if (ds->inputs) free(ds->inputs);
    if (ds->outputs) free(ds->outputs);
    if (ds->source_file) free(ds->source_file);
    free(ds);
}

int add_sample_to_json_dataset(JsonDataset* ds, const char* input, const char* output) {
    if (!ds || !input || !output) return 0;
    
    if (ds->num_samples >= ds->capacity) {
        int new_capacity = ds->capacity * 2;
        char** new_inputs = realloc(ds->inputs, new_capacity * sizeof(char*));
        char** new_outputs = realloc(ds->outputs, new_capacity * sizeof(char*));
        if (!new_inputs || !new_outputs) {
            if (new_inputs) free(new_inputs);
            if (new_outputs) free(new_outputs);
            return 0;
        }
        ds->inputs = new_inputs;
        ds->outputs = new_outputs;
        ds->capacity = new_capacity;
    }
    
    ds->inputs[ds->num_samples] = strdup(input);
    ds->outputs[ds->num_samples] = strdup(output);
    
    if (!ds->inputs[ds->num_samples] || !ds->outputs[ds->num_samples]) {
        if (ds->inputs[ds->num_samples]) free(ds->inputs[ds->num_samples]);
        if (ds->outputs[ds->num_samples]) free(ds->outputs[ds->num_samples]);
        return 0;
    }
    
    ds->num_samples++;
    return 1;
}

JsonDataset* parse_json_dataset(const char* json_content, const char* source_file) {
    if (!json_content) return NULL;
    
    JsonDataset* ds = create_json_dataset(1000);
    if (!ds) return NULL;
    
    ds->source_file = source_file ? strdup(source_file) : NULL;
    
    // Simple JSON parsing for {"input": "...", "output": "..."} format
    char* content_copy = strdup(json_content);
    if (!content_copy) {
        free_json_dataset(ds);
        return NULL;
    }
    
    char* line = strtok(content_copy, "\n");
    while (line && ds->num_samples < MAX_DATASET_SIZE) {
        // Skip empty lines and comments
        while (*line && isspace(*line)) line++;
        if (*line == '\0' || *line == '#') {
            line = strtok(NULL, "\n");
            continue;
        }
        
        // Look for input and output fields
        char* input_start = strstr(line, "\"input\"");
        char* output_start = strstr(line, "\"output\"");
        
        if (input_start && output_start) {
            // Extract input
            char* input_quote = strchr(input_start + 7, '"');
            char* input_end = strchr(input_quote + 1, '"');
            if (input_quote && input_end) {
                *input_end = '\0';
                char* input = input_quote + 1;
                
                // Extract output
                char* output_quote = strchr(output_start + 8, '"');
                char* output_end = strchr(output_quote + 1, '"');
                if (output_quote && output_end) {
                    *output_end = '\0';
                    char* output = output_quote + 1;
                    
                    if (!add_sample_to_json_dataset(ds, input, output)) {
                        free(content_copy);
                        free_json_dataset(ds);
                        return NULL;
                    }
                }
            }
        }
        
        line = strtok(NULL, "\n");
    }
    
    free(content_copy);
    return ds;
}

int validate_json_format(const char* json_content) {
    if (!json_content) return 0;
    
    // Simple validation - check for basic JSON structure
    int brace_count = 0;
    int quote_count = 0;
    int has_input = 0;
    int has_output = 0;
    
    for (int i = 0; json_content[i]; i++) {
        char c = json_content[i];
        if (c == '{') brace_count++;
        else if (c == '}') brace_count--;
        else if (c == '"') quote_count++;
        
        // Check for input/output fields
        if (strncmp(&json_content[i], "\"input\"", 7) == 0) has_input = 1;
        if (strncmp(&json_content[i], "\"output\"", 8) == 0) has_output = 1;
    }
    
    return brace_count == 0 && quote_count % 2 == 0 && has_input && has_output;
}

// Combined dataset functions
Dataset* create_dataset(DatasetType type) {
    Dataset* ds = malloc(sizeof(Dataset));
    if (!ds) return NULL;
    
    ds->type = type;
    ds->text_data = NULL;
    ds->json_data = NULL;
    ds->total_samples = 0;
    ds->file_paths = NULL;
    ds->num_files = 0;
    
    if (type == DATASET_TYPE_TEXT || type == DATASET_TYPE_MIXED) {
        ds->text_data = create_text_dataset(1000);
        if (!ds->text_data) {
            free(ds);
            return NULL;
        }
    }
    
    if (type == DATASET_TYPE_JSON || type == DATASET_TYPE_MIXED) {
        ds->json_data = create_json_dataset(1000);
        if (!ds->json_data) {
            if (ds->text_data) free_text_dataset(ds->text_data);
            free(ds);
            return NULL;
        }
    }
    
    return ds;
}

void free_dataset(Dataset* dataset) {
    if (!dataset) return;
    
    if (dataset->text_data) free_text_dataset(dataset->text_data);
    if (dataset->json_data) free_json_dataset(dataset->json_data);
    
    for (int i = 0; i < dataset->num_files; i++) {
        if (dataset->file_paths[i]) free(dataset->file_paths[i]);
    }
    if (dataset->file_paths) free(dataset->file_paths);
    
    free(dataset);
}

Dataset* load_dataset_from_file(const char* file_path) {
    if (!file_path) return NULL;
    
    if (is_json_file(file_path)) {
        Dataset* ds = create_dataset(DATASET_TYPE_JSON);
        if (!ds) return NULL;
        
        char* content = read_file_content(file_path);
        if (!content) {
            free_dataset(ds);
            return NULL;
        }
        
        ds->json_data = parse_json_dataset(content, file_path);
        free(content);
        
        if (!ds->json_data) {
            free_dataset(ds);
            return NULL;
        }
        
        ds->total_samples = ds->json_data->num_samples;
        return ds;
    } else {
        Dataset* ds = create_dataset(DATASET_TYPE_TEXT);
        if (!ds) return NULL;
        
        ds->text_data = load_text_dataset_from_file(file_path);
        if (!ds->text_data) {
            free_dataset(ds);
            return NULL;
        }
        
        ds->total_samples = ds->text_data->num_lines;
        return ds;
    }
}

Dataset* load_dataset_from_directory(const char* dir_path, DatasetType type) {
    DIR* dir = opendir(dir_path);
    if (!dir) return NULL;
    
    Dataset* dataset = create_dataset(type);
    if (!dataset) {
        closedir(dir);
        return NULL;
    }
    
    // Count files first
    int file_count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            if ((type == DATASET_TYPE_TEXT && is_text_file(entry->d_name)) ||
                (type == DATASET_TYPE_JSON && is_json_file(entry->d_name)) ||
                type == DATASET_TYPE_MIXED) {
                file_count++;
            }
        }
    }
    
    if (file_count == 0) {
        free_dataset(dataset);
        closedir(dir);
        return NULL;
    }
    
    // Allocate file paths array
    dataset->file_paths = malloc(file_count * sizeof(char*));
    dataset->num_files = file_count;
    
    // Reset directory
    rewinddir(dir);
    
    int file_idx = 0;
    while ((entry = readdir(dir)) != NULL && file_idx < file_count) {
        if (entry->d_type == DT_REG) {
            if ((type == DATASET_TYPE_TEXT && is_text_file(entry->d_name)) ||
                (type == DATASET_TYPE_JSON && is_json_file(entry->d_name)) ||
                type == DATASET_TYPE_MIXED) {
                
                // Create full path
                char full_path[MAX_FILENAME_LEN];
                snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
                dataset->file_paths[file_idx] = strdup(full_path);
                
                // Load file content
                if (is_json_file(entry->d_name)) {
                    char* content = read_file_content(full_path);
                    if (content) {
                        JsonDataset* json_data = parse_json_dataset(content, full_path);
                        if (json_data) {
                            // Merge with existing JSON data
                            for (int i = 0; i < json_data->num_samples; i++) {
                                add_sample_to_json_dataset(dataset->json_data, 
                                                         json_data->inputs[i], 
                                                         json_data->outputs[i]);
                            }
                            free_json_dataset(json_data);
                        }
                        free(content);
                    }
                } else {
                    TextDataset* text_data = load_text_dataset_from_file(full_path);
                    if (text_data) {
                        // Merge with existing text data
                        for (int i = 0; i < text_data->num_lines; i++) {
                            add_line_to_text_dataset(dataset->text_data, text_data->lines[i]);
                        }
                        free_text_dataset(text_data);
                    }
                }
                
                file_idx++;
            }
        }
    }
    
    closedir(dir);
    
    // Calculate total samples
    dataset->total_samples = 0;
    if (dataset->text_data) dataset->total_samples += dataset->text_data->num_lines;
    if (dataset->json_data) dataset->total_samples += dataset->json_data->num_samples;
    
    return dataset;
}

Dataset* load_finetune_dataset(const char* dir_path) {
    return load_dataset_from_directory(dir_path, DATASET_TYPE_JSON);
}

void shuffle_dataset(Dataset* dataset) {
    if (!dataset) return;
    
    srand(time(NULL));
    
    if (dataset->text_data && dataset->text_data->num_lines > 0) {
        for (int i = dataset->text_data->num_lines - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            char* tmp = dataset->text_data->lines[i];
            dataset->text_data->lines[i] = dataset->text_data->lines[j];
            dataset->text_data->lines[j] = tmp;
        }
    }
    
    if (dataset->json_data && dataset->json_data->num_samples > 0) {
        for (int i = dataset->json_data->num_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            char* tmp_input = dataset->json_data->inputs[i];
            char* tmp_output = dataset->json_data->outputs[i];
            dataset->json_data->inputs[i] = dataset->json_data->inputs[j];
            dataset->json_data->outputs[i] = dataset->json_data->outputs[j];
            dataset->json_data->inputs[j] = tmp_input;
            dataset->json_data->outputs[j] = tmp_output;
        }
    }
}

int get_dataset_size(Dataset* dataset) {
    return dataset ? dataset->total_samples : 0;
}

// Batch creation
Batch* create_batch(Dataset* dataset, int start_idx, int batch_size, int seq_len, int vocab_size) {
    if (!dataset || batch_size <= 0 || seq_len <= 0) return NULL;
    
    Batch* batch = malloc(sizeof(Batch));
    if (!batch) return NULL;
    
    batch->batch_size = batch_size;
    batch->seq_len = seq_len;
    batch->actual_batch_size = 0;
    
    int total_size = batch_size * seq_len;
    batch->input_ids = malloc(total_size * sizeof(int));
    batch->target_ids = malloc(total_size * sizeof(int));
    
    if (!batch->input_ids || !batch->target_ids) {
        if (batch->input_ids) free(batch->input_ids);
        if (batch->target_ids) free(batch->target_ids);
        free(batch);
        return NULL;
    }
    
    // Initialize with padding
    for (int i = 0; i < total_size; i++) {
        batch->input_ids[i] = 0;  // PAD token
        batch->target_ids[i] = 0; // PAD token
    }
    
    // Fill batch with data
    int batch_idx = 0;
    int dataset_idx = start_idx;
    
    while (batch_idx < batch_size && dataset_idx < dataset->total_samples) {
        char* text = NULL;
        
        // Get text from appropriate dataset
        if (dataset->text_data && dataset_idx < dataset->text_data->num_lines) {
            text = dataset->text_data->lines[dataset_idx];
        } else if (dataset->json_data) {
            int json_idx = dataset_idx - (dataset->text_data ? dataset->text_data->num_lines : 0);
            if (json_idx >= 0 && json_idx < dataset->json_data->num_samples) {
                // For JSON, concatenate input and output
                int total_len = strlen(dataset->json_data->inputs[json_idx]) + 
                               strlen(dataset->json_data->outputs[json_idx]) + 2;
                text = malloc(total_len);
                if (text) {
                    snprintf(text, total_len, "%s %s", 
                            dataset->json_data->inputs[json_idx],
                            dataset->json_data->outputs[json_idx]);
                }
            }
        }
        
        if (text) {
            // Simple tokenization (character-level)
            int seq_pos = 0;
            for (int i = 0; text[i] && seq_pos < seq_len; i++) {
                int token_id = (int)text[i] % vocab_size;
                batch->input_ids[batch_idx * seq_len + seq_pos] = token_id;
                if (seq_pos < seq_len - 1) {
                    batch->target_ids[batch_idx * seq_len + seq_pos] = token_id;
                }
                seq_pos++;
            }
            
            // Free temporary text if it was allocated
            if (text != dataset->text_data->lines[dataset_idx]) {
                free(text);
            }
            
            batch->actual_batch_size++;
        }
        
        batch_idx++;
        dataset_idx++;
    }
    
    return batch;
}

void free_batch(Batch* batch) {
    if (!batch) return;
    if (batch->input_ids) free(batch->input_ids);
    if (batch->target_ids) free(batch->target_ids);
    free(batch);
}