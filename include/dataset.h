#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
// JSON parsing will be done manually without external dependencies

#define MAX_DATASET_SIZE 10000000
#define MAX_LINE_LEN 8192
#define MAX_FILENAME_LEN 512
#define MAX_FILES_PER_DIR 10000

// Dataset types
typedef enum {
    DATASET_TYPE_TEXT,      // Plain text files
    DATASET_TYPE_JSON,      // JSON finetuning format
    DATASET_TYPE_MIXED      // Mixed text and JSON
} DatasetType;

// Text dataset structure
typedef struct {
    char** lines;
    int num_lines;
    int capacity;
    char* source_file;
} TextDataset;

// JSON finetuning dataset structure
typedef struct {
    char** inputs;
    char** outputs;
    int num_samples;
    int capacity;
    char* source_file;
} JsonDataset;

// Combined dataset structure
typedef struct {
    DatasetType type;
    TextDataset* text_data;
    JsonDataset* json_data;
    int total_samples;
    char** file_paths;
    int num_files;
} Dataset;

// Dataset loading functions
Dataset* load_dataset_from_file(const char* file_path);
Dataset* load_dataset_from_directory(const char* dir_path, DatasetType type);
Dataset* load_finetune_dataset(const char* dir_path);

// Dataset manipulation
void shuffle_dataset(Dataset* dataset);
void free_dataset(Dataset* dataset);
int get_dataset_size(Dataset* dataset);

// Batch creation
typedef struct {
    int* input_ids;
    int* target_ids;
    int batch_size;
    int seq_len;
    int actual_batch_size;
} Batch;

Batch* create_batch(Dataset* dataset, int start_idx, int batch_size, int seq_len, int vocab_size);
void free_batch(Batch* batch);

// Utility functions
int is_text_file(const char* filename);
int is_json_file(const char* filename);
char* read_file_content(const char* file_path);
int count_lines_in_file(const char* file_path);
int count_files_in_directory(const char* dir_path, const char* extension);

// JSON parsing helpers
JsonDataset* parse_json_dataset(const char* json_content, const char* source_file);
int validate_json_format(const char* json_content);

#endif // DATASET_H