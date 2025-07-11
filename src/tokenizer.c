#include "tokenizer.h"
#include <string.h>
#include <stdlib.h>

int encode(const char* text, int* tokens, int max_tokens) {
    int n = 0;
    for (int i = 0; text[i] != '\0' && i < max_tokens; i++) {
        tokens[i] = (int)text[i];
        n++;
    }
    return n;
}

char* decode(const int* tokens, int num_tokens) {
    char* text = (char*)malloc(num_tokens + 1);
    for (int i = 0; i < num_tokens; i++) {
        text[i] = (char)tokens[i];
    }
    text[num_tokens] = '\0';
    return text;
}
