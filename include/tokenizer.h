#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

// A very simple character-level tokenizer for demonstration.
// Maps characters to integers.

// Encodes a string into a sequence of integer tokens.
// Returns the number of tokens.
int encode(const char* text, int* tokens, int max_tokens);

// Decodes a sequence of integer tokens back into a string.
char* decode(const int* tokens, int num_tokens);

#endif // TOKENIZER_H 