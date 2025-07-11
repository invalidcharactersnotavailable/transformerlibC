#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

// A simple memory arena for fast, temporary allocations.
typedef struct {
    char* start;
    size_t size;
    size_t offset;
} Arena;

Arena* create_arena(size_t size);
void destroy_arena(Arena* arena);
void* arena_alloc(Arena* arena, size_t size);
void arena_reset(Arena* arena);

#endif // MEMORY_H 