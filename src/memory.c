#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define ALIGNMENT 8

// Align size to the nearest multiple of ALIGNMENT
static inline size_t align_up(size_t size) {
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

Arena* create_arena(size_t size) {
    Arena* arena = (Arena*)malloc(sizeof(Arena));
    if (!arena) return NULL;

    arena->size = size;
    arena->offset = 0;
    arena->start = (char*)malloc(size);
    if (!arena->start) {
        free(arena);
        return NULL;
    }
    return arena;
}

void destroy_arena(Arena* arena) {
    if (arena) {
        free(arena->start);
        free(arena);
    }
}

void* arena_alloc(Arena* arena, size_t size) {
    size = align_up(size);
    if (arena->offset + size > arena->size) {
        // Not enough space
        return NULL; 
    }
    void* ptr = arena->start + arena->offset;
    arena->offset += size;
    return ptr;
}

void arena_reset(Arena* arena) {
    arena->offset = 0;
} 