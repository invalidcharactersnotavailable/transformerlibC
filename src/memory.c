#include "memory.h"
#include <stdlib.h>
#include <string.h>

Arena* create_arena(size_t size) {
    Arena* arena = (Arena*)malloc(sizeof(Arena));
    if (!arena) return NULL;
    arena->start = (char*)malloc(size);
    if (!arena->start) {
        free(arena);
        return NULL;
    }
    arena->size = size;
    arena->offset = 0;
    return arena;
}

void destroy_arena(Arena* arena) {
    if (arena) {
        free(arena->start);
        free(arena);
    }
}

void* arena_alloc(Arena* arena, size_t size) {
    // Align to 16 bytes for performance, good for SIMD
    size = (size + 15) & ~15;
    if (arena->offset + size > arena->size) {
        return NULL; // Out of memory
    }
    void* ptr = arena->start + arena->offset;
    arena->offset += size;
    return ptr;
}

void arena_reset(Arena* arena) {
    arena->offset = 0;
} 