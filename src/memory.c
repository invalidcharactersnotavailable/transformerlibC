#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// create a new arena of given size (bytes)
Arena* create_arena(size_t size) {
    if (size == 0) return NULL;
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

// free all memory used by the arena
void destroy_arena(Arena* arena) {
    if (!arena) return;
    free(arena->start);
    arena->start = NULL;
    arena->size = 0;
    arena->offset = 0;
    free(arena);
}

// allocate memory from the arena, optionally zeroed if zero_mem is nonzero
void* arena_alloc_ex(Arena* arena, size_t size, int zero_mem) {
    if (!arena || size == 0) return NULL;
    // align to 16 bytes for performance
    size = (size + 15) & ~15;
    if (arena->offset > arena->size || size > arena->size - arena->offset) {
        return NULL; // out of memory or overflow
    }
    void* ptr = arena->start + arena->offset;
    arena->offset += size;
    if (zero_mem) memset(ptr, 0, size);
    return ptr;
}

// allocate memory from the arena (not zeroed)
void* arena_alloc(Arena* arena, size_t size) {
    return arena_alloc_ex(arena, size, 0);
}

// reset the arena for reuse (does not free memory)
void arena_reset(Arena* arena) {
    if (!arena) return;
    arena->offset = 0;
}

// get the number of bytes used in the arena
size_t arena_used(Arena* arena) {
    if (!arena) return 0;
    return arena->offset;
}

// get the number of bytes available in the arena
size_t arena_available(Arena* arena) {
    if (!arena) return 0;
    if (arena->offset > arena->size) return 0;
    return arena->size - arena->offset;
} 