#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdlib.h>

typedef struct Arena {
    char* buffer;
    size_t size;
    size_t offset;
    struct Arena* next;
} Arena;

/**
 * create_arena - create a new memory arena
 * @size: size of the arena in bytes
 * returns: pointer to arena or NULL on failure
 */
Arena* create_arena(size_t size);

/**
 * destroy_arena - destroy an arena and free all memory
 * @arena: arena to destroy
 */
void destroy_arena(Arena* arena);

/**
 * arena_alloc - allocate memory from arena
 * @arena: arena to allocate from
 * @size: size to allocate
 * returns: pointer to allocated memory or NULL on failure
 */
void* arena_alloc(Arena* arena, size_t size);

/**
 * arena_reset - reset arena to beginning (free all allocations)
 * @arena: arena to reset
 */
void arena_reset(Arena* arena);

/**
 * arena_get_used - get amount of memory used in arena
 * @arena: arena to check
 * returns: number of bytes used
 */
size_t arena_get_used(Arena* arena);

/**
 * arena_get_remaining - get amount of memory remaining in arena
 * @arena: arena to check
 * returns: number of bytes remaining
 */
size_t arena_get_remaining(Arena* arena);

#endif // ARENA_H