
#include <stdlib.h>
#include <assert.h>

#include "memory.h"

void init_memory(size_t total_size, TemporaryMemory *memory) {
    memory->memory = (uint8 *) malloc(total_size * sizeof(uint8));
    memory->total = total_size;
    memory->used = 0;
}

uint8 *allocate_memory(size_t size, TemporaryMemory *memory) {
    uint8 *p = memory->memory + memory->used;

    size_t new_used = memory->used + size;
    assert(memory->total >= new_used);
    memory->used = new_used;
    
    return p;
}

void reset_memory(TemporaryMemory *memory) {
    memory->used = 0;
}

void free_memory(TemporaryMemory *memory) {
    free(memory->memory);
}

// @TODO: create a TemporaryStackMemory and set that in the beginning, i.e. when calling fractal k-means.
// This way we don't have to call malloc and free so many times, i.e. in k-means and random_init
// We could just parse the allocated memory as argument into random_init instead for performance reasons,
// but the code becomes too complicated this way.
// maskResult will not use temporary memory, because it is of variable size. So simpler just to split those 2.