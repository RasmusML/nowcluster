#ifndef __memory_h
#define __memory_h

#include "types.h"

struct TemporaryMemory {
  uint8 *memory;
  size_t total;

  size_t used;
};

void init_memory(size_t total_size, TemporaryMemory *memory);
uint8 *allocate_memory(size_t size, TemporaryMemory *memory);
void reset_memory(TemporaryMemory *memory);
void free_memory(TemporaryMemory *memory);

#endif