#ifndef __memory_h
#define __memory_h

#include "types.h"

struct StackMemory {
  uint8 *memory;
  size_t total;

  size_t used;
};

void init_memory(size_t total_size, StackMemory *memory);
uint8 *push_memory(size_t size, StackMemory *memory);
void pop_memory(size_t size, StackMemory *memory);
void free_memory(StackMemory *memory);

#endif