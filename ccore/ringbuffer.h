#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include "types.h"

typedef struct RingBuffer RingBuffer;
struct RingBuffer {
  uint32 first;
  uint32 last;
  uint32 capacity;
  uint32 used;

  size_t element_size;
  void *memory;
};

void ringbuffer_init(uint32 capacity, size_t element_size, RingBuffer *buffer);
void *ringbuffer_alloc(RingBuffer *buffer);
void *ringbuffer_get(RingBuffer *buffer);
void ringbuffer_free(RingBuffer *buffer);

#endif