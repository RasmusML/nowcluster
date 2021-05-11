#include <stdlib.h>
#include <assert.h>

#include "ringbuffer.h"

void ringbuffer_init(uint32 capacity, size_t element_size, RingBuffer *buffer) {
  buffer->first = 0;
  buffer->last = 0;
  buffer->used = 0;
  buffer->capacity = capacity;
  buffer->element_size = element_size;
  buffer->memory = (void *)malloc(capacity * element_size);
}

void *ringbuffer_alloc(RingBuffer *buffer) {
  if (buffer->used == buffer->capacity) assert(0 && "buffer is full!");

  void *pointer = (void *) ((uint8 *) buffer->memory + buffer->last * buffer->element_size);
  buffer->used += 1;
  
  buffer->last += 1;
  if (buffer->last >= buffer->capacity) buffer->last = 0;

  return pointer;
}

void *ringbuffer_get(RingBuffer *buffer) {
  return (void *) ((uint8 *)buffer->memory + buffer->first * buffer->element_size);
}

void ringbuffer_free(RingBuffer *buffer) {
  if (buffer->used > 0) {
    buffer->used -= 1;
    buffer->first += 1;
    if (buffer->first >= buffer->capacity) buffer->first = 0;
  }
}