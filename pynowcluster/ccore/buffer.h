#ifndef BUFFER_H
#define BUFFER_H

struct Buffer {
  void *memory;
  size_t size;
};

typedef struct Buffer Buffer;

#endif