#pragma once

struct Buffer {
  void *memory;
  size_t size;
};

typedef struct Buffer Buffer;
