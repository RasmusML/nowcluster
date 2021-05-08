#ifndef buffer__h
#define buffer__h

struct Buffer {
  void *memory;
  size_t size;
};

typedef struct Buffer Buffer;

#endif