#ifndef buffer__h
#define buffer__h

typedef struct Buffer Buffer;
struct Buffer {
  void *memory;
  size_t size;
};

#endif