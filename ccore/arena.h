#ifndef ARENA_H
#define ARENA_H

#define DEFAULT_ALIGNMENT (2*sizeof(void *))

struct Arena {
	unsigned char *buf;
	size_t buf_len;
	size_t prev_offset;
	size_t curr_offset;
};

typedef struct Arena Arena;

void arena_init(Arena *a, void *backing_buffer, size_t backing_buffer_length);
void *arena_alloc(Arena *a, size_t size);
void arena_free_all(Arena *a);

#endif