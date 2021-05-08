#ifndef queue__h
#define queue__h

typedef struct Node node;
struct Node {
	void *value;
	Node *next;
};

typedef struct Queue Queue;
struct Queue {
	int size;
	Node *head;
	Node *tail;
};

Queue *queue_create();
int queue_enqueue(void *value, Queue *q);
void* queue_dequeue(Queue *q);
void queue_free(Queue *q);

#endif