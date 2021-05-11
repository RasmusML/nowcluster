#ifndef QUEUE_H
#define QUEUE_H

struct Node {
	void *value;
	struct Node *next;
};

typedef struct Node Node;

struct Queue {
	int size;
	Node *head;
	Node *tail;
};

typedef struct Queue Queue;

Queue *queue_create();
int queue_enqueue(void *value, Queue *q);
void* queue_dequeue(Queue *q);
void queue_free(Queue *q);

#endif