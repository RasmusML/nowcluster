#include <stdlib.h>
#include "queue.h"

Queue *queue_create() {
	Queue *q = (Queue *)malloc(sizeof(Queue));

	if (q == NULL) {
    return NULL;
  }

	q->size = 0;
	q->head = NULL;
	q->tail = NULL;

	return q;
}

int queue_enqueue(void *value, Queue *q) {
	Node *node = (Node *)malloc(sizeof(Node));

	if (node == NULL) {
		return q->size;
	}

	node->value = value;
	node->next = NULL;

	if (q->head == NULL) {
		q->head = node;
		q->tail = node;
		q->size = 1;

		return q->size;
	}

	q->tail->next = node;
	q->tail = node;
	q->size += 1;

	return q->size;
}

void* queue_dequeue(Queue *q) {
	if (q->size == 0) {
		return NULL;
	}

	void *value = NULL;
	Node *tmp = NULL;

	value = q->head->value;
	tmp = q->head;
	q->head = q->head->next;
	q->size -= 1;

	free(tmp);

	return value;
}

void queue_free(Queue *q) {
	if (q == NULL) {
		return;
	}

	while (q->head != NULL) {
		Node *tmp = q->head;
		q->head = q->head->next;
		if (tmp->value != NULL) {
			free(tmp->value);
		}

		free(tmp);
	}

	free(q);
}