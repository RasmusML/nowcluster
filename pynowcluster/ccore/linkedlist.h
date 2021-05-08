#ifndef LINKEDLIST_HEADER
#define LINKEDLIST_HEADER

typedef struct Node Node;
struct Node {
  void *data;
  Node *next;
};

typedef struct List List;
struct List {
  Node *head; 
};


List *make_list();
void add_first(void *data, List *list);
void add(void *data, List *list);
void *remove(int index, List *list);
void display(List *list);
void reverse(List *list);
void destroy(List *list);

#endif