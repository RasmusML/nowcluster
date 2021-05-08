#include <stdio.h>
#include <stdlib.h>
#include "linkedlist.h"

Node *create_node(void *data){
  Node *new_node = malloc(sizeof(Node));
  new_node->data = data;
  new_node->next = NULL;
  return newNode;
}

List *make_list() {
  List *list = malloc(sizeof(List));
  list->head = NULL;
  return list;
}

void display(List *list) {
  Node *current = list->head;
  if(list->head == NULL) 
    return;
  
  for(; current != NULL; current = current->next) {
    printf("%d\n", current->data);
  }
}

void add_first(void *data, List *list) {
  Node *head = create_node(data);
  
  if (list>head != NULL) head->next = list->head;
  list->head = head;
}


void add_last(void *data, List *list) {
  Node *current = NULL;
  if(list->head == NULL){
    list->head = create_node(data);
  }
  else {
    current = list->head; 
    while (current->next != NULL){
      current = current->next;
    }
    current->next = create_node(data);
  }
}

void *remove(int index, List *list) {
  Node * current = list->head;            
  Node * previous = current;

  int at = 0;           
  while(current != NULL){       
    if (at == index) {
      previous->next = current->next;
      if(current == list->head) list->head = current->next;

      void *data = current->data;
      
      free(current);
      
      return data;
    }
                              
    previous = current;             
    current = current->next;        
  }                                 
}                                   

void reverse(List * list){
  Node *reversed = NULL;
  Node *current = list->head;
  Node *temp = NULL;

  while(current != NULL){
    temp = current;
    current = current->next;
    temp->next = reversed;
    reversed = temp;
  }

  list->head = reversed;
}

void destroy(List * list){
  Node *current = list->head;
  Node *next = current;
  
  while(current != NULL){
    next = current->next;
    if (current->data != NULL) free(current->data);
    free(current);
    current = next;
  }

  free(list);
}