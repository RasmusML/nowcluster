#ifndef __timer_h
#define __timer_h

#include <time.h>

clock_t t;

void start() {
    t = clock();
}

void stop() {
    t = clock() - t;
}

double elapsed() {
    return t / (double) CLOCKS_PER_SEC;
}

#endif