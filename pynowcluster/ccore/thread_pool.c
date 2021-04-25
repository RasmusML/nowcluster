#include <thread>
#include <vector>

#include "types.h"


// https://www.cplusplus.com/reference/condition_variable/condition_variable/

void loop() {
    while (1) {

    }
}

void start() {
    uint32 num_of_logical_processers = thread::hardware_concurrency();

    std::vector<std::thread> workers;

    for (uint32 i = 0; i < num_of_logical_processers; i++) {
        workers.push_back(std::thread(loop));
    }
}