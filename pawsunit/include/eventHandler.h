#ifndef EVENT_HANDLER_H
#define EVENT_HANDLER_H

#include "gpuPhaseTypes.h"
#include <stdio.h>

class eventHandler {
public:
    FILE* fptr;
    void writeEvent(bool begin, gpuPhase& gpuPhase); 
private:

};

#endif
















