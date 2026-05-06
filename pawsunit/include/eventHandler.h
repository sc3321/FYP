#ifndef EVENT_HANDLER_H
#define EVENT_HANDLER_H

#include "gpuPhaseTypes.h"
#include <stdio.h>

class eventHandler {
public:
    FILE* fptr;
    void writeEvent(gpuPhase& gpuPhase); 
private:

};

#endif
















