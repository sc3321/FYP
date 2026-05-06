#include "../../include/gpuPhaseTypes.h"
#include "../../include/eventHandler.h"
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <atomic>
#include <threads.h>

using phase_id_t = uint64_t;
static std::atomic<uint64_t> nextPhaseId{1};

workload_Class getPriority(char* priority){
    for(int i = 0; i < strlen(priority); ++i){
        priority[i] = toupper(priority[i]);
    }
    if(strcmp(priority, "BE") == 0){
        return workload_Class::BE;
    } else if (strcmp(priority, "LC") == 0){
        return workload_Class::LC;
    }
    else{
        return workload_Class::UNK;
    }
};

gpuPhase::gpuPhase(char* semanticIdentifier, char* priority){
    semanticIdentifier = semanticIdentifier;
    phaseMetadata.pid = getpid();
    phaseMetadata.tid = gettid();
    clock_gettime(CLOCK_MONOTONIC_COARSE, &phaseMetadata.timeNow); 
    workloadClass = getPriority(priority);
}

void phaseManager::initPhaseManager(){
   phaseWriter = (eventHandler*)std::malloc(sizeof(eventHandler)); 
}

void phaseManager::setPhaseId(gpuPhase& curPhase){
    curPhase.phaseMetadata.phaseId = {curPhase.phaseMetadata.pid, nextPhaseId.fetch_add(1, std::memory_order_relaxed)};
}

void phaseManager::updatePhaseTable(const gpuPhase& newPhase){
   activePhases.curPhases.push(newPhase);
   if(newPhase.workloadClass == workload_Class::LC){
       activePhases.numLC += 1;
   }
}

void phaseManager::phaseBegin(char* semanticIdentifier, char* priority){
    activePhases.curPhases.emplace(semanticIdentifier, priority);
    gpuPhase& ref = activePhases.curPhases.top();
    setPhaseId(ref);
    updatePhaseTable(ref);
     
}

void phaseManager::phaseEnd(){
    gpuPhase& mostRecent = activePhases.curPhases.top();
    phaseWriter->writeEvent(mostRecent);
    activePhases.curPhases.pop();
}



