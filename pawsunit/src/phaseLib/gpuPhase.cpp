#include "../../include/gpuPhaseTypes.h"
#include "../../include/eventHandler.h"
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <atomic>
#include <threads.h>

const char* sharedMemName = "/sharedMemName";

using phase_id_t = uint64_t;
static std::atomic<uint64_t> nextPhaseId{1};

workload_Class getPriority(const char* priority) {
    if (priority == nullptr) {
        return workload_Class::UNK;
    }

    if (strcmp(priority, "LC") == 0) {
        return workload_Class::LC;
    }

    if (strcmp(priority, "BE") == 0) {
        return workload_Class::BE;
    }

    if (strcmp(priority, "UNK") == 0) {
        return workload_Class::UNK;
    }

    return workload_Class::UNK;
}

granularity getGranularity(const char* granularity) {
    if (granularity == nullptr) {
        return granularity::UNK;
    }

    if (strcmp(granularity, "SHORT") == 0) {
        return granularity::SHORT;
    }

    if (strcmp(granularity, "LONG") == 0) {
        return granularity::LONG;
    }
    return granularity::UNK;
}

gpuPhase::gpuPhase(const char* inputSemanticIdentifier,const char* priority, const char* granularity){
    semanticIdentifier = inputSemanticIdentifier;
    phaseMetadata.pid = getpid();
    phaseMetadata.depth = 0;
    phaseMetadata.parentId = 0;
    phaseMetadata.tid = gettid();
    clock_gettime(CLOCK_MONOTONIC_COARSE, &phaseMetadata.startTime); 
    workloadClass = getPriority(priority);
    workloadGranularity = getGranularity(granularity);
}

void phaseManager::initPhaseManager(){
   void* rawWriter = (eventHandler*)std::malloc(sizeof(eventHandler));
   if(rawWriter == nullptr){
        throw "Could not allocate raw bytes for eventWriter";
   }
   phaseWriter = ::new (rawWriter) eventHandler();
   void* rawBytes = (memManager*)std::malloc(sizeof(memManager));
   if(rawBytes == nullptr){
        throw "Could not allocate raw bytes for memoryManager";
   }
   memoryManager = ::new (rawBytes) memManager(sharedMemName);
   void* rawBytesPolicy   = (policyManager*)std::malloc(sizeof(policyManager));
   if(rawBytesPolicy == nullptr){
        throw "Could not allocate raw bytes for policyManager";
   }
   policyManagerHandler = ::new (rawBytesPolicy) policyManager(*memoryManager);
}

void phaseManager::setPhaseData(gpuPhase& curPhase){
    curPhase.phaseMetadata.phaseId = {curPhase.phaseMetadata.pid, nextPhaseId.fetch_add(1, std::memory_order_relaxed)};
    clock_gettime(CLOCK_MONOTONIC_COARSE, &curPhase.phaseMetadata.startTime); 

}

void phaseManager::updatePhaseTable(gpuPhase& newPhase){
   if(newPhase.workloadClass == workload_Class::LC){
       activePhases.numLC += 1;
   }
   if(!activePhases.curPhases.empty()){
       gpuPhase& ref = activePhases.curPhases.top();
       newPhase.phaseMetadata.parentId = ref.phaseMetadata.phaseId.second;
       newPhase.phaseMetadata.depth = ref.phaseMetadata.depth + 1;
   }
   activePhases.curPhases.push(newPhase);
}

void phaseManager::phaseBegin(const char* semanticIdentifier, char* priority, const char* granularity){
    gpuPhase newPhase(semanticIdentifier, priority, granularity);
    // Apply Policy(newPhase)
    setPhaseData(newPhase);
    updatePhaseTable(newPhase);
    policyManagerHandler->beginPDUpdate(newPhase);
    policyManagerHandler->readPolicyData();
    std::cout << "activeLC: " << policyManagerHandler->curReadData->activeLC << "   active BE: " << policyManagerHandler->curReadData->activeBELong << "\n";
    phaseWriter->writeEvent(true, newPhase);
}

void phaseManager::phaseEnd(){
    gpuPhase& mostRecent = activePhases.curPhases.top();
    policyManagerHandler->endPDUpdate(mostRecent);
    policyManagerHandler->readPolicyData();
    std::cout << "activeLC: " << policyManagerHandler->curReadData->activeLC << "   active BE: " << policyManagerHandler->curReadData->activeBELong << "\n";
    clock_gettime(CLOCK_MONOTONIC_COARSE, &mostRecent.phaseMetadata.endTime); 
    phaseWriter->writeEvent(false, mostRecent);
    activePhases.curPhases.pop();
     
}

void phaseManager::cleanup(){
    munmap(memoryManager->ptrToShm, sizeof(policyData));
    shm_unlink(sharedMemName);
}

