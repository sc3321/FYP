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

static policyMode parse_policy_mode(const std::string& policy) {
    if (policy == "none") {
        return policyMode::NONE;
    }

    if (policy == "naive") {
        return policyMode::NAIVE_THROTTLE;
    }

    if (policy == "proper" || policy == "cap") {
        return policyMode::CAP;
    }

    return policyMode::UNK;
}

static policyMode readEnvPolicyMode(const char* name, policyMode defaultValue){
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0'){
        return defaultValue;
    }
    if(parse_policy_mode(raw) != policyMode::UNK){
        return parse_policy_mode(raw);
    }
    return defaultValue;
}


void phaseManager::setPolicyMode(){
    currentPolicyMode = readEnvPolicyMode("POLICY_MODE", policyMode::CAP);
}

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

gpuPhase::gpuPhase(const char* inputSemanticIdentifier,workload_Class priority, granularity granularity){
    semanticIdentifier = inputSemanticIdentifier;
    phaseMetadata.pid = getpid();
    phaseMetadata.depth = 0;
    phaseMetadata.parentId = 0;
    phaseMetadata.tid = gettid();
    clock_gettime(CLOCK_MONOTONIC_COARSE, &phaseMetadata.startTime);
    workloadClass = priority;
    workloadGranularity = granularity;
}

phaseManager::phaseManager(){
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
   setPolicyMode();
   policyManagerHandler = ::new (rawBytesPolicy) policyManager(*memoryManager);

}

void phaseManager::setPhaseData(gpuPhase& curPhase){
    curPhase.phaseMetadata.phaseId = {curPhase.phaseMetadata.pid, nextPhaseId.fetch_add(1, std::memory_order_relaxed)};
    clock_gettime(CLOCK_MONOTONIC_COARSE, &curPhase.phaseMetadata.startTime);

}

void phaseManager::updatePhaseTable(gpuPhase& newPhase){

   if(!activePhases.curPhases.empty()){
       gpuPhase& ref = activePhases.curPhases.front();
       newPhase.phaseMetadata.parentId = ref.phaseMetadata.phaseId.second;
       newPhase.phaseMetadata.depth = ref.phaseMetadata.depth + 1;
   }
   activePhases.curPhases.insert(activePhases.curPhases.begin(), newPhase);
}

phaseID phaseManager::phaseBegin(const char* semanticIdentifier, workload_Class priority, granularity granularity){
    gpuPhase newPhase(semanticIdentifier, priority, granularity);
    policyManagerHandler->applyPolicy(newPhase, currentPolicyMode);
    setPhaseData(newPhase);
    updatePhaseTable(newPhase);
    phaseWriter->writeEvent(true, newPhase);
    return newPhase.phaseMetadata.phaseId;
}

void phaseManager::phaseEnd(phaseID idToEnd){
    gpuPhase* endPhase = nullptr;
    for(size_t i = 0; i < activePhases.curPhases.size(); ++i){
        if(activePhases.curPhases[i].phaseMetadata.phaseId == idToEnd){
          policyManagerHandler->endPDUpdate(activePhases.curPhases[i]);
          clock_gettime(CLOCK_MONOTONIC_COARSE, &activePhases.curPhases[i].phaseMetadata.endTime);
          phaseWriter->writeEvent(false, activePhases.curPhases[i]);
          activePhases.curPhases.erase(activePhases.curPhases.begin() + i);
          return;
        }
    }
}

void phaseManager::cleanup(){
    munmap(memoryManager->ptrToShm, sizeof(policyData));
    shm_unlink(sharedMemName);
}


phaseGuard::phaseGuard(phaseManager& curManager, const char* semanticIdentifier, workload_Class wClass, granularity wGran){
    ptrToPhaseManager = &curManager;
    phase_id = ptrToPhaseManager->phaseBegin(semanticIdentifier, wClass, wGran);

}

phaseGuard::~phaseGuard(){
    if(ptrToPhaseManager){
        ptrToPhaseManager->phaseEnd(phase_id);
    }
}


