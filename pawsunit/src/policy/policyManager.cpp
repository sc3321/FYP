#include "../../include/gpuPhaseTypes.h"

policyManager::policyManager(memManager& memoryManager){
    ptrMemoryManager = &memoryManager;
}

void policyManager::beginPDUpdate(gpuPhase& curPhase){
   robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);
   if(curPhase.workloadClass == workload_Class::LC){
       ptrMemoryManager->ptrToShm->activeLC++;
   }
   if(curPhase.workloadClass == workload_Class::BE && curPhase.workloadGranularity == granularity::LONG){
        ptrMemoryManager->ptrToShm->activeBELong++;
   }
}

void policyManager::endPDUpdate(gpuPhase& curPhase){
    robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);
    if(curPhase.workloadClass == workload_Class::LC){
        ptrMemoryManager->ptrToShm->activeLC--;
    }
    if(curPhase.workloadClass == workload_Class::BE && curPhase.workloadGranularity == granularity::LONG){
        ptrMemoryManager->ptrToShm->activeBELong--;
    }
}

void policyManager::readPolicyData(){
    robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);
    curReadData = ptrMemoryManager->ptrToShm;

} 
