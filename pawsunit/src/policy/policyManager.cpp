#include "../../include/gpuPhaseTypes.h"
#include <memory>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <iostream>

#define MAX_BE_LONG 0 

static uint64_t readEnvUInt64(const char* name, uint64_t defaultValue){
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0'){
        return defaultValue;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long value = std::strtoull(raw, &end, 10);

    if (errno != 0 || end == raw || *end != '\0') {
        std::cerr << "[PolicyConfig] Invalid " << name
                  << "='" << raw << "', using default "
                  << defaultValue << "\n";
        return defaultValue;
    }

    return static_cast<uint64_t>(value);

}

policyManager::policyManager(memManager& memoryManager){
    ptrMemoryManager = &memoryManager;
    delay = readEnvUInt64("BE_DELAY_US", 50);
}

void policyManager::beginPDUpdate(gpuPhase& curPhase){
   if(curPhase.workloadClass == workload_Class::LC){
       ptrMemoryManager->ptrToShm->activeLC++;
   }
   if(curPhase.workloadClass == workload_Class::BE){
       if(curPhase.workloadGranularity == granularity::LONG){
           ptrMemoryManager->ptrToShm->activeBELong++;
       }
       else{
           ptrMemoryManager->ptrToShm->activeBEChunked++;
       }
   }
}

void policyManager::endPDUpdate(gpuPhase& curPhase){
    robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);
    if(curPhase.workloadClass == workload_Class::LC){
        ptrMemoryManager->ptrToShm->activeLC--;
    }
    if(curPhase.workloadClass == workload_Class::BE){
       if(curPhase.workloadGranularity == granularity::LONG){
           ptrMemoryManager->ptrToShm->activeBELong--;
       }
       else{
           ptrMemoryManager->ptrToShm->activeBEChunked--;
       }
   }
}

void policyManager::readPolicyData(const char* where){
    robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);

    auto* s = ptrMemoryManager->ptrToShm;

    std::cerr
        << "\n[PolicyCounters] " << where << "\n"
        << "  policyChecks=" << s->policyChecks << "\n"

        << "  activeLC=" << s->activeLC << "\n"
        << "  activeBELong=" << s->activeBELong << "\n"
        << "  activeBEChunked=" << s->activeBEChunked << "\n"
    
        << "  beLongSawLCActive=" << s->beLongSawLCActive << "\n"

        << "  BEImmAdmit=" << s->BEImmAdmit << "\n"
        << "  BEDelayAdmit=" << s->BEDelayAdmit << "\n"
        << "  BEThrottleCount=" << s->BEThrottleCount << "\n"
        << "  BEWaitus=" << s->BEWaitus << "\n"

        << "  BELongImmAdmit=" << s->BELongImmAdmit << "\n"
        << "  BELongDelayAdmit=" << s->BELongDelayAdmit << "\n"
        << "  BELongThrottleCount=" << s->BELongThrottleCount << "\n"
        << "  BELongWaitus=" << s->BELongWaitus << "\n"

        << "  configured_delay_us=" << delay << "\n"
        << std::endl;
}

bool policyManager::naivePolicy(gpuPhase& curPhase, bool& tried){
    bool admitted = true;
    robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);

    if(curPhase.workloadClass == workload_Class::BE){
        if(ptrMemoryManager->ptrToShm->activeLC > 0){
            if(!tried){
                ptrMemoryManager->ptrToShm->BEDelayAdmit++;
                tried = true;
            }
            ptrMemoryManager->ptrToShm->BEThrottleCount++;
            ptrMemoryManager->ptrToShm->BEWaitus += delay;
            return !admitted;
        }
        else{
            if(!tried){
                ptrMemoryManager->ptrToShm->BEImmAdmit++;
            }
            beginPDUpdate(curPhase);
            return admitted;
        }
    }
    else{
        beginPDUpdate(curPhase);
        return admitted;
    }

}

bool policyManager::properPolicy(gpuPhase& curPhase, bool& tried){
    bool admitted = true;
    robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);

    if(curPhase.workloadClass == workload_Class::BE && curPhase.workloadGranularity == granularity::LONG){
        if(ptrMemoryManager->ptrToShm->activeLC > 0) ptrMemoryManager->ptrToShm->beLongSawLCActive++;
        if(ptrMemoryManager->ptrToShm->activeLC > 0 && ptrMemoryManager->ptrToShm->activeBELong >= MAX_BE_LONG){
            if(!tried){
                ptrMemoryManager->ptrToShm->BELongDelayAdmit++;
                tried = true;
            }
            ptrMemoryManager->ptrToShm->BELongThrottleCount++;
            ptrMemoryManager->ptrToShm->BELongWaitus += delay;
            return !admitted;
        }else{
            if(!tried){
                ptrMemoryManager->ptrToShm->BELongImmAdmit++;
            }
            beginPDUpdate(curPhase);
            return admitted;
        }     
    }
    else{
        beginPDUpdate(curPhase);
        return admitted;
    }
}

void policyManager::applyPolicy(gpuPhase& curPhase, policyMode policy){
    {
    robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);
    ptrMemoryManager->ptrToShm->policyChecks++;
    }
    if(policy == policyMode::NONE){
        robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);
        beginPDUpdate(curPhase);
        return;
    }
    else if(policy == policyMode::NAIVE_THROTTLE){
        bool tried = false;
        while(!naivePolicy(curPhase, tried)){
            usleep(delay);
        }
    }
    else if(policy == policyMode::CAP){
        bool tried = false;
        while(!properPolicy(curPhase, tried)){
            usleep(delay);
        }
    }
}
