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

    // Capture wall-clock timestamp for time-series alignment.
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    // Build the record once; write to file if configured, else stderr.
    char header[256];
    snprintf(header, sizeof(header),
        "\n[PolicyCounters] pid=%d ts=%ld.%09ld where=%s\n",
        (int)getpid(), (long)ts.tv_sec, ts.tv_nsec, where);

    const char* logPath = std::getenv("GPU_PHASE_POLICY_LOG");
    FILE* out = nullptr;
    bool ownsFile = false;

    if (logPath != nullptr && logPath[0] != '\0') {
        // Append mode so multiple processes can share one file; each line
        // self-identifies via pid and timestamp.
        out = std::fopen(logPath, "a");
        if (out != nullptr) {
            ownsFile = true;
        }
    }

    if (out == nullptr) {
        out = stderr;
    }

    std::fprintf(out, "%s", header);
    std::fprintf(out, "  policyChecks=%d\n", s->policyChecks);
    std::fprintf(out, "  activeLC=%d\n", s->activeLC);
    std::fprintf(out, "  activeBELong=%d\n", s->activeBELong);
    std::fprintf(out, "  activeBEChunked=%d\n", s->activeBEChunked);
    std::fprintf(out, "  beLongSawLCActive=%d\n", s->beLongSawLCActive);
    std::fprintf(out, "  BEImmAdmit=%d\n", s->BEImmAdmit);
    std::fprintf(out, "  BEDelayAdmit=%ld\n", s->BEDelayAdmit);
    std::fprintf(out, "  BEThrottleCount=%d\n", s->BEThrottleCount);
    std::fprintf(out, "  BEWaitus=%lld\n", s->BEWaitus);
    std::fprintf(out, "  BELongImmAdmit=%d\n", s->BELongImmAdmit);
    std::fprintf(out, "  BELongDelayAdmit=%ld\n", s->BELongDelayAdmit);
    std::fprintf(out, "  BELongThrottleCount=%d\n", s->BELongThrottleCount);
    std::fprintf(out, "  BELongWaitus=%lld\n", s->BELongWaitus);
    std::fprintf(out, "  configured_delay_us=%lu\n", (unsigned long)delay);
    std::fflush(out);

    if (ownsFile) {
        std::fclose(out);
    }
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

    if(!curPhase.getPolicyInformation()){
        robustLockGuard lock(ptrMemoryManager->ptrToShm->writeAllowed);
        beginPDUpdate(curPhase);
        return;
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
