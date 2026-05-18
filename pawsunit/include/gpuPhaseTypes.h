#ifndef GPU_PHASE_TYPES_H
#define GPU_PHASE_TYPES_H

#include <atomic>
#include <ctime>
#include <string>
#include <event2/event_struct.h>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include "memManager.h"
#include "policyManager.h"

class eventHandler;

enum class workload_Class{
    BE,
    LC,
    UNK
};

enum class granularity{
    LONG,
    SHORT,
    UNK
};

typedef struct {
    pid_t pid;
    pid_t tid;
    int parentId;
    int depth;
    std::pair<pid_t, int> phaseId;
    struct timespec startTime;
    struct timespec endTime;
} metadata;

class gpuPhase{
    public:
        gpuPhase(const char* semanticIdentifier, const char* priority, const char* granularity);
        ~gpuPhase() = default;
        workload_Class workloadClass;
        std::string semanticIdentifier;
        granularity workloadGranularity;
        metadata phaseMetadata;
    private:
};

struct active_Phases{
    std::stack<gpuPhase> curPhases;
    std::atomic_uint64_t numLC{0};
}; 

class phaseManager{
    public:
        void initPhaseManager();
        ~phaseManager() = default;
        void phaseBegin(const char* semanticIdentifier, char* priority, const char* granularity);
        void phaseEnd();
        void setPhaseData(gpuPhase& gpuPhase);
        void updatePhaseTable(gpuPhase& newPhase);
        active_Phases activePhases;
        eventHandler* phaseWriter = nullptr;
        //policyAdditions
        memManager* memoryManager;
        policyManager* policyManagerHandler;
    private:
};

#endif
