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

class eventHandler;

enum class workload_Class{
    BE,
    LC,
    UNK
};

typedef struct {
    pid_t pid;
    pid_t tid;
    std::pair<pid_t, int> phaseId;
    struct timespec timeNow;    
} metadata;

class gpuPhase{
    public:
        gpuPhase(const char* semanticIdentifier, const char* priority);
        ~gpuPhase() = default;
        workload_Class workloadClass;
        std::string semanticIdentifier;
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
        void phaseBegin(const char* semanticIdentifier, char* priority);
        void phaseEnd();
        void setPhaseId(gpuPhase& gpuPhase);
        void updatePhaseTable(const gpuPhase& newPhase);
        active_Phases activePhases;
        eventHandler* phaseWriter = nullptr;
    private:
};

#endif
