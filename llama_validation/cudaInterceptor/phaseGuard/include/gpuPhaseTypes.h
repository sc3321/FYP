#ifndef GPU_PHASE_TYPES_H
#define GPU_PHASE_TYPES_H

#include <atomic>
#include <ctime>
#include <vector>
#include <string>
#include <event2/event_struct.h>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <thread>
#include "memManager.h"
#include "policyManager.h"

class eventHandler;

typedef std::pair<pid_t, int> phaseID;

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
    phaseID phaseId;
    struct timespec startTime;
    struct timespec endTime;
} metadata;

class gpuPhase{
    public:
        gpuPhase(const char* semanticIdentifier,workload_Class priority,granularity granularity, bool usePolicy);
        ~gpuPhase() = default;
        workload_Class workloadClass;
        std::string semanticIdentifier;
        granularity workloadGranularity;
        metadata phaseMetadata;
        bool getPolicyInformation() const {return usePolicy; }
    private:
        bool usePolicy;
};

struct active_Phases{
   std::vector<gpuPhase> curPhases;
   pthread_mutex_t addPhaseAllowed; 
};

class phaseManager{
    public:
        phaseManager();
        ~phaseManager() = default;
        phaseID phaseBegin(const char* semanticIdentifier,workload_Class priority, granularity granularity, bool usePolicy = true);
        void phaseEnd(phaseID);
        void setPhaseData(gpuPhase& gpuPhase);
        void updatePhaseTable(gpuPhase& newPhase);
        active_Phases activePhases;
        eventHandler* phaseWriter = nullptr;
        //policyAdditions
        memManager* memoryManager = nullptr;
        policyManager* policyManagerHandler = nullptr;
        void cleanup();
        void setPolicyMode();
    private:
        policyMode currentPolicyMode = policyMode::NONE;
        // Diagnostic sampling thread; only active if GPU_PHASE_POLICY_SAMPLE_MS is set.
        std::thread samplingThread;
        std::atomic<bool> samplingStop{false};
        void startSamplingIfRequested();
        void stopSamplingIfRunning();
};

// use this for short lived phases.
class phaseGuard {
    public:
        phaseGuard(phaseManager& phase_Manager, const char* semanticIdentifier, workload_Class wClass, granularity wGran, bool usePolicy);

        ~phaseGuard();
    private:
        phaseManager* ptrToPhaseManager = nullptr;
        phaseID phase_id;
        bool active;

};

#endif
