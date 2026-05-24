#include <cstdint>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/shm.h>
#include <unistd.h>

class memManager;
class gpuPhase;
struct policyData;

enum class policyMode {
    NONE,
    NAIVE_THROTTLE,
    CAP
};

class policyManager{
    public:
        policyManager(memManager& memoryManager);
        // beginPDUpdate must be called with writeAllowed lock held
        void beginPDUpdate(gpuPhase& curPhase);
        void endPDUpdate(gpuPhase& curPhase);
        void applyPolicy(gpuPhase& curPhase, policyMode policy);
        void readPolicyData(const char* where);
        policyData* curReadData;
        bool naivePolicy(gpuPhase& curPhase, bool& tried);
        bool properPolicy(gpuPhase& curPhase, bool& tried);       
    private:
        memManager* ptrMemoryManager;
        uint64_t delay;
};
