#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/shm.h>
#include <unistd.h>

class memManager;
class gpuPhase;
struct policyData;

class policyManager{
    public:
        policyManager(memManager& memoryManager);
        void beginPDUpdate(gpuPhase& curPhase);
        void endPDUpdate(gpuPhase& curPhase);
        void applyPolicy();
        void readPolicyData();
        policyData* curReadData;
    private:
        memManager* ptrMemoryManager;
};
