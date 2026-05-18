#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/shm.h>
#include <unistd.h>

class memManager;
class gpuPhase;

class policyManager{
    public:
        policyManager(memManager& memoryManager);
        void beginPDUpdate(gpuPhase& curPhase);
        void endPDUpdate();
        void applyPolicy();
    private:
        memManager* ptrMemoryManager;
};
