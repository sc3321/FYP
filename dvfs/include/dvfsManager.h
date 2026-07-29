#include "../../phaseGuard/include/memManager.h"
#include <nvml.h>
#include <stdlib.h>
#include <chrono>
#include <cstdio>

struct dvfsData {
    int activeLC = 0;
    int activeBE = 0;
};

class dvfsManager {
    public:
        dvfsManager();
        bool powerController(unsigned int target);
        unsigned int classify();
        nvmlDevice_t gpuDevice;
        unsigned int deviceDefaultPower;
        void logSample(unsigned int target, bool changed);
    
    private:
       memManager* dvfsMemManager;
       dvfsData* snapShot;
       FILE* logFile = nullptr;
       std::chrono::time_point<std::chrono::steady_clock> lastChanged = std::chrono::steady_clock::now();
      
       unsigned int currentPower;

};

