#include "../../phaseGuard/include/memManager.h"
#include <nvml.h>
#include <stdlib.h>
#include <chrono>


struct dvfsData {
    int activeLC = 0;
    int activeBE = 0;
};

class dvfsManager {
    public:
        dvfsManager();
        ~dvfsManager();
        void powerController(int target);
        unsigned int classify();
        nvmlDevice_t gpuDevice;
        unsigned int deviceMaxPower;
    
    private:
       memManager* dvfsMemManager;
       dvfsData* snapShot;

       std::chrono::time_point<std::chrono::steady_clock> lastChanged = std::chrono::steady_clock::now();
      
       unsigned int currentPower;

};

