#include "../include/dvfsManager.h"
#include <csignal>
#include <atomic>
#include <nvml.h>
#include <thread>


std::atomic<bool> keepRunning{true};
void handleSignal(int) {keepRunning = false};

int main() {
    signal(SIGTERM, handleSignal);
    signal(SIGINT, handleSignal );
    
    dvfsManager Manager;
    unsigned int targetPower = 0;
    nvmlReturn_t result;

    while(keepRunning){

        targetPower = Manager.classify();
        Manager.powerController(targetPower);

    }    
    
    result = nvmlDeviceSetPowerManagementLimit(Manager.gpuDevice,Manager.deviceMaxPower);
    
    if(result != NVML_SUCCESS) {
        std::cerr << "Failed to reset GPU power: " << nvmlErrorString(result) << std::endl;
    }
    nvmlShutdown();

    return 0;
}



