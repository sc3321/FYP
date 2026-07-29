#include "../include/dvfsManager.h"
#include <csignal>
#include <atomic>
#include <nvml.h>
#include <thread>
#include <iostream>

std::atomic<bool> keepRunning{true};
void handleSignal(int) {keepRunning = false ;}

int main() {
    signal(SIGTERM, handleSignal);
    signal(SIGINT, handleSignal );
    
    dvfsManager Manager;
    unsigned int targetPower = 0;
    nvmlReturn_t result;

    while(keepRunning){

        targetPower = Manager.classify();
        bool changed = Manager.powerController(targetPower);
        Manager.logSample(targetPower, changed);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }    
    
    result = nvmlDeviceSetPowerManagementLimit(Manager.gpuDevice,Manager.deviceDefaultPower);
    
    if(result != NVML_SUCCESS) {
        std::cerr << "Failed to reset GPU power: " << nvmlErrorString(result) << std::endl;
    }
    nvmlShutdown();

    return 0;
}



