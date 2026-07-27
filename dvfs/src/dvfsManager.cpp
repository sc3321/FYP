#include "../../phaseGuard/include/memManager.h"
#include <iostream>
#include <nvml.h>
#include <stdlib.h>

int main() {
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    
    unsigned int numOfGPUs = 0;

    result = nvmlDeviceGetCount_v2(&numOfGPUs);   
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }

    std::cout << "Number of GPUs found: " << numOfGPUs << std::endl;
    
    unsigned int deviceIndex = 0;
    nvmlDevice_t gpuDevice;

    result = nvmlDeviceGetHandleByIndex_v2(deviceIndex, &gpuDevice);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }
    
    unsigned int gpuPowerLimit;
    result = nvmlDeviceGetPowerManagementLimit(gpuDevice, &gpuPowerLimit)
    std::cout << "default power limit: " gpuPowerLimit << "\n";

    result = nvmlDeviceSetPowerManagementLimit(gpuDevice, 200);
    
    result = nvmlDeviceGetPowerManagementLimit(gpuDevice, &gpuPowerLimit)
    std::cout << "new power limit: " gpuPowerLimit << "\n" ;


    nvmlShutdown();
    return 0;
}

