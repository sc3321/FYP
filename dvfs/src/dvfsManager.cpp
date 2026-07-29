#include "../include/dvfsManager.h"
#include <iostream>
#include <nvml.h>
#include <stdlib.h>
#include <cerrno>
#include <chrono>
#include <cstdio>

constexpr int MINSLEEP      =      2;
constexpr int DEFAULTMHZ    = 225000;
constexpr int MAXMHZ        = 300000;
constexpr int IDLE          = 175000;

static const char* getShmName() {
    const char* n = std::getenv("GPU_PHASE_SHM_NAME");
    return (n && n[0]) ? n : "/sharedMemName";
}

dvfsManager::dvfsManager() {
  
    void* rawBytes = (memManager*)std::malloc(sizeof(memManager));
   if(rawBytes == nullptr){
        throw "Could not allocate raw bytes for memoryManager";
   }
   dvfsMemManager = ::new (rawBytes) memManager(getShmName());
   
   void* rawBytesData = (dvfsData*)std::malloc(sizeof(dvfsData));
   if(rawBytesData == nullptr){
        throw "Could not allocate raw bytes for dvfssnapshot";
   }
   snapShot = (dvfsData*)rawBytesData;
   
   // initialisation
   nvmlInit();
   nvmlReturn_t result;
    
   
   result = nvmlDeviceGetHandleByIndex_v2(0, &gpuDevice);
    
   if(result != NVML_SUCCESS){
        std::cerr << "Failed to get GPU device: " << nvmlErrorString(result) << std::endl;
   }
   
   result = nvmlDeviceGetPowerManagementDefaultLimit(gpuDevice, &deviceMaxPower);
   
   if(result != NVML_SUCCESS){
        std::cerr << "Failed to get GPU max power: " << nvmlErrorString(result) << std::endl;
   }

   currentPower = deviceMaxPower;

}

unsigned int dvfsManager::classify() {
    
    {
      robustLockGuard lock(dvfsMemManager->ptrToShm->writeAllowed);
      snapShot->activeLC = dvfsMemManager->ptrToShm->activeLC;
      snapShot->activeBE = dvfsMemManager->ptrToShm->activeBEChunked + dvfsMemManager->ptrToShm->activeBELong;
    
    }

    if(snapShot->activeLC > 0) {
       return MAXMHZ; 
    }
    else if(snapShot->activeBE > 0){
        return DEFAULTMHZ;
    }

    return IDLE;
}

void dvfsManager::powerController(int target) {
    
    nvmlReturn_t result; 
    // timeElapsed = timeNow - LastChanged
    auto timeNow = std::chrono::steady_clock::now();
    
    if(target != currentPower && (timeNow - lastChanged) > std::chrono::seconds(2)){

       result = nvmlDeviceSetPowerManagementLimit(gpuDevice, (unsigned int)target);
       if(result == NVML_SUCCESS){
            nvmlDeviceGetPowerManagementLimit(gpuDevice, &currentPower);
            lastChanged = timeNow;
       }
       else {
            fprintf(stderr, "set limit failed: %s\n", nvmlErrorString(result));
       }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

}
