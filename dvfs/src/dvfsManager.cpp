#include "../include/dvfsManager.h"
#include <iostream>
#include <nvml.h>
#include <stdlib.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <thread>

constexpr int DEFAULTMWS    = 225000;
constexpr int MAXMWS        = 300000;
constexpr int IDLE          = 175000;

static const char* getLogPath() {
    const char* p = std::getenv("DVFS_LOG_PATH");
    return (p && p[0]) ? p : "dvfs_log.csv";
}

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
   
   result = nvmlDeviceGetPowerManagementDefaultLimit(gpuDevice, &deviceDefaultPower);
   
   if(result != NVML_SUCCESS){
        std::cerr << "Failed to get GPU default power: " << nvmlErrorString(result) << std::endl;
   }

   result = nvmlDeviceGetPowerManagementLimit(gpuDevice, &currentPower);
 
   if(result != NVML_SUCCESS){
        std::cerr << "Failed to get GPU current power: " << nvmlErrorString(result) << std::endl;
   }

   logFile = std::fopen(getLogPath(), "w");
   if (logFile == nullptr) {
        std::cerr << "Warning: could not open DVFS log at " << getLogPath() << std::endl;
   } else {
       std::fprintf(logFile,
            "wall_ms,activeLC,activeBELong,activeBEChunked,"
            "targetMw,limitMw,changed,powerDrawMw,smClockMHz,tempC\n");
       std::fflush(logFile);
   }
}

unsigned int dvfsManager::classify() {
    
    {
      robustLockGuard lock(dvfsMemManager->ptrToShm->writeAllowed);
      snapShot->activeLC = dvfsMemManager->ptrToShm->activeLC;
      snapShot->activeBE = dvfsMemManager->ptrToShm->activeBELong;
    
    }

    if(snapShot->activeLC > 0) {
       return MAXMWS; 
    }
    else if(snapShot->activeBE > 0){
        return DEFAULTMWS;
    }

    return IDLE;
}

bool dvfsManager::powerController(unsigned int target) {
    
    nvmlReturn_t result; 
    // timeElapsed = timeNow - LastChanged
    auto timeNow = std::chrono::steady_clock::now();
    
    if((target != currentPower && (timeNow - lastChanged) > std::chrono::seconds(2)){

       result = nvmlDeviceSetPowerManagementLimit(gpuDevice, (unsigned int)target);
       if(result == NVML_SUCCESS){
            nvmlDeviceGetPowerManagementLimit(gpuDevice, &currentPower);
            lastChanged = timeNow;
       }
       else {
            fprintf(stderr, "set limit failed: %s\n", nvmlErrorString(result));
            return false;
       }
       return true;
    }
    return false;

}

void dvfsManager::logSample(unsigned int target, bool changed) {
    if (logFile == nullptr) return;

    unsigned int clockMHz = 0, powerDrawMw = 0, tempC = 0;
    nvmlDeviceGetClock(gpuDevice, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &clockMHz);
    nvmlDeviceGetPowerUsage(gpuDevice, &powerDrawMw);
    nvmlDeviceGetTemperature(gpuDevice, NVML_TEMPERATURE_GPU, &tempC);

    auto wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::fprintf(logFile, "%lld,%d,%d,%d,%u,%u,%d,%u,%u,%u\n",
        (long long)wallMs,
        snapShot.activeLC, snapShot.activeBELong, snapShot.activeBEChunked,
        target, currentPower, changed ? 1 : 0,
        powerDrawMw, clockMHz, tempC);
    std::fflush(logFile);
}



