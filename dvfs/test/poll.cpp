#include <nvml.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <cerrno>

int isProcessAlive(pid_t pid){
    if (kill(pid, 0) == 0) {
        return 1;
    }
    return (errno == ESRCH) ? 0 : 1;
}


int main(int argc, char* argv[]){
    
    if(argc < 2){
        printf("insufficient argumetns\n");
        return 1;
    }

    pid_t processToWatch;
    processToWatch = atoi(argv[1]);
    
    nvmlInit();

    nvmlReturn_t result;
    nvmlDevice_t gpuDevice;

    result = nvmlDeviceGetHandleByIndex_v2(0, &gpuDevice);
    unsigned int clockSpeed = 0;
    unsigned int avgClockSpeed = 0;
    int count = 0;
    while(isProcessAlive(processToWatch)){
       result = nvmlDeviceGetClock(gpuDevice, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &clockSpeed);
       if(result != NVML_SUCCESS){
            break;
       }
       count++;
       avgClockSpeed += ((double)clockSpeed - avgClockSpeed) / count;

       sleep(1);
    }
    
    printf("Target process %d has terminated. Cleaning up NVML.\n", processToWatch);
    printf("The average clock speed was: %u", avgClockSpeed);
    nvmlShutdown();
    return 0;

    return 0;
}
