#include "../../include/memManager.h"
#include <cerrno>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

robustLockGuard::robustLockGuard(pthread_mutex_t& mutex){
    lock = &mutex;

    int result = pthread_mutex_lock(lock);
    if(result == EOWNERDEAD){
        pthread_mutex_consistent(lock);
    }
    else if (result != 0){
        throw std::system_error(result, std::generic_category(), "Failed to lock");
    }
}

memManager::memManager(const char* name){
    
    *shm_fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if(*shm_fd == -1){
        throw "Could not create/find shared memory";
    }
    ftruncate(*shm_fd, sizeof(policyData));
    ptrToShm = static_cast<policyData*>(mmap(0, sizeof(policyData), PROT_WRITE, MAP_SHARED, *shm_fd, 0));

    pthread_mutex_lock(&(ptrToShm->writeAllowed));
    if(!ptrToShm->isInitialized){
       
       pthread_mutexattr_t attr;
       pthread_mutexattr_init(&attr);
       pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
       pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
       pthread_mutex_init(&(ptrToShm->writeAllowed), &attr);
       pthread_mutexattr_destroy(&attr);

       ptrToShm->activeBELong          = 0;       
       ptrToShm->activeLC              = 0;           
       ptrToShm->activeBEChunked       = 0;   
       ptrToShm->BELongAdmitCount      = 0; 
       ptrToShm->BELongWaitCount       = 0; 
       ptrToShm->BELongWaitns          = 0; 
       ptrToShm->BELongThrottleCount   = 0; 
       
       ptrToShm->isInitialized = true;
       std::cout << "Process [" << getpid() << "] initialised the policy data\n";
    } else {
       std::cout << "Process [" << getpid() << "] connected to existing policy data\n";
    }
}


