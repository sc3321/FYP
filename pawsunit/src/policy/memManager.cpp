#include "../../include/memManager.h"
#include <cerrno>
#include <fcntl.h>
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
    
    bool creator = false;
    shm_fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0666);
    if(shm_fd >= 0){
        creator = true;
    }
    else if(errno == EEXIST){
        shm_fd = shm_open(name, O_RDWR, 0666);
        if(shm_fd == -1){
            throw "Could not create/find shared memory";
            std::exit(1);
        }
    } else{
        throw "Could not create/find shared memory";
        std::exit(1);
    }
    if(creator){
        ftruncate(shm_fd, sizeof(policyData));
    }
    ptrToShm = static_cast<policyData*>(mmap(0, sizeof(policyData), PROT_WRITE, MAP_SHARED, shm_fd, 0));

    if(creator){
       
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
       
       ptrToShm->isInitialized.store(true, std::memory_order_release);
       std::cout << "Process [" << getpid() << "] initialised the policy data\n";
    } else {
        while(!ptrToShm->isInitialized.load(std::memory_order_acquire)){
            usleep(500);
        }
       std::cout << "Process [" << getpid() << "] connected to existing policy data\n";
    
    } 
     
    close(shm_fd);
}



