#include "../../include/memManager.h"
#include <sys/mman.h>
#include <unistd.h>

memManager::memManager(const char* name){
    
    *shm_fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if(*shm_fd == -1){
        throw "Could not create/find shared memory";
    }
    ftruncate(*shm_fd, sizeof(policyData));
    ptrToShm = mmap(0, sizeof(policyData), PROT_WRITE, MAP_SHARED, *shm_fd, 0);
}
