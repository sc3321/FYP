#include <csignal>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <pthread.h>
#include <stdexcept>
#include <string>

class robustLockGuard {
    public:
        robustLockGuard(pthread_mutex_t& mutex);
        ~robustLockGuard(){
            pthread_mutex_unlock(lock);
        }
        robustLockGuard(const robustLockGuard&) = delete;
        robustLockGuard& operator=(const robustLockGuard&) = delete;
    private:
        pthread_mutex_t* lock;
};

struct policyData {
   int              activeBELong        ;
   int              activeLC            ;
   int              activeBEChunked     ;
   int              BELongAdmitCount    ;
   long             BELongWaitCount     ;
   long long        BELongWaitns        ;
   int              BELongThrottleCount ;
   bool             isInitialized       ;
   pthread_mutex_t  writeAllowed        ;
};

class memManager{
    public:
        memManager(const char* name);
        int*        shm_fd   = nullptr;
        policyData* ptrToShm = nullptr;
        // add locking, mapping, sizing and modifying functions.
        // Generate a test program with read/write to the shared memory and test to see if it works
    private:
    
};

