#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <stdexcept>
#include <string>

struct policyData {
   int          activeBELong        = 0;
   int          activeLC            = 0;
   int          activeBEChunked     = 0;
   int          BELongAdmitCount    = 0;
   long         BELongWaitCount     = 0;
   long long    BELongWaitns        = 0;
   int          BELongThrottleCount = 0;
   std::mutex   writeAllowed           ;
};

class memManager{
    public:
        memManager(const char* name);
        int* shm_fd = nullptr;
        void* ptrToShm = nullptr;
        // add locking, mapping, sizing and modifying functions.
        // Generate a test program with read/write to the shared memory and test to see if it works
    private:

};

