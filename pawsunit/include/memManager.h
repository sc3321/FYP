#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <stdexcept>
#include <string>



struct policyData {
   std::mutex   writeAllowed;
   int          activeBELong;
   int          activeLC;
   int          activeBEChunked;
   long         BELongWaitCount;
   long long    BELongWaitns;
   int          BELongAdmitCount;
   int          BELongThrottleCount;
};


class memManager{
    public:
        memManager(const char* name);
        
    private:

};








