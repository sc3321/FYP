#include <stdlib.h>
#include <cstdio>

class Backend
{
    public:
        void dev_malloc();
        void dev_memcpyh2d();
        void dev_memcpyd2h();
        void dev_free();
        void dev_sync();
        void dev_addone();
};

Backend* make_backend();
