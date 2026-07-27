#include "backend.h"
#include <cstdio>

class Cpu_Backend : public Backend {
    public: 
      virtual const char* name() const;
      virtual void* dev_malloc(std::size_t bytes);
      virtual void  dev_free(void* p);
      virtual void  add_one(void* d, int N);
      virtual void  h2d(void* d, const void* h, std::size_t bytes);
      virtual void  d2h(void* h, const void* d, std::size_t bytes);
      virtual void  sync(); 
};
