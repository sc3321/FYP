#include "backend.h"

class Patterns
{
    public:

        void device_baseline(Backend& B,int bytes, int iterations, int* base_array);
        
        void device_alloc(Backend& B, int bytes, int iterations, int* base_array);
        void device_kernels(Backend& B, int bytes, int iterations, int* base_array);

};

