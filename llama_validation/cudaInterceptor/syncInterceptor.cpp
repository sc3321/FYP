#include "phaseGuard/include/gpuPhaseTypes.h"
#include <cuda_runtime_api.h> 
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <driver_types.h>

phaseManager* llamaPhaseManager = nullptr;

__attribute__((constructor))
void llamaPhaseManagerSetup(){

    llamaPhaseManager = new phaseManager;
}

template <typename Function>
Function resolve_next(const char* name){
    
    dlerror();
    void* symbol = dlsym(RTLD_NEXT, name);

    const char* error = dlerror();
    if(error != nullptr){
        std::fprintf(stderr, "interceptor could not resolve %s: %s\n", name, error);
        std::abort();
    }
   
    return reinterpret_cast<Function>(symbol);

}

extern "C"
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    using realSignature = cudaError_t(*)(cudaStream_t);

    static realSignature cudaSyncSig = resolve_next<realSignature>("cudaStreamSynchronize");
    
    phaseID curPhaseId = llamaPhaseManager->phaseBegin("AUTO_CUDA_STREAM_SYNC", workload_Class::UNK, granularity::UNK, false);
    
    cudaError_t result;

    result = cudaSyncSig(stream);
    
    llamaPhaseManager->phaseEnd(curPhaseId);

    return result;

}


extern "C"
cudaError_t cudaDeviceSynchronize()
{
    using realSignature = cudaError_t(*)(void);

    static realSignature cudaDevSyncSig = resolve_next<realSignature>("cudaDeviceSynchronize");
    
    phaseID curPhaseId = llamaPhaseManager->phaseBegin("AUTO_CUDA_DEV_SYNC", workload_Class::UNK, granularity::UNK, false);
    
    cudaError_t result;

    result = cudaDevSyncSig();
    
    llamaPhaseManager->phaseEnd(curPhaseId);

    return result;

}



extern "C"
cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    using realSignature = cudaError_t(*)(cudaEvent_t);

    static realSignature cudaEventSyncSig = resolve_next<realSignature>("cudaEventSynchronize");
    
    phaseID curPhaseId = llamaPhaseManager->phaseBegin("AUTO_CUDA_EVENT_SYNC", workload_Class::UNK, granularity::UNK, false);
    
    cudaError_t result;

    result = cudaEventSyncSig(event);
    
    llamaPhaseManager->phaseEnd(curPhaseId);

    return result;

}


