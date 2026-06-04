#include "../../include/gpuPhaseTypes.h"
#include "../../include/eventHandler.h"
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <atomic>
#include <threads.h>
#include <thread>
#include <chrono>

static const char* getShmName() {
    const char* n = std::getenv("GPU_PHASE_SHM_NAME");
    return (n && n[0]) ? n : "/sharedMemName";
}

using phase_id_t = uint64_t;
static std::atomic<uint64_t> nextPhaseId{1};

static std::string upper_string(std::string s) {
    for(char &c : s){
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

static policyMode parse_policy_mode(const std::string& policy) {

    std::string p = upper_string(policy);


    if (p == "NONE") {
        return policyMode::NONE;
    }

    if (p == "NAIVE") {
        return policyMode::NAIVE_THROTTLE;
    }

    if (p == "PROPER" || p == "CAP") {
        return policyMode::CAP;
    }

    return policyMode::UNK;
}

static policyMode readEnvPolicyMode(const char* name, policyMode defaultValue){
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0'){
        return defaultValue;
    }
    if(parse_policy_mode(raw) != policyMode::UNK){
        return parse_policy_mode(raw);
    }
    return defaultValue;
}


void phaseManager::setPolicyMode(){
    currentPolicyMode = readEnvPolicyMode("POLICY_MODE", policyMode::NONE);
}

workload_Class getPriority(const char* priority) {
    if (priority == nullptr) {
        return workload_Class::UNK;
    }

    if (strcmp(priority, "LC") == 0) {
        return workload_Class::LC;
    }

    if (strcmp(priority, "BE") == 0) {
        return workload_Class::BE;
    }

    if (strcmp(priority, "UNK") == 0) {
        return workload_Class::UNK;
    }

    return workload_Class::UNK;
}

granularity getGranularity(const char* granularity) {
    if (granularity == nullptr) {
        return granularity::UNK;
    }

    if (strcmp(granularity, "SHORT") == 0) {
        return granularity::SHORT;
    }

    if (strcmp(granularity, "LONG") == 0) {
        return granularity::LONG;
    }
    return granularity::UNK;
}

gpuPhase::gpuPhase(const char* inputSemanticIdentifier,workload_Class priority, granularity granularity, bool usePolicySupplied){
    semanticIdentifier = inputSemanticIdentifier;
    phaseMetadata.pid = getpid();
    usePolicy = usePolicySupplied;
    phaseMetadata.depth = 0;
    phaseMetadata.parentId = 0;
    phaseMetadata.tid = gettid();
    clock_gettime(CLOCK_MONOTONIC, &phaseMetadata.startTime);
    workloadClass = priority;
    workloadGranularity = granularity;
}

phaseManager::phaseManager(){
   void* rawWriter = (eventHandler*)std::malloc(sizeof(eventHandler));
   if(rawWriter == nullptr){
        throw "Could not allocate raw bytes for eventWriter";
   }
   phaseWriter = ::new (rawWriter) eventHandler();
   void* rawBytes = (memManager*)std::malloc(sizeof(memManager));
   if(rawBytes == nullptr){
        throw "Could not allocate raw bytes for memoryManager";
   }
   memoryManager = ::new (rawBytes) memManager(getShmName());
   void* rawBytesPolicy   = (policyManager*)std::malloc(sizeof(policyManager));
   if(rawBytesPolicy == nullptr){
        throw "Could not allocate raw bytes for policyManager";
   }
   setPolicyMode();
   policyManagerHandler = ::new (rawBytesPolicy) policyManager(*memoryManager);

   policyManagerHandler->readPolicyData("startup");

   startSamplingIfRequested();
}

void phaseManager::setPhaseData(gpuPhase& curPhase){
    curPhase.phaseMetadata.phaseId = {curPhase.phaseMetadata.pid, nextPhaseId.fetch_add(1, std::memory_order_relaxed)};
    clock_gettime(CLOCK_MONOTONIC, &curPhase.phaseMetadata.startTime);

}

void phaseManager::updatePhaseTable(gpuPhase& newPhase){

   if(!activePhases.curPhases.empty()){
       gpuPhase& ref = activePhases.curPhases.front();
       newPhase.phaseMetadata.parentId = ref.phaseMetadata.phaseId.second;
       newPhase.phaseMetadata.depth = ref.phaseMetadata.depth + 1;
   }
   activePhases.curPhases.insert(activePhases.curPhases.begin(), newPhase);
}

phaseID phaseManager::phaseBegin(const char* semanticIdentifier, workload_Class priority, granularity granularity, bool usePolicy){
    gpuPhase newPhase(semanticIdentifier, priority, granularity, usePolicy);
    policyManagerHandler->applyPolicy(newPhase, currentPolicyMode);
    setPhaseData(newPhase);
    updatePhaseTable(newPhase);
    phaseWriter->writeEvent(true, newPhase);
    return newPhase.phaseMetadata.phaseId;
}

void phaseManager::phaseEnd(phaseID idToEnd){
    gpuPhase* endPhase = nullptr;
    for(size_t i = 0; i < activePhases.curPhases.size(); ++i){
        if(activePhases.curPhases[i].phaseMetadata.phaseId == idToEnd){
          policyManagerHandler->endPDUpdate(activePhases.curPhases[i]);
          clock_gettime(CLOCK_MONOTONIC, &activePhases.curPhases[i].phaseMetadata.endTime);
          phaseWriter->writeEvent(false, activePhases.curPhases[i]);
          activePhases.curPhases.erase(activePhases.curPhases.begin() + i);
          return;
        }
    }
}

void phaseManager::cleanup(){
    stopSamplingIfRunning();

    if (policyManagerHandler != nullptr && memoryManager != nullptr && memoryManager->ptrToShm != nullptr) {
        policyManagerHandler->readPolicyData("shutdown");
    }

    munmap(memoryManager->ptrToShm, sizeof(policyData));
    shm_unlink(getShmName());
}


phaseGuard::phaseGuard(phaseManager& curManager, const char* semanticIdentifier, workload_Class wClass, granularity wGran, bool usePolicy){
    ptrToPhaseManager = &curManager;
    phase_id = ptrToPhaseManager->phaseBegin(semanticIdentifier, wClass, wGran, usePolicy);

}

phaseGuard::~phaseGuard(){
    if(ptrToPhaseManager){
        ptrToPhaseManager->phaseEnd(phase_id);
    }
}

void phaseManager::startSamplingIfRequested(){
    const char* sampleEnv = std::getenv("GPU_PHASE_POLICY_SAMPLE_MS");
    if (sampleEnv == nullptr || sampleEnv[0] == '\0') {
        return;
    }

    int sampleMs = std::atoi(sampleEnv);
    if (sampleMs <= 0) {
        return;
    }

    samplingStop.store(false, std::memory_order_release);

    samplingThread = std::thread([this, sampleMs]() {
        // Sampling labels carry a monotonic timestamp so the time series
        // can be reconstructed even if records from multiple processes
        // are interleaved in a shared log file.
        while (!samplingStop.load(std::memory_order_acquire)) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            char label[64];
            snprintf(label, sizeof(label), "sample_%ld.%09ld",
                     (long)ts.tv_sec, ts.tv_nsec);
            policyManagerHandler->readPolicyData(label);
            std::this_thread::sleep_for(std::chrono::milliseconds(sampleMs));
        }
    });
}

void phaseManager::stopSamplingIfRunning(){
    samplingStop.store(true, std::memory_order_release);
    if (samplingThread.joinable()) {
        samplingThread.join();
    }
}
