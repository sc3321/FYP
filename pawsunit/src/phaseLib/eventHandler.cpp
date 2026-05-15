#include "../../include/eventHandler.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

const char* workloadClasstoString(workload_Class wc){
    switch (wc){
        case workload_Class::BE:
            return "BE";
        case workload_Class::LC:
            return "LC";
        case workload_Class::UNK:
            return "UNK";
        default:
            return "INVALID";
    }
} 

void eventHandler::writeEvent(bool begin, gpuPhase& phase){
    const char* logDir = std::getenv("GPU_PHASE_LOG_DIR");
    if (logDir == NULL || std::strlen(logDir) == 0) {
        logDir = ".";
    }

    char *fileName = (char*)std::malloc(512 * sizeof(char));
    if(fileName != NULL){
        snprintf(fileName, 512, "%s/%d", logDir, (int)phase.phaseMetadata.pid);
    }
    fptr = fopen(fileName, "a"); 
    if(fptr == NULL){
       printf("File not opened\n");
       free(fileName);
       return;
    }

    char phaseData[720];
    char *threadId = (char*)std::malloc(64 * sizeof(char));
    char *semanticInfo = (char*)std::malloc(128 * sizeof(char));
    char *workloadClass = (char*)std::malloc(64 * sizeof(char));
    char *phaseId = (char*)std::malloc(128 * sizeof(char));
    

    snprintf(threadId, 64, "%d", (int)phase.phaseMetadata.tid);
    strcpy(workloadClass, workloadClasstoString(phase.workloadClass));
    strcpy(semanticInfo, phase.semanticIdentifier.c_str());
    snprintf(phaseId, 128, "%ld, %d", (long)phase.phaseMetadata.phaseId.first, phase.phaseMetadata.phaseId.second);

    if(begin){
        snprintf(phaseData, sizeof(phaseData),"Event type = BEGIN: PhaseId:[%s],Thread Id: %s, parent_id: %d, depth: %d,  Timestamp: %ld s %ld ns, phase type: (%s), workload class: %s\n", 
         phaseId, 
         threadId,
         phase.phaseMetadata.parentId,
         phase.phaseMetadata.depth,
         phase.phaseMetadata.startTime.tv_sec,
         phase.phaseMetadata.startTime.tv_nsec,
         phase.semanticIdentifier.c_str(), 
         workloadClass);
    }
    else{
        snprintf(phaseData, sizeof(phaseData),"Event type = END: PhaseId:[%s],Thread Id: %s, parent_id: %d, depth: %d,  Timestamp: %ld s %ld ns, phase type: (%s), workload class: %s\n", 
         phaseId, 
         threadId,
         phase.phaseMetadata.parentId,
         phase.phaseMetadata.depth,
         phase.phaseMetadata.endTime.tv_sec,
         phase.phaseMetadata.endTime.tv_nsec,
         phase.semanticIdentifier.c_str(), 
         workloadClass);
    }
	fputs(phaseData, fptr);
    fclose(fptr);

    free(fileName);
    free(threadId);
    free(phaseId);
    free((char*)workloadClass);
    free(semanticInfo);
}
