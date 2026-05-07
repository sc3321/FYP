#include "../../include/eventHandler.h"
#include <cstdio>
#include <cstring>

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

void eventHandler::writeEvent(gpuPhase& phase){
    char *fileName = (char*)std::malloc(16 * sizeof(char));
    if(fileName != NULL){
        snprintf(fileName, 16, "%d", (int)phase.phaseMetadata.pid);
    }
    fptr = fopen(fileName, "w"); 
    if(fptr == NULL){
       printf("File not opened\n");
       return;
    }

    char phaseData[512];
    char *threadId = (char*)std::malloc(64 * sizeof(char));
    char *semanticInfo = (char*)std::malloc(128 * sizeof(char));
    char *workloadClass = (char*)std::malloc(64 * sizeof(char));
    char *phaseId = (char*)std::malloc(128 * sizeof(char));

    snprintf(threadId, 64, "%d", (int)phase.phaseMetadata.tid);
    strcpy(workloadClass, workloadClasstoString(phase.workloadClass));
    strcpy(semanticInfo, phase.semanticIdentifier.c_str());
    snprintf(phaseId, 128, "%ld, %d", (long)phase.phaseMetadata.phaseId.first, phase.phaseMetadata.phaseId.second);

	snprintf(phaseData, sizeof(phaseData), "%s: %s, %s, %s\n", 
         threadId, 
         phaseId, 
         phase.semanticIdentifier.c_str(), 
         workloadClass);	
	
	fputs(phaseData, fptr);
    fclose(fptr);

    free(threadId);
    free(phaseId);
    free((char*)workloadClass);
    free(semanticInfo);
}
