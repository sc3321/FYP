
HIPCPU_ROOT ?= /home/sc/HIP-CPU

# The include dir must contain hip/hip_runtime.h
HIPCPU_INC  := $(HIPCPU_ROOT)/include

CXX ?= g++
CXXFLAGS += -std=c++17 -O2 -I$(HIPCPU_INC)

# Optional: helpful sanity check target for any experiment Makefile
.PHONY: check_hipcpu
check_hipcpu:
	@test -f "$(HIPCPU_INC)/hip/hip_runtime.h" || ( \
	  echo "ERROR: missing $(HIPCPU_INC)/hip/hip_runtime.h"; \
	  echo "Set HIPCPU_ROOT correctly (currently: $(HIPCPU_ROOT))"; \
	  exit 1 )
	@echo "OK: found hip_runtime.h at $(HIPCPU_INC)/hip/hip_runtime.h"


