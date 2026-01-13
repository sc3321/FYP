# ---------- toolchain ----------
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread

SRC_DIR := src
BUILD_DIR := build
SRC := $(SRC_DIR)/trace.cc

# ---------- HIP-CPU configuration ----------
# Default to a local checkout at ~/HIP-CPU. Override like:
#   make HIPCPU_HOME=/path/to/HIP-CPU
HIPCPU_HOME ?= $(HOME)/HIP-CPU

# Mode:
#  - local: use HIPCPU_HOME
#  - installed: use /usr/local
HIPCPU_MODE ?= local

ifeq ($(HIPCPU_MODE),installed)
  HIPCPU_INC := /usr/local/include
  HIPCPU_LIB := /usr/local/lib
else
  HIPCPU_INC := $(HIPCPU_HOME)/include

  # Try to locate the HIP-CPU runtime shared library directory (if it exists).
  # If it doesn't exist, we will build assuming header-only usage of the APIs.
  HIPCPU_LIB := $(shell find $(HIPCPU_HOME)/build -maxdepth 4 -type f -name "libhip_cpu_rt.so" -printf "%h\n" 2>/dev/null | head -n 1)
endif

# ---------- link flags (auto fallback if libhip_cpu_rt isn't present) ----------
# TBB + dl are needed by HIP-CPU.
LDLIBS_BASE := -ltbb -ldl

# If HIPCPU_LIB is empty (local mode and no lib found), don't try to link -lhip_cpu_rt.
ifeq ($(strip $(HIPCPU_LIB)),)
  HIPCPU_LDFLAGS :=
  HIPCPU_LDLIBS  := $(LDLIBS_BASE)
else
  HIPCPU_LDFLAGS := -L$(HIPCPU_LIB) -Wl,-rpath,$(HIPCPU_LIB)
  HIPCPU_LDLIBS  := -lhip_cpu_rt $(LDLIBS_BASE)
endif

INCLUDES := -I$(HIPCPU_INC)

# ---------- outputs ----------
BASELINE_EXE := $(BUILD_DIR)/baseline
ALLOC_EXE    := $(BUILD_DIR)/alloc
KERNELS_EXE  := $(BUILD_DIR)/kernels


# ---------- phony targets ----------
.PHONY: all clean info

all: $(BASELINE_EXE) $(ALLOC_EXE) $(KERNELS_EXE)

info:
	@echo "HIPCPU_MODE=$(HIPCPU_MODE)"
	@echo "HIPCPU_HOME=$(HIPCPU_HOME)"
	@echo "HIPCPU_INC=$(HIPCPU_INC)"
	@echo "HIPCPU_LIB=$(HIPCPU_LIB)"
	@echo "HIPCPU_LDFLAGS=$(HIPCPU_LDFLAGS)"
	@echo "HIPCPU_LDLIBS=$(HIPCPU_LDLIBS)"
	@echo "CXX=$(CXX)"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ---------- builds ----------
# Baseline stays minimal: no HIP include/lib needed.
$(BASELINE_EXE): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -DBASELINE

# Alloc variant: HIP-CPU include + (optional) runtime lib
$(ALLOC_EXE): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ -DALLOC $(HIPCPU_LDFLAGS) $(HIPCPU_LDLIBS)

# Kernels variant: HIP-CPU include + (optional) runtime lib
$(KERNELS_EXE): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ -DUSE_HIP -DMODE_KERNELS \
	  $(HIPCPU_LDFLAGS) $(HIPCPU_LDLIBS)

clean:
	rm -rf $(BUILD_DIR)

