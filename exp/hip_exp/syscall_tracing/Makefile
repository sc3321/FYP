# syscall_tracing/Makefile

# ---------- Project dirs ----------
ROOT     := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SRC_DIR  := $(ROOT)/src
INC_DIR  := $(ROOT)/include
BUILD    := $(ROOT)/build
OBJ_DIR  := $(BUILD)/obj
BIN_DIR  := $(BUILD)/bin

# ---------- Toolchains ----------
CXX      := g++
NVCC     := nvcc
# If you use hipcc for HIP, swap CXX -> hipcc for HIP target below.

# ---------- Flags ----------
CXXFLAGS   := -O2 -std=c++17 -I$(INC_DIR) -Wall -Wextra
NVCCFLAGS  := -O2 -std=c++17 -I$(INC_DIR)
LDFLAGS    :=

CUDA_LIBS  := -lcudart

# ---------- Sources ----------
COMMON_SRC := \
  $(SRC_DIR)/main.cc \
  $(SRC_DIR)/harness.cc \
  $(SRC_DIR)/patterns.cc

CPU_BACKEND_SRC  := $(SRC_DIR)/backend/cpu.cc
# HIP_BACKEND_SRC  := $(SRC_DIR)/backends/hip.cc
CUDA_BACKEND_SRC := $(SRC_DIR)/backend/cuda.cu

# ---------- Objects (in build/obj, preserving paths) ----------
# turn /abs/path/src/foo.cc into build/obj/src/foo.o
COMMON_OBJ := $(patsubst $(ROOT)/%.cc,$(OBJ_DIR)/%.o,$(COMMON_SRC))

CPU_BACKEND_OBJ  := $(patsubst $(ROOT)/%.cc,$(OBJ_DIR)/%.o,$(CPU_BACKEND_SRC))
# HIP_BACKEND_OBJ  := $(patsubst $(ROOT)/%.cc,$(OBJ_DIR)/%.o,$(HIP_BACKEND_SRC))
CUDA_BACKEND_OBJ := $(patsubst $(ROOT)/%.cu,$(OBJ_DIR)/%.o,$(CUDA_BACKEND_SRC))

# ---------- Binaries ----------
TB_CPU  := $(BIN_DIR)/tb_cpu
# TB_HIP  := $(BIN_DIR)/tb_hip
TB_CUDA := $(BIN_DIR)/tb_cuda

.PHONY: all clean dirs cpu hip cuda

all: dirs $(TB_CPU) $(TB_CUDA)
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# ============================================================
# Link rules
# ============================================================

$(TB_CPU):  $(COMMON_OBJ) $(CPU_BACKEND_OBJ) | dirs
	$(CXX) $^ $(LDFLAGS) -o $@

# $(TB_HIP):  $(COMMON_OBJ) $(HIP_BACKEND_OBJ) | dirs
#	$(CXX) $^ $(LDFLAGS) -o $@

$(TB_CUDA): $(COMMON_OBJ) $(CUDA_BACKEND_OBJ) | dirs
	$(NVCC) $^ $(CUDA_LIBS) -o $@

# Convenience targets
cpu:  dirs $(TB_CPU)
# hip:  dirs $(TB_HIP)
cuda: dirs $(TB_CUDA)

# ============================================================
# Compile rules (objects in build/obj/...)
# Each rule ensures the parent directory exists.
# ============================================================

# C++ sources
$(OBJ_DIR)/%.o: $(ROOT)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA sources
$(OBJ_DIR)/%.o: $(ROOT)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ============================================================
# Clean
# ============================================================

clean:
	rm -rf $(BUILD)

