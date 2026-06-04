UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),x86_64)
ARCH := x86
else ifeq ($(UNAME_M),aarch64)
ARCH := arm64
else
ARCH := x86
endif

LIBBPF_FLAGS := $(shell pkg-config --cflags --libs libbpf 2>/dev/null)
ifeq ($(strip $(LIBBPF_FLAGS)),)
LIBBPF_FLAGS := -lbpf -lelf -lz
endif

all: syscall_trace

syscall_trace.bpf.o: syscall_trace.bpf.c common.h vmlinux.h
	clang -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) -I. -c syscall_trace.bpf.c -o syscall_trace.bpf.o

syscall_trace.skel.h: syscall_trace.bpf.o
	bpftool gen skeleton syscall_trace.bpf.o > syscall_trace.skel.h

syscall_trace: syscall_trace.c common.h syscall_trace.skel.h
	cc -g -O2 -Wall -I. syscall_trace.c -o syscall_trace $(LIBBPF_FLAGS)

clean:
	rm -f syscall_trace syscall_trace.bpf.o syscall_trace.skel.h

.PHONY: all clean
