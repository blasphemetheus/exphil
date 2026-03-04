# Build ThunderKittens Linear Scan NIF
#
# Two-stage build:
#   1. Compile CUDA kernel (kernel.cu → libthunderkittens_kernel.so)
#   2. Compile C NIF (thunderkittens_scan_nif.c → libthunderkittens_scan_nif.so)
#
# Usage:
#   make            # Build both kernel and NIF
#   make install    # Copy to priv/native/
#   make clean      # Remove build artifacts

# Erlang NIF include path
ERL_INCLUDE := $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)])' -s erlang halt)

# CUDA paths
CUDA_PATH ?= $(shell dirname $$(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc))/..
WSL_LIB ?= /usr/lib/wsl/lib

# CUDA architecture (sm_75 = T400/Turing, sm_80+ for actual ThunderKittens)
CUDA_ARCH ?= sm_75

# Compilers
NVCC := nvcc
CC := gcc

# Flags
NVCC_FLAGS := -Xcompiler -fPIC -O3 -arch=$(CUDA_ARCH) -c
CFLAGS := -shared -fPIC -O2 -Wall
CFLAGS += -I$(ERL_INCLUDE)
CFLAGS += -I$(CUDA_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib -L$(WSL_LIB)
LDFLAGS += -lcudart -ldl

# Targets
KERNEL_OBJ := kernel.o
KERNEL_SO := libthunderkittens_kernel.so
NIF_SO := libthunderkittens_scan_nif.so
PRIV_DIR := ../../priv/native

.PHONY: all clean install

all: $(NIF_SO)

# Step 1: Compile CUDA kernel to object file (nvcc -c avoids device link)
$(KERNEL_OBJ): kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Step 2: Link kernel object into shared library (gcc, avoids nvlink needing libcudadevrt)
$(KERNEL_SO): $(KERNEL_OBJ)
	$(CC) -shared -fPIC -o $@ $< -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib -L$(WSL_LIB) -lcudart -lstdc++

# Step 3: Compile C NIF (links against CUDA runtime for cudaMalloc/cudaMemcpy)
$(NIF_SO): thunderkittens_scan_nif.c $(KERNEL_SO)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(KERNEL_SO) $(NIF_SO)

install: $(NIF_SO) $(KERNEL_SO)
	mkdir -p $(PRIV_DIR)
	cp $(KERNEL_SO) $(PRIV_DIR)/
	cp $(NIF_SO) $(PRIV_DIR)/

# Build with ThunderKittens headers (requires sm_80+ GPU and TK source)
# Usage: make tk TK_PATH=/path/to/ThunderKittens
tk:
	@echo "Building with ThunderKittens headers (requires sm_80+ GPU)"
	$(NVCC) -shared -Xcompiler -fPIC -O3 -arch=sm_80 \
		-DTK_AVAILABLE -I$(TK_PATH)/include \
		-o $(KERNEL_SO) kernel.cu
	$(CC) $(CFLAGS) -o $(NIF_SO) thunderkittens_scan_nif.c $(LDFLAGS)
