# Makefile for Triton AOT Linear Scan NIF
#
# Build steps:
#   1. python3 compile_aot.py → .cubin + metadata header
#   2. gcc -shared → triton_scan_nif.so
#
# Usage:
#   make          # Build everything (requires Triton for AOT step)
#   make nif      # Build NIF only (cubin already compiled)
#   make clean    # Remove generated files
#   make install  # Copy to priv/native/

# Erlang NIF include path
ERL_INCLUDE := $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)])' -s erlang halt)

# CUDA paths
CUDA_PATH ?= $(shell dirname $$(which nvcc))/..

# Compiler flags
CC := gcc
CFLAGS := -shared -fPIC -O2 -Wall
CFLAGS += -I$(ERL_INCLUDE)
CFLAGS += -I$(CUDA_PATH)/include
CFLAGS += -I.
# WSL2: libcuda.so is in /usr/lib/wsl/lib, not the CUDA toolkit
WSL_LIB ?= /usr/lib/wsl/lib
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib -L$(WSL_LIB) -lcuda

# Output
NIF_SO := triton_scan_nif.so
PRIV_DIR := ../../priv/native

# Triton AOT compilation (requires Python + Triton at build time only)
PYTHON := python3

.PHONY: all nif clean install aot

all: aot nif

# Step 1: AOT compile Triton kernel to cubin
aot: linear_scan_kernel.cubin

linear_scan_kernel.cubin linear_scan_kernel_meta.h: linear_scan_kernel.py compile_aot.py
	$(PYTHON) compile_aot.py

# Step 2: Compile NIF shared library (only needs CUDA driver, not Triton)
nif: $(NIF_SO)

$(NIF_SO): triton_scan_nif.c linear_scan_kernel_meta.h
	$(CC) $(CFLAGS) -o $@ triton_scan_nif.c $(LDFLAGS)

# Install to priv/native/ (both .so and .cubin)
install: $(NIF_SO) linear_scan_kernel.cubin
	mkdir -p $(PRIV_DIR)
	cp $(NIF_SO) $(PRIV_DIR)/libtriton_scan_nif.so
	cp linear_scan_kernel.cubin $(PRIV_DIR)/

clean:
	rm -f linear_scan_kernel.cubin linear_scan_kernel_meta.h $(NIF_SO)
