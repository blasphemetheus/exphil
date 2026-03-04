# Makefile for Futhark Linear Scan NIF
#
# Build steps:
#   1. futhark cuda --library linear_scan.fut  → linear_scan.{c,h}
#   2. gcc -shared → futhark_scan_nif.so
#
# Usage:
#   make          # Build everything
#   make clean    # Remove generated files
#   make test     # Run Futhark's built-in tests

# Erlang NIF include path
ERL_INCLUDE := $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)])' -s erlang halt)

# CUDA paths (from shell.nix)
CUDA_PATH ?= $(shell dirname $$(which nvcc))/..

# Compiler flags
CC := gcc
CFLAGS := -shared -fPIC -O2 -Wall
CFLAGS += -I$(ERL_INCLUDE)
CFLAGS += -I$(CUDA_PATH)/include
# WSL2: libcuda.so is in /usr/lib/wsl/lib, not the CUDA toolkit
WSL_LIB ?= /usr/lib/wsl/lib
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib -L$(WSL_LIB) -lcuda -lnvrtc -lcudart

# Output
NIF_SO := futhark_scan_nif.so
PRIV_DIR := ../../priv/native

# Futhark compiler
FUTHARK := futhark

.PHONY: all clean test install

all: $(NIF_SO)

# Step 1: Compile Futhark to C library
linear_scan.c linear_scan.h: linear_scan.fut
	$(FUTHARK) cuda --library linear_scan.fut

# Step 2: Compile NIF shared library
$(NIF_SO): futhark_scan_nif.c linear_scan.c linear_scan.h
	$(CC) $(CFLAGS) -o $@ futhark_scan_nif.c linear_scan.c $(LDFLAGS)

# Install to priv/native/
install: $(NIF_SO)
	mkdir -p $(PRIV_DIR)
	cp $(NIF_SO) $(PRIV_DIR)/libfuthark_scan_nif.so

# Run Futhark's built-in tests (test blocks in .fut file)
test: linear_scan.fut
	$(FUTHARK) test --backend=cuda linear_scan.fut

# Also support C backend for testing without GPU
test-c: linear_scan.fut
	$(FUTHARK) test --backend=c linear_scan.fut

clean:
	rm -f linear_scan.c linear_scan.h $(NIF_SO)
	rm -rf linear_scan
