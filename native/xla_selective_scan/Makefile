# Build XLA Custom Call kernel for selective scan
#
# Usage:
#   make            # Build shared library
#   make clean      # Remove build artifacts
#   make install    # Copy to priv/native/

NVCC = nvcc
CFLAGS = -shared -Xcompiler -fPIC -O3

# Detect CUDA compute capability
# Default to sm_120 for RTX 5090 (Blackwell)
CUDA_ARCH ?= sm_120

TARGET = libxla_selective_scan.so
SRC = selective_scan_kernel.cu
INSTALL_DIR = ../../priv/native

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -arch=$(CUDA_ARCH) -o $@ $<

clean:
	rm -f $(TARGET)

install: $(TARGET)
	mkdir -p $(INSTALL_DIR)
	cp $(TARGET) $(INSTALL_DIR)/

# Build for multiple architectures (slower but more portable)
portable:
	$(NVCC) $(CFLAGS) \
		-gencode arch=compute_80,code=sm_80 \
		-gencode arch=compute_86,code=sm_86 \
		-gencode arch=compute_89,code=sm_89 \
		-gencode arch=compute_120,code=sm_120 \
		-o $(TARGET) $(SRC)
