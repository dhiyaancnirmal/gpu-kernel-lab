# Builds $(OUT) (default build/libgkl.so) with the CUDA runtime linked statically.
# NVCC defaults to nvcc on PATH.
NVCC ?= nvcc
OUT ?= build/libgkl.so
# Ampere (A100, A10), Ada (L4, L40S) and Hopper, plus PTX for newer GPUs.
ARCHS ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 \
         -gencode arch=compute_90,code=compute_90
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler -fPIC -Xptxas -warn-spills
SRCS := csrc/matmul.cu csrc/softmax.cu csrc/util.cu

$(OUT): $(SRCS) csrc/gkl.h
	@mkdir -p $(dir $(OUT))
	$(NVCC) $(ARCHS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -shared $(SRCS) -o $@

clean:
	rm -rf build

.PHONY: clean
