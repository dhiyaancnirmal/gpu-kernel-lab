// C ABI exported by libgkl.so. Python calls these through ctypes, passing raw
// device pointers from torch tensors and the current CUDA stream.
#pragma once
#include <cuda_runtime.h>

// Return codes: 0 is success, positive values are cudaError_t, negative values
// are ours.
#define GKL_ERR_BAD_STAGE (-1)
#define GKL_ERR_UNSUPPORTED (-2)

extern "C" {
int gkl_matmul(int stage, int M, int N, int K, const float* A, const float* B, float* C,
               cudaStream_t stream);
int gkl_softmax(int stage, int rows, int cols, const float* x, float* y, cudaStream_t stream);
int gkl_copy(const float* src, float* dst, long long n, cudaStream_t stream);
// Fills sm_count, clock_khz (peak SM clock), cc_major, cc_minor.
int gkl_device_info(int* sm_count, int* clock_khz, int* cc_major, int* cc_minor);
}
