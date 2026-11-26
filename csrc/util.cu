// Helpers for the roofline: a bandwidth probe and device facts.

#include <cuda_runtime.h>

#include "gkl.h"

// Grid-stride float4 copy. Its throughput is the practical DRAM bandwidth
// ceiling used in the roofline chart. n must be a multiple of 4.
__global__ void copy_f4(const float4* __restrict__ src, float4* __restrict__ dst, long long n4) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n4;
       i += (long long)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

extern "C" int gkl_copy(const float* src, float* dst, long long n, cudaStream_t stream) {
  if (n % 4 != 0) return GKL_ERR_UNSUPPORTED;
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  copy_f4<<<sms * 8, 256, 0, stream>>>(reinterpret_cast<const float4*>(src),
                                       reinterpret_cast<float4*>(dst), n / 4);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int gkl_device_info(int* sm_count, int* clock_khz, int* cc_major, int* cc_minor) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return static_cast<int>(e);
  cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(clock_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(cc_major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(cc_minor, cudaDevAttrComputeCapabilityMinor, dev);
  return 0;
}
