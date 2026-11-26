// Row-wise softmax over a rows x cols fp32 matrix, y = exp(x - max) / sum.
// Memory-bound: the ideal kernel reads x once and writes y once.

#include <cuda_runtime.h>
#include <math.h>

#include "gkl.h"

// ---------------------------------------------------------------------------
// Stage 0: naive. One thread per row, three passes over the row (max, sum,
// normalize). Neighbouring threads touch addresses `cols` apart, so loads are
// uncoalesced, and short of rows the GPU is mostly idle.
// ---------------------------------------------------------------------------
__global__ void softmax_naive(int rows, int cols, const float* x, float* y) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + (long long)r * cols;
  float* yr = y + (long long)r * cols;
  float m = -INFINITY;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
  float s = 0.f;
  for (int c = 0; c < cols; ++c) s += expf(xr[c] - m);
  for (int c = 0; c < cols; ++c) yr[c] = expf(xr[c] - m) / s;
}

constexpr int SM_THREADS = 256;
constexpr int SM_WARPS = SM_THREADS / 32;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// Block-wide reduction: shuffle within each warp, then warp 0 reduces the
// per-warp partials. Every thread gets the result.
template <bool IS_MAX>
__device__ float block_reduce(float v, float* scratch) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = IS_MAX ? warp_max(v) : warp_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < SM_WARPS ? scratch[lane] : (IS_MAX ? -INFINITY : 0.f);
    v = IS_MAX ? warp_max(v) : warp_sum(v);
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  float out = scratch[0];
  __syncthreads();  // scratch is reused by the next reduction
  return out;
}

// ---------------------------------------------------------------------------
// Stage 1: block per row. 256 threads stride across one row, so every load is
// coalesced, and max and sum come from warp-shuffle reductions. Still three
// passes: the row is read three times and written once.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(SM_THREADS)
softmax_block(int rows, int cols, const float* x, float* y) {
  __shared__ float scratch[SM_WARPS];
  const float* xr = x + (long long)blockIdx.x * cols;
  float* yr = y + (long long)blockIdx.x * cols;
  float m = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += SM_THREADS) m = fmaxf(m, xr[c]);
  m = block_reduce<true>(m, scratch);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += SM_THREADS) s += expf(xr[c] - m);
  s = block_reduce<false>(s, scratch);
  float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += SM_THREADS) yr[c] = expf(xr[c] - m) * inv;
}

// ---------------------------------------------------------------------------
// Stage 2: online softmax (Milakov & Gimelshein, 2018). Each thread keeps a
// running max m and a running sum d of exp(x - m); when m grows, d is
// rescaled by exp(m_old - m_new). Max and sum then come from ONE read of the
// row, plus one more read to write y: 2 reads + 1 write instead of 3 + 1.
// Loads are float4 when cols % 4 == 0 (keeps every row 16-byte aligned).
// ---------------------------------------------------------------------------
struct MD {
  float m, d;
};

__device__ __forceinline__ MD md_merge(MD a, MD b) {
  float m = fmaxf(a.m, b.m);
  // Both -inf means no finite value seen yet; exp(-inf - -inf) would be NaN.
  if (m == -INFINITY) return {m, 0.f};
  return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

__device__ __forceinline__ MD md_push(MD a, float v) { return md_merge(a, {v, 1.f}); }

__device__ MD block_reduce_md(MD v, MD* scratch) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    MD other = {__shfl_xor_sync(0xffffffff, v.m, o), __shfl_xor_sync(0xffffffff, v.d, o)};
    v = md_merge(v, other);
  }
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < SM_WARPS ? scratch[lane] : MD{-INFINITY, 0.f};
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      MD other = {__shfl_xor_sync(0xffffffff, v.m, o), __shfl_xor_sync(0xffffffff, v.d, o)};
      v = md_merge(v, other);
    }
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

__global__ void __launch_bounds__(SM_THREADS)
softmax_online(int rows, int cols, const float* x, float* y) {
  __shared__ MD scratch[SM_WARPS];
  const float* xr = x + (long long)blockIdx.x * cols;
  float* yr = y + (long long)blockIdx.x * cols;
  MD md = {-INFINITY, 0.f};
  const bool vec = (cols % 4 == 0);
  if (vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    for (int c = threadIdx.x; c < cols / 4; c += SM_THREADS) {
      float4 v = x4[c];
      float m4 = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
      if (m4 != -INFINITY) {
        float d4 = expf(v.x - m4) + expf(v.y - m4) + expf(v.z - m4) + expf(v.w - m4);
        md = md_merge(md, {m4, d4});
      }
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += SM_THREADS) md = md_push(md, xr[c]);
  }
  md = block_reduce_md(md, scratch);
  const float m = md.m, inv = 1.f / md.d;
  if (vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    float4* y4 = reinterpret_cast<float4*>(yr);
    for (int c = threadIdx.x; c < cols / 4; c += SM_THREADS) {
      float4 v = x4[c];
      y4[c] = make_float4(expf(v.x - m) * inv, expf(v.y - m) * inv, expf(v.z - m) * inv,
                          expf(v.w - m) * inv);
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += SM_THREADS) yr[c] = expf(xr[c] - m) * inv;
  }
}

extern "C" int gkl_softmax(int stage, int rows, int cols, const float* x, float* y,
                           cudaStream_t stream) {
  if (rows == 0 || cols == 0) return 0;
  switch (stage) {
    case 0:
      softmax_naive<<<(rows + 127) / 128, 128, 0, stream>>>(rows, cols, x, y);
      break;
    case 1:
      softmax_block<<<rows, SM_THREADS, 0, stream>>>(rows, cols, x, y);
      break;
    case 2:
      softmax_online<<<rows, SM_THREADS, 0, stream>>>(rows, cols, x, y);
      break;
    default:
      return GKL_ERR_BAD_STAGE;
  }
  return static_cast<int>(cudaGetLastError());
}
